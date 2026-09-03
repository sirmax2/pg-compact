"""Command line interface.

Flags follow the same conventions as psql/libpq (-h/-p/-U/-d, PG*
environment variable fallbacks, .pgpass support via libpq itself) so
anyone who has used psql, pg_dump, or pg_restore already knows the
connection options. Everything except the target table is optional and
has a sane default - see the README for the full option reference.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from pg_compact import __version__, db, reindex
from pg_compact.compactor import CompactionConfig, Outcome, compact_table
from pg_compact.logging_utils import Logger, fmt_bytes, fmt_size_change, verbosity_from_flags
from pg_compact.ui import RichUI, should_use_rich_ui


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pg-compact",
        description=(
            "Reduce PostgreSQL table and TOAST bloat online, without heavy locks, "
            "by relocating rows out of tail pages and letting VACUUM truncate them."
        ),
        add_help=False,  # -h is used for --host (psql convention); only --help remains
    )
    parser.add_argument("--help", action="help", help="show this help message and exit")
    parser.add_argument("-V", "--version", action="version", version=f"pg-compact {__version__}")

    conn_group = parser.add_argument_group("Connection options")
    conn_group.add_argument("-h", "--host", metavar="HOST",
                            help="database server host or socket directory (default: $PGHOST, or libpq default)")
    conn_group.add_argument("-p", "--port", metavar="PORT", type=int,
                            help="database server port (default: $PGPORT, or 5432)")
    conn_group.add_argument("-U", "--user", metavar="USER", help="database user name (default: $PGUSER, or OS user)")
    conn_group.add_argument("-W", "--password", metavar="PASSWORD",
                            help="database password (prefer $PGPASSWORD or ~/.pgpass instead)")
    conn_group.add_argument("-d", "--dbname", metavar="DBNAME",
                            help="database name to connect to (default: $PGDATABASE)")

    target_group = parser.add_argument_group("Target options")
    target_group.add_argument(
        "-t",
        "--table",
        metavar="[SCHEMA.]TABLE",
        required=True,
        help="table to compact, optionally schema-qualified (default schema: public)",
    )

    behavior_group = parser.add_argument_group("Behavior options")
    behavior_group.add_argument(
        "-f", "--force",
        action="store_true",
        help="compact even if the table is below the minimum bloat/size thresholds",
    )
    behavior_group.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="report the bloat estimate and planned action without modifying the table",
    )
    behavior_group.add_argument(
        "--no-reindex",
        action="store_true",
        help="skip REINDEX TABLE CONCURRENTLY after compaction",
    )
    behavior_group.add_argument(
        "--no-initial-vacuum",
        action="store_true",
        help="skip the VACUUM run before measuring the table (default: on)",
    )
    behavior_group.add_argument(
        "--min-compact-percent",
        metavar="PERCENT",
        type=float,
        default=10.0,
        help="minimum estimated reclaimable percent required to proceed (default: 10.0)",
    )
    behavior_group.add_argument(
        "--min-compact-pages",
        metavar="PAGES",
        type=int,
        default=10,
        help="minimum table size in pages required to proceed (default: 10)",
    )
    behavior_group.add_argument(
        "--no-throttle",
        action="store_true",
        help="never throttle. By default throttling is adaptive - it pauses "
             "only when a rewrite runs markedly slower per row than the recent "
             "average (lock contention, I/O spike); a normal run is unaffected.",
    )
    behavior_group.add_argument(
        "--throttle-delay-s",
        metavar="SEC",
        type=float,
        default=1.5,
        help="maximum pause after a slow rewrite when throttling; the actual "
             "pause scales with how long the rewrite took, up to this cap "
             "(default: 1.5)",
    )

    output_group = parser.add_argument_group("Output options")
    verbosity = output_group.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true", help="show detailed progress messages")
    verbosity.add_argument("-q", "--quiet", action="store_true", help="show only errors and the final result")
    interactivity = output_group.add_mutually_exclusive_group()
    interactivity.add_argument(
        "--interactive",
        dest="interactive",
        action="store_true",
        default=None,
        help="show a startup banner, live progress bar, and summary table (default: auto-detect a terminal)",
    )
    interactivity.add_argument(
        "--no-interactive",
        dest="interactive",
        action="store_false",
        help="plain line-oriented output only, no banner/progress bar/summary table",
    )

    safety_group = parser.add_argument_group("Safety options")
    safety_group.add_argument(
        "--min-free-disk-mb",
        metavar="MB",
        type=int,
        default=10240,
        help="pause between rounds if free disk space drops below this (default: 10240)",
    )
    safety_group.add_argument(
        "--no-disk-check",
        action="store_true",
        help="skip the disk space guard entirely",
    )
    safety_group.add_argument(
        "--lock-timeout-ms",
        metavar="MS",
        type=int,
        default=5000,
        help="max time to wait for a row lock before backing off and retrying (default: 5000)",
    )
    safety_group.add_argument(
        "--no-toast-compact",
        action="store_true",
        help="skip rewriting TOASTed columns; only compact the main heap",
    )

    return parser


def parse_table_arg(value: str) -> tuple[str, str]:
    if "." in value:
        schema, table = value.split(".", 1)
        if not schema or not table:
            raise argparse.ArgumentTypeError(
                f"Invalid --table value {value!r}: expected [schema.]table with non-empty parts."
            )
        return schema, table
    return "public", value


def main(argv: list | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    log = Logger(verbosity_from_flags(args.verbose, args.quiet))

    try:
        schema, table = parse_table_arg(args.table)
    except argparse.ArgumentTypeError as exc:
        log("error", str(exc))
        return 2

    conn_params = db.ConnectionParams.from_args_and_env(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        dbname=args.dbname,
    )

    try:
        conn = db.connect(conn_params)
    except Exception as exc:  # noqa: BLE001 - surface any connection failure plainly
        log("error", f"Could not connect to the database: {exc}")
        return 1

    try:
        try:
            db.check_min_version(conn)
        except db.UnsupportedServerVersionError as exc:
            log("error", str(exc))
            return 1

        try:
            db.check_required_extensions(conn)
        except db.MissingExtensionError as exc:
            log("error", str(exc))
            return 1

        config = CompactionConfig(
            force=args.force,
            dry_run=args.dry_run,
            min_compact_pages=args.min_compact_pages,
            min_compact_percent=args.min_compact_percent,
            throttle=not args.no_throttle,
            throttle_delay_s=args.throttle_delay_s,
            initial_vacuum=not args.no_initial_vacuum,
            reindex=not args.no_reindex,
            min_free_disk_mb=args.min_free_disk_mb,
            disk_check=not args.no_disk_check,
            lock_timeout_ms=args.lock_timeout_ms,
            toast_compact=not args.no_toast_compact,
        )

        use_rich = should_use_rich_ui(args.interactive)
        ui = RichUI(verbose=args.verbose) if use_rich else None

        if ui is not None:
            try:
                from pg_compact.stats import get_bloat_stats, get_toast_bloat_stats

                # Prime the FSM before measuring for the banner, otherwise
                # the banner reflects a pre-VACUUM snapshot: dead tuples left
                # by prior activity aren't yet recorded as free space, so the
                # heap size and reclaimable estimate disagree with what the
                # compaction (which VACUUMs first) actually sees.  Do it here
                # once and tell compact_table not to repeat it.
                if config.initial_vacuum and not config.dry_run:
                    log("info", "Running initial VACUUM...")
                    db.vacuum(conn, schema, table)
                    config = replace(config, initial_vacuum=False)

                size = db.get_size_stats(conn, schema, table)
                bloat = get_bloat_stats(conn, schema, table)
                toast_bloat = get_toast_bloat_stats(conn, schema, table)
                toast_reclaimable = toast_bloat.reclaimable_bytes if toast_bloat else 0
                total_target = size.total_bytes - bloat.reclaimable_bytes - toast_reclaimable
                total_reclaimable_pct = (
                    100.0 * (size.total_bytes - total_target) / size.total_bytes
                ) if size.total_bytes else 0.0

                ui.show_banner(
                    schema, table, str(db.get_server_version(conn)), size.total_bytes,
                    total_target, total_reclaimable_pct, "fsm",
                    heap_bytes=size.table_bytes, toast_bytes=size.toast_bytes,
                    indexes_bytes=size.indexes_bytes,
                    bloat=bloat, toast_bloat=toast_bloat,
                )
            except Exception:  # noqa: BLE001 - the banner is a nice-to-have, never block the real run on it
                pass
            if not config.dry_run:
                ui.start_progress()

        # When the rich UI is active, route log messages through its
        # console so they render above the live progress bar cleanly
        # instead of colliding with it on raw stderr.
        effective_log = ui.log if ui is not None else log

        def progress_cb(update):
            if ui is not None:
                ui.update_progress(update)

        effective_log("info", f'Starting compaction of "{schema}"."{table}"...')
        try:
            result = compact_table(conn, schema, table, config, effective_log, progress_cb)
        except KeyboardInterrupt:
            if ui is not None:
                ui.stop_progress(aborted=True)
            _report_interrupt(log)
            return 130
        except Exception as exc:  # noqa: BLE001 - report and exit non-zero, never swallow
            if ui is not None:
                ui.stop_progress(aborted=True)
            effective_log("error", f"Compaction failed: {exc}")
            return 1

        if ui is not None:
            ui.stop_progress()
            ui.show_summary(result)
        else:
            _report_result(result, log)
        return 0 if result.outcome in _SUCCESSFUL_OUTCOMES else 1
    finally:
        conn.close()


_SUCCESSFUL_OUTCOMES = {
    Outcome.COMPLETED,
    Outcome.SKIPPED_EMPTY,
    Outcome.SKIPPED_BELOW_THRESHOLD,
}


def _report_interrupt(log: Logger) -> None:
    if reindex.is_reindex_active:
        log(
            "warning",
            "Interrupted during REINDEX CONCURRENTLY. Any invalid index left behind is "
            "harmless and is dropped automatically on the next run.",
        )
    else:
        log("warning", "Interrupted; exiting.")


def _report_result(result, log: Logger) -> None:
    if result.message:
        level = "warning" if result.outcome != Outcome.COMPLETED else "info"
        log(level, result.message)

    if result.size_before and result.size_after:
        before = result.size_before
        after = result.size_after
        parts = [f"heap {fmt_bytes(before.table_bytes)} -> {fmt_bytes(after.table_bytes)}"]
        if before.toast_bytes or after.toast_bytes:
            parts.append(f"TOAST {fmt_bytes(before.toast_bytes)} -> {fmt_bytes(after.toast_bytes)}")
        parts.append(f"indexes {fmt_bytes(before.indexes_bytes)} -> {fmt_bytes(after.indexes_bytes)}")
        log(
            "always",
            f'"{result.schema}"."{result.table}": '
            f"{fmt_bytes(before.total_bytes)} -> {fmt_bytes(after.total_bytes)} total "
            f"({fmt_size_change(before.total_bytes, after.total_bytes)}) [{', '.join(parts)}].",
        )
    elif result.size_before:
        log(
            "always",
            f'"{result.schema}"."{result.table}": '
            f"{fmt_bytes(result.size_before.total_bytes)}, {result.size_before.page_count} heap pages.",
        )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
