"""Interactive terminal experience, built on top of the plain logger.

Everything in this module is additive and optional:

- In a real terminal (an interactive TTY on both stdin and stdout, per
  ``sys.stdin.isatty()``/``sys.stdout.isatty()``), and with the ``rich``
  package installed, pg-compact shows a startup banner with context and a
  live-updating progress display, then a summary table at the end.
- Piped output, redirected files, cron/systemd, or ``rich`` simply not
  being installed, all fall back to the existing plain line-oriented
  ``Logger`` with no banner or live display - nothing about the plain
  path changes. ``--interactive``/``--no-interactive`` let a user override
  the automatic detection either way (e.g. to force the rich output into
  a captured terminal session, or to suppress it in an unusual TTY-backed
  automation setup).

No confirmation prompt is shown in either mode - this is informational
and progress reporting only, never a gate the run waits on.
"""

from __future__ import annotations

import sys

from pg_compact.logging_utils import fmt_bytes, fmt_duration, fmt_size_change

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskID,
        TextColumn,
    )
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:  # rich is an optional dependency - see pyproject.toml
    RICH_AVAILABLE = False


def is_interactive_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def should_use_rich_ui(interactive_flag: bool | None) -> bool:
    """Resolve the effective choice from --interactive/--no-interactive and auto-detection.

    interactive_flag is True/False if the user passed an explicit CLI
    flag, or None to auto-detect from the terminal. Rich's absence always
    wins over any flag, since there is nothing to render with.
    """
    if not RICH_AVAILABLE:
        return False
    if interactive_flag is not None:
        return interactive_flag
    return is_interactive_terminal()


class RichUI:
    """Groups the banner/progress/summary pieces so cli.py has one object to drive.

    Only instantiate this after should_use_rich_ui() returns True - it
    assumes rich is importable.
    """

    def __init__(self, verbose: bool = False) -> None:
        self.console = Console(stderr=True)
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        self._verbose = verbose
        # Live display wrapping the fragmentation map + progress bar.
        self._live: Live | None = None
        self._frag_cells: list[str] | None = None
        self._frag_label: str | None = None

    def show_banner(
        self,
        schema: str,
        table: str,
        server_version: str,
        current_bytes: int,
        target_bytes: int | None,
        free_percent: float | None,
        precision: str | None,
        toast_note: str | None = None,
        heap_bytes: int | None = None,
        toast_bytes: int | None = None,
        indexes_bytes: int | None = None,
        alignment_note: str | None = None,  # kept for back-compat, ignored when bloat is set
        bloat: object | None = None,  # BloatStats — typed as object to avoid circular import
        toast_bloat: object | None = None,  # BloatStats for TOAST
    ) -> None:
        lines = [f"[bold]Target:[/bold] {schema}.{table}  (PostgreSQL {server_version})", ""]
        lines.append(f"Total size:    {fmt_bytes(current_bytes)}")
        breakdown = []
        if heap_bytes is not None:
            breakdown.append(f"heap {fmt_bytes(heap_bytes)}")
        if toast_bytes and toast_bytes > 0:
            breakdown.append(f"TOAST {fmt_bytes(toast_bytes)}")
        if indexes_bytes is not None:
            breakdown.append(f"indexes {fmt_bytes(indexes_bytes)}")
        if breakdown:
            lines.append(f"               ({', '.join(breakdown)})")

        # Detailed breakdown split by WHO can reclaim it.
        has_breakdown = (
            bloat is not None
            and heap_bytes
            and (getattr(bloat, "reclaimable_bytes", 0) or 0) > 0
        )
        if has_breakdown and heap_bytes:
            b = bloat
            _free = getattr(b, "free_bytes", 0) or 0
            _online = getattr(b, "online_reclaimable_bytes", 0) or 0
            _align = getattr(b, "alignment_waste_bytes", 0) or 0
            _reclaim = getattr(b, "reclaimable_bytes", 0) or 0
            _live = max(heap_bytes - _free, 0)

            def _val(v: int) -> str:
                return f"{fmt_bytes(v):>10}"

            def _pct(v: float) -> str:
                return f"{v:>5.1f}%"

            def _pct_of(v: int) -> str:
                return _pct(100.0 * v / heap_bytes) if heap_bytes else _pct(0.0)

            # Top-level Live data / Unusable padding / Bloat sum to 100%.
            # Bloat is itemised into two indented sub-lines that sum to it:
            # what pg-compact frees now (online) and what only a full rewrite
            # frees — so "can free now" is never confused with the total.
            _vf_only = max(_reclaim - _online, 0)
            lines.append("")
            lines.append(f"[bold]Heap[/bold] ({fmt_bytes(heap_bytes)}):")
            lines.append(f"  Live data               {_val(_live)}  {_pct_of(_live)}  "
                         "[dim]rows in use[/dim]")
            lines.append(f"  Unusable padding        {_val(_align)}  {_pct_of(_align)}  "
                         "[dim]never reclaimable[/dim]")
            lines.append(f"  Bloat                   {_val(_reclaim)}  {_pct_of(_reclaim)}")
            lines.append(f"    [green]pg-compact frees now[/green]  {_val(_online)}  {_pct_of(_online)}")
            lines.append(f"    needs VACUUM FULL     {_val(_vf_only)}  {_pct_of(_vf_only)}")

            # TOAST line in the same breakdown style.
            if toast_bloat is not None and toast_bytes and toast_bytes > 0:
                t_free = getattr(toast_bloat, "free_bytes", 0) or 0
                t_online = getattr(toast_bloat, "online_reclaimable_bytes", 0) or 0
                t_reclaim = getattr(toast_bloat, "reclaimable_bytes", 0) or 0
                t_waste = getattr(toast_bloat, "alignment_waste_bytes", 0) or 0
                t_live = max(toast_bytes - t_free, 0)

                _tvf_only = max(t_reclaim - t_online, 0)

                def _tpct(v: int) -> str:
                    return _pct(100.0 * v / toast_bytes) if toast_bytes else _pct(0.0)

                lines.append(f"[bold]TOAST[/bold] ({fmt_bytes(toast_bytes)}):")
                lines.append(f"  Live data               {_val(t_live)}  {_tpct(t_live)}  "
                             "[dim]values in use[/dim]")
                lines.append(f"  Unusable padding        {_val(t_waste)}  {_tpct(t_waste)}  "
                             "[dim]never reclaimable[/dim]")
                lines.append(f"  Bloat                   {_val(t_reclaim)}  {_tpct(t_reclaim)}")
                lines.append(f"    [green]pg-compact frees now[/green]  {_val(t_online)}  {_tpct(t_online)}")
                lines.append(f"    needs VACUUM FULL     {_val(_tvf_only)}  {_tpct(_tvf_only)}")

            lines.append("[dim]\"frees now\" is an upper bound; actual shown as before -> after.[/dim]")
        elif target_bytes is not None and free_percent is not None:
            lines.append(
                f"Target size:   {fmt_bytes(target_bytes)}  "
                f"({free_percent:.1f}% reclaimable)"
            )

        if toast_note:
            lines.append(toast_note)
        lines.append("")
        lines.append(
            "[dim]Relocates rows via no-op UPDATEs (WAL), VACUUMs between rounds, then "
            "REINDEX CONCURRENTLY. Safe to interrupt.[/dim]"
        )
        self.console.print(Panel("\n".join(lines), title="pg-compact", border_style="cyan"))

    # Map cell code -> (glyph, rich style).  Mirrors a disk defragmenter's
    # block map: each cell is a range of the relation's pages.
    _FRAG_GLYPHS = {
        "L": ("█", "green"),        # live / packed data
        "V": ("▓", "yellow"),       # free but sub-unit holes -> needs VACUUM FULL
        "R": ("░", "cyan"),         # free in relocatable holes -> reclaimable now
        "E": ("·", "dim"),          # (near-)empty -> truncatable tail
    }

    def _frag_renderable(self):
        """Build the fragmentation-map line (a Text) or None if no map yet.

        Just the coloured block bar - no header or legend (the colour meaning
        is documented in the README).  Keeping it to a single line also avoids
        reflowing multiple lines on every refresh, which is what made the map
        flicker.
        """
        if not self._frag_cells:
            return None
        bar = Text("  ")
        for code in self._frag_cells:
            glyph, style = self._FRAG_GLYPHS.get(code, ("?", ""))
            bar.append(glyph, style=style)
        return bar

    def _render_group(self):
        """The Live renderable: fragmentation map above the progress bar."""
        frag = self._frag_renderable()
        if frag is not None and self._progress is not None:
            return Group(frag, self._progress)
        return self._progress if self._progress is not None else Text("")

    def start_progress(self) -> None:
        # ETA/rate come from the engine (real page progress) and are baked
        # into {detail}; we deliberately omit rich's TimeRemainingColumn,
        # which extrapolates from the bar's fill rate and disagrees with the
        # engine's estimate (and uses a different H:MM:SS format).
        # A SpinnerColumn animates even when the engine is quiet (e.g. the
        # long chunk-map build), so the user can tell it's still working.
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TextColumn("{task.fields[detail]}"),
            TextColumn("{task.fields[status]}"),
            console=self.console,
            transient=False,
        )
        self._task_id = self._progress.add_task("starting...", total=100, detail="", status="")
        # Live pulls the renderable itself each tick via get_renderable (no
        # manual update() calls that would double-render).  refresh_per_second
        # is kept modest: high enough for a smooth spinner, low enough to
        # limit full-region repaints.  On terminals that support it, rich
        # wraps each repaint in a synchronized-output sequence so the frame
        # is swapped atomically (no visible tearing/flicker); the moderate
        # rate keeps it acceptable elsewhere.
        self._live = Live(
            console=self.console,
            get_renderable=self._render_group,
            auto_refresh=True,
            refresh_per_second=8,
            transient=False,
        )
        self._live.start()

    def update_progress(self, update) -> None:
        if self._progress is None or self._task_id is None:
            return
        # A fresh fragmentation map (attached after each VACUUM) replaces the
        # previous one; None means "keep showing the last map".
        if getattr(update, "frag_map", None) is not None:
            self._frag_cells = update.frag_map
            self._frag_label = getattr(update, "frag_map_label", None) or self._frag_label
        total = update.pages_total or 1
        percent = min(100.0, 100.0 * update.pages_done / total)

        parts: list[str] = []

        # 5.5 GB -> 5.1 GB (target ~4.8 GB) | disk 42 GB
        if update.current_size_bytes is not None:
            size_str = ""
            if update.before_size_bytes is not None:
                size_str = f"{fmt_bytes(update.before_size_bytes)} -> "
            size_str += fmt_bytes(update.current_size_bytes)
            if update.target_size_bytes is not None:
                size_str += f" (target ~{fmt_bytes(update.target_size_bytes)})"
            parts.append(size_str)

        # Rate and ETA come from the engine's real page-progress estimate,
        # formatted the same way as the flat status log ('1.5h', '3m').
        if update.mb_per_s is not None:
            parts.append(f"{update.mb_per_s:.1f} MB/s")
        if update.eta_seconds is not None:
            parts.append(f"ETA {fmt_duration(update.eta_seconds)}")

        if update.free_disk_bytes is not None:
            free_str = fmt_bytes(update.free_disk_bytes)
            if update.min_free_disk_bytes is not None:
                if update.free_disk_bytes < update.min_free_disk_bytes:
                    free_str = f"[bold red]free {free_str}[/]"
                elif update.free_disk_bytes < update.min_free_disk_bytes * 2:
                    free_str = f"[yellow]free {free_str}[/]"
                else:
                    free_str = f"free {free_str}"
            else:
                free_str = f"free {free_str}"
            parts.append(free_str)

        detail = " | ".join(parts) if parts else ""

        # Status text after ETA: "throttle", "vacuum", "waiting disk", or empty
        status_text = ""
        if update.status_icon == "\u23f8":  # ⏸ — throttle or disk wait
            status_text = "[dim]throttle[/]" if update.phase != "waiting for disk" else "[yellow]waiting disk[/]"
        elif update.status_icon == "\U0001f9f9":  # 🧹
            status_text = "[dim]vacuum[/]"

        self._progress.update(
            self._task_id,
            description=update.phase,
            completed=percent,
            detail=detail,
            status=status_text,
        )
        # Live pulls the renderable via get_renderable on its own timer, so we
        # just update the underlying task/state here; no manual redraw needed.

    def log(self, level: str, message: str) -> None:
        """Route log messages through rich's console so they render cleanly
        above the live progress bar instead of colliding with it on stderr.

        notice-level messages are milestones (chunk-map build, phase
        decisions, stop reasons), not per-round spam, so they are shown -
        they explain long opaque phases that would otherwise leave the bar
        sitting on a static label.  debug-level messages (the periodic
        status line, which the live bar already shows) are hidden unless
        -v/--verbose was passed.
        """
        if level == "debug" and not self._verbose:
            return
        style_map = {"error": "bold red", "warning": "yellow", "notice": "dim",
                     "debug": "dim", "always": "bold"}
        style = style_map.get(level, "")
        text = f"[{style}]{message}[/{style}]" if style else message
        # Print above the Live region so messages don't collide with the map/bar.
        if self._live is not None:
            self._live.console.print(text)
        else:
            self.console.print(text)

    def note(self, message: str) -> None:
        """A one-off message printed above/around the live progress display, if any."""
        if self._live is not None:
            self._live.console.print(f"[dim]{message}[/dim]")
        else:
            self.console.print(f"[dim]{message}[/dim]")

    def stop_progress(self, aborted: bool = False) -> None:
        if self._progress is not None:
            # Per-iteration updates track the cursor against an *estimated*
            # total, so the bar can stop short of 100% (the estimate was
            # conservative, or the relation reached its ideal size early).
            # When the run finishes normally, snap the bar to 100% and clear
            # the transient status rather than leaving it frozen mid-way.
            # When aborted (Ctrl+C or a crash), leave it where it stopped -
            # claiming 100% would misrepresent an unfinished run.
            if not aborted and self._task_id is not None:
                self._progress.update(self._task_id, completed=100.0, status="")
            if self._live is not None:
                self._live.refresh()  # final frame at 100%
                self._live.stop()
                self._live = None
            self._progress = None
            self._task_id = None

    def show_summary(self, result) -> None:
        # Dry run never reaches the "after" measurement - showing a table
        # full of dashes is confusing, just print the message.
        if result.size_after is None:
            if result.message:
                style = "yellow" if result.outcome.value != "completed" else "dim"
                self.console.print(f"[{style}]{result.message}[/{style}]")
            return

        table = Table(title=f'"{result.schema}"."{result.table}" - {result.outcome.value}')
        table.add_column("Metric")
        table.add_column("Before", justify="right")
        table.add_column("After", justify="right")
        table.add_column("Change", justify="right")

        before, after = result.size_before, result.size_after
        self._add_size_row(table, "Heap", before.table_bytes, after.table_bytes)
        if before.toast_bytes or after.toast_bytes:
            self._add_size_row(table, "TOAST", before.toast_bytes, after.toast_bytes)
        self._add_size_row(table, "Indexes", before.indexes_bytes, after.indexes_bytes)
        self._add_size_row(table, "Total", before.total_bytes, after.total_bytes)

        self.console.print(table)
        if result.message:
            style = "yellow" if result.outcome.value != "completed" else "dim"
            self.console.print(f"[{style}]{result.message}[/{style}]")

    @staticmethod
    def _add_size_row(table: Table, label: str, before: int, after: int) -> None:
        table.add_row(label, fmt_bytes(before), fmt_bytes(after), fmt_size_change(before, after))
