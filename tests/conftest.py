"""Shared pytest fixtures for the pg-compact test suite.

Tests run against real PostgreSQL servers started via
``tests/docker-compose.yml`` (see the README's "Running the tests"
section). We deliberately do NOT spin up containers automatically from
pytest - bloat/compaction behavior is sensitive to autovacuum timing and
background activity, so tests are easier to reason about (and to debug
interactively) against containers the developer explicitly controls.

Tests are parametrized across both supported PostgreSQL versions (12, the
minimum supported, and 17, a recent one) via the ``pg_conn`` fixture,
unless a test only cares about one version.
"""

from __future__ import annotations

import uuid

import psycopg
import pytest

from pg_compact import db

PG_PORTS = {
    "pg12": 55412,
    "pg17": 55417,
}


def _connect(port: int) -> psycopg.Connection:
    params = db.ConnectionParams(
        host="localhost", port=port, user="postgres", password="postgres", dbname="pgcompact_test"
    )
    return db.connect(params)


def pytest_addoption(parser):
    parser.addoption(
        "--pg-version",
        action="store",
        default=None,
        help="Restrict tests to one server: pg12 or pg17 (default: run against both).",
    )


def pytest_generate_tests(metafunc):
    if "pg_version" in metafunc.fixturenames:
        restrict = metafunc.config.getoption("--pg-version")
        versions = [restrict] if restrict else list(PG_PORTS.keys())
        metafunc.parametrize("pg_version", versions, indirect=True)


@pytest.fixture
def pg_version(request):
    return request.param


@pytest.fixture
def pg_conn(pg_version):
    """A live connection to the requested PostgreSQL container.

    Verifies connectivity upfront with a clear skip (not an error) if the
    container isn't running, since these are integration tests that
    require `docker compose -f tests/docker-compose.yml up -d` to have
    been run first.
    """
    port = PG_PORTS[pg_version]
    try:
        conn = _connect(port)
    except psycopg.OperationalError as exc:
        pytest.skip(
            f"Could not connect to the {pg_version} test container on port {port}: {exc}. "
            "Run 'docker compose -f tests/docker-compose.yml up -d' first."
        )
    # Ensure required extensions are installed (idempotent).
    with conn.cursor() as cur:
        for ext in ("pageinspect", "pg_freespacemap"):
            cur.execute(f"CREATE EXTENSION IF NOT EXISTS {ext}")
    yield conn
    conn.close()


@pytest.fixture
def table_name():
    """A unique table name per test, so parallel/parametrized runs never collide."""
    return f"t_{uuid.uuid4().hex[:12]}"


@pytest.fixture
def pg_table(pg_conn, table_name):
    """Creates an empty table and guarantees cleanup, whatever the test does to it."""
    yield table_name
    with pg_conn.cursor() as cur:
        cur.execute(f'DROP TABLE IF EXISTS "{table_name}"')


def make_bloated_table(
    conn: psycopg.Connection,
    table: str,
    total_rows: int = 5_000,
    keep_rows: int = 500,
    row_padding: int = 200,
    vacuum_after_delete: bool = True,
) -> None:
    """Build a table with reclaimable bloat: insert then delete most of it.

    Deleting the leading rows and keeping the trailing ``keep_rows`` leaves
    the live tuples at the tail of the file with free space ahead of them.
    A plain VACUUM after that (the default here) marks the freed space
    reusable in the FSM but does not shrink the file, since VACUUM only
    truncates *trailing* empty pages - exactly the situation pg-compact is
    meant to address by relocating the tail rows into the free front.
    """
    with conn.cursor() as cur:
        cur.execute(f'DROP TABLE IF EXISTS "{table}"')
        # Disable autovacuum on the table so no background worker vacuums
        # or truncates it mid-test: the bloat/FSM state a test observes must
        # be produced solely by this fixture's own explicit VACUUM, not by
        # autovacuum timing (which otherwise makes FSM-derived assertions
        # like free_percent flaky when the suite runs as a whole).
        cur.execute(
            f'CREATE TABLE "{table}" (id serial primary key, val text) '
            f'WITH (autovacuum_enabled = false)'
        )
        cur.execute(
            f'INSERT INTO "{table}" (val) SELECT repeat(\'x\', %s) FROM generate_series(1, %s)',
            (row_padding, total_rows),
        )
        delete_up_to = total_rows - keep_rows
        cur.execute(f'DELETE FROM "{table}" WHERE id <= %s', (delete_up_to,))
    if vacuum_after_delete:
        db.vacuum(conn, "public", table)
    db.analyze_table(conn, "public", table)


def make_toast_bloated_table(
    conn: psycopg.Connection,
    table: str,
    total_rows: int = 500,
    keep_rows: int = 50,
) -> None:
    """Build a table whose TOAST storage (not just the heap) has reclaimable bloat.

    Each row gets a large, unique text value (~2.5KB, built from md5 hashes
    so it can't compress away) that is forced out to TOAST storage. Most
    rows are then deleted, leaving the TOAST relation itself bloated -
    this is the specific scenario pg-compact's TOAST-rewrite phase exists
    for (see toast.py's module docstring): a plain heap-only no-op UPDATE
    never touches TOAST at all, since PostgreSQL's UPDATE leaves an
    unchanged TOASTed value's on-disk bytes untouched and just copies the
    TOAST pointer into the new heap tuple.
    """
    with conn.cursor() as cur:
        cur.execute(f'DROP TABLE IF EXISTS "{table}"')
        cur.execute(
            f'CREATE TABLE "{table}" (id serial primary key, tag text, val text) '
            f'WITH (autovacuum_enabled = false)'
        )
        cur.execute(
            f"""
            INSERT INTO "{table}" (tag, val)
            SELECT 'row' || gs,
                   (SELECT string_agg(md5(gs::text || '-' || x::text), '')
                    FROM generate_series(1, %s) x)
            FROM generate_series(1, %s) gs
            """,
            (80, total_rows),
        )
        delete_up_to = total_rows - keep_rows
        cur.execute(f'DELETE FROM "{table}" WHERE id <= %s', (delete_up_to,))
    db.vacuum(conn, "public", table)
    db.analyze_table(conn, "public", table)
