"""
Admin Dashboard server.

Usage:
    python frontend/dashboard.py

Then open http://localhost:8080/dashboard.html

Environment detection (from DATABASE_URL in .env):
  - host = localhost / 127.0.0.1  →  DEV  mode: tries Docker psql, falls back to asyncpg
  - host = anything else          →  PROD mode: asyncpg direct, no Docker
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse


class DashboardHandler(SimpleHTTPRequestHandler):

    # Tables excluded from discovery (Alembic internals, PostGIS, etc.)
    _EXCLUDED_TABLES = {'alembic_version', 'spatial_ref_sys'}

    # Per-table SQL overrides — used when a table needs custom ordering/filtering.
    # Key: table name.  Value: the inner SELECT (no wrapping json_agg needed here,
    # that is added by _build_snapshot_sql automatically).
    _TABLE_SQL_OVERRIDES = {
        'messages': (
            "SELECT * FROM messages "
            "ORDER BY created_at ASC, "
            "CASE role WHEN 'user' THEN 0 WHEN 'system' THEN 1 ELSE 2 END"
        ),
    }

    # SQL to list all user-created tables in the public schema
    _DISCOVERY_SQL = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_type = 'BASE TABLE'
        ORDER BY table_name;
    """

    def _build_snapshot_sql(self, table_names: list[str]) -> str:
        """Build a single json_build_object query from the discovered table list."""
        parts = []
        for name in table_names:
            if name in self._TABLE_SQL_OVERRIDES:
                inner = (
                    f"SELECT json_agg(row_to_json(t)) "
                    f"FROM ({self._TABLE_SQL_OVERRIDES[name]}) t"
                )
            else:
                inner = f"SELECT json_agg(row_to_json(t)) FROM {name} t"
            parts.append(f"'{name}', ({inner})")
        return f"SELECT json_build_object({', '.join(parts)}) AS data;"

    # ── CORS ─────────────────────────────────────────────────────────

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/db-snapshot':
            self.handle_db_snapshot()
        elif parsed.path == '/api/qdrant-collections':
            self.handle_qdrant_collections()
        elif parsed.path == '/api/container-status':
            self.handle_container_status()
        else:
            super().do_GET()

    # ── Environment detection ─────────────────────────────────────────

    def _parse_database_url(self):
        """Parse DATABASE_URL into components. Returns None if not configured."""
        raw = self.get_config_value('DATABASE_URL')
        if not raw:
            return None
        # Normalise driver prefix so urlparse can handle it
        clean = raw
        for prefix in ('postgresql+asyncpg://', 'postgresql+psycopg2://', 'postgres://'):
            if clean.startswith(prefix):
                clean = 'postgresql://' + clean[len(prefix):]
                break
        p = urlparse(clean)
        return {
            'host':     p.hostname or 'localhost',
            'port':     p.port or 5432,
            'user':     p.username or 'postgres',
            'password': p.password or '',
            'database': (p.path or '/postgres').lstrip('/') or 'postgres',
        }

    def _is_local_db(self, db_info):
        """True when the DB host is on the same machine (dev mode)."""
        return db_info['host'] in ('localhost', '127.0.0.1', '::1')

    # ── DB snapshot ───────────────────────────────────────────────────

    def handle_db_snapshot(self):
        try:
            db_info = self._parse_database_url()
            if not db_info:
                self.send_error_response(500, "DATABASE_URL not set in .env")
                return

            if self._is_local_db(db_info):
                container = self.find_container('postgres')
                if container:
                    print(f"[DEV] Using Docker container: {container}")
                    table_names, data = self._fetch_via_docker(container, db_info)
                else:
                    print("[DEV] No Docker container — falling back to asyncpg")
                    table_names, data = self._fetch_via_asyncpg(
                        self.get_config_value('DATABASE_URL')
                    )
            else:
                print(f"[PROD] Remote DB at {db_info['host']} — using asyncpg")
                table_names, data = self._fetch_via_asyncpg(
                    self.get_config_value('DATABASE_URL')
                )

            # Ensure every discovered table has at least an empty list
            for t in table_names:
                if data.get(t) is None:
                    data[t] = []

            # Include table list in response so the frontend needs no hardcoded list
            self.send_json_response({'tables': table_names, **data})

        except subprocess.TimeoutExpired:
            self.send_error_response(504, "Database query timed out")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send_error_response(500, str(e))

    def _psql(self, container, db_info, sql):
        """Run a SQL string inside a Docker postgres container. Returns stdout."""
        result = subprocess.run(
            ['docker', 'exec', container,
             'psql', '-U', db_info['user'], '-d', db_info['database'],
             '-c', sql, '-t', '-A'],
            capture_output=True, text=True,
            timeout=15, encoding='utf-8', errors='replace',
        )
        if result.returncode != 0:
            raise RuntimeError(f"psql error: {result.stderr.strip()}")
        return result.stdout.strip()

    def _fetch_via_docker(self, container, db_info) -> tuple[list[str], dict]:
        """Discover tables then fetch snapshot via docker exec psql."""
        # Step 1 — discover tables
        raw_tables = self._psql(container, db_info, self._DISCOVERY_SQL)
        table_names = [
            t for t in raw_tables.splitlines()
            if t and t not in self._EXCLUDED_TABLES
        ]
        print(f"[DEV] Discovered {len(table_names)} tables: {table_names}")

        # Step 2 — fetch all data in one query
        snapshot_sql = self._build_snapshot_sql(table_names)
        raw_data = self._psql(container, db_info, snapshot_sql)
        first_line = raw_data.split('\n')[0] if '\n' in raw_data else raw_data
        try:
            data = json.loads(first_line)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"JSON parse error: {e}; got: {first_line[:200]}")
        return table_names, data

    def _fetch_via_asyncpg(self, database_url) -> tuple[list[str], dict]:
        """Discover tables then fetch snapshot via asyncpg."""
        import asyncio
        import asyncpg

        dsn = database_url
        for prefix in ('postgresql+asyncpg://', 'postgresql+psycopg2://'):
            if dsn.startswith(prefix):
                dsn = 'postgresql://' + dsn[len(prefix):]
                break

        excluded = self._EXCLUDED_TABLES
        discovery_sql = self._DISCOVERY_SQL

        async def run():
            conn = await asyncpg.connect(dsn)
            try:
                # Step 1 — discover tables
                rows = await conn.fetch(discovery_sql)
                table_names = [
                    r['table_name'] for r in rows
                    if r['table_name'] not in excluded
                ]
                print(f"Discovered {len(table_names)} tables: {table_names}")

                # Step 2 — fetch snapshot
                snapshot_sql = self._build_snapshot_sql(table_names)
                raw = await conn.fetchval(snapshot_sql)
                if raw is None:
                    data = {}
                elif isinstance(raw, str):
                    data = json.loads(raw)
                else:
                    data = dict(raw)
                return table_names, data
            finally:
                await conn.close()

        return asyncio.run(run())

    # ── Qdrant ────────────────────────────────────────────────────────

    def handle_qdrant_collections(self):
        try:
            import urllib.request

            db_info = self._parse_database_url()
            is_local = db_info and self._is_local_db(db_info)

            # Dev default: localhost:6333  |  Prod: must be set in QDRANT_URL
            qdrant_url = (self.get_config_value('QDRANT_URL') or '').rstrip('/')
            if not qdrant_url:
                if is_local:
                    qdrant_url = 'http://localhost:6333'
                else:
                    self.send_error_response(
                        500,
                        "QDRANT_URL not set — required for non-local environments"
                    )
                    return

            req = urllib.request.Request(f'{qdrant_url}/collections')
            api_key = self.get_config_value('QDRANT_API_KEY')
            if api_key:
                req.add_header('api-key', api_key)

            try:
                with urllib.request.urlopen(req, timeout=5) as resp:
                    self.send_json_response(json.loads(resp.read()))
            except Exception as e:
                self.send_json_response({
                    'status': 'error',
                    'message': f'Cannot reach Qdrant at {qdrant_url}: {e}',
                    'collections': [],
                })

        except Exception as e:
            self.send_error_response(500, str(e))

    # ── Container / env status ────────────────────────────────────────

    def handle_container_status(self):
        try:
            db_info = self._parse_database_url()
            is_local = db_info and self._is_local_db(db_info)
            env_mode = 'dev' if is_local else 'prod'

            postgres_container = self.find_container('postgres') if is_local else None
            qdrant_container  = self.find_container('qdrant')  if is_local else None
            qdrant_url = (self.get_config_value('QDRANT_URL') or '').rstrip('/')

            self.send_json_response({
                'env_mode': env_mode,
                'db': {
                    'host':     db_info['host'] if db_info else None,
                    'database': db_info['database'] if db_info else None,
                    'using':    'docker' if (is_local and postgres_container) else 'asyncpg',
                },
                'postgres': {
                    'running':   postgres_container is not None,
                    'container': postgres_container,
                },
                'qdrant': {
                    'running':   qdrant_container is not None,
                    'container': qdrant_container,
                    'url':       qdrant_url or ('http://localhost:6333' if is_local else None),
                },
            })

        except Exception as e:
            self.send_error_response(500, str(e))

    # ── Utilities ─────────────────────────────────────────────────────

    def find_container(self, name_pattern):
        """Return the first running Docker container whose name contains name_pattern."""
        try:
            result = subprocess.run(
                ['docker', 'ps', '--format', '{{.Names}}'],
                capture_output=True, text=True, timeout=5,
                encoding='utf-8', errors='replace',
            )
            if result.returncode != 0:
                return None
            for name in result.stdout.strip().split('\n'):
                if name_pattern.lower() in name.lower():
                    return name
            return None
        except Exception:
            return None

    def get_config_value(self, key):
        """Read a config key from process env, then from the project .env file."""
        value = os.getenv(key)
        if value:
            return value.strip().strip('"').strip("'")

        env_path = Path(__file__).resolve().parent.parent / '.env'
        if not env_path.exists():
            return None

        try:
            for raw_line in env_path.read_text(encoding='utf-8').splitlines():
                line = raw_line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                if k.strip() == key:
                    return v.strip().strip('"').strip("'")
        except Exception:
            return None

        return None

    def send_json_response(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def send_error_response(self, code, message):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'error': message}).encode())

    def log_message(self, fmt, *args):
        # Suppress noisy GET /dashboard.html 200 lines; keep errors
        if args and str(args[1]) not in ('200', '304'):
            super().log_message(fmt, *args)


def main():
    port = int(os.getenv('DASHBOARD_PORT', '8080'))

    print("=" * 60)
    print("MRAG Database Dashboard")
    print("=" * 60)
    print(f"  http://localhost:{port}/dashboard.html")
    print()
    print("  Auto-detects environment from DATABASE_URL in .env:")
    print("    localhost  →  DEV  (Docker psql → asyncpg fallback)")
    print("    remote     →  PROD (asyncpg direct)")
    print()
    print("  Press Ctrl+C to stop")
    print("=" * 60)

    server = HTTPServer(('localhost', port), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == '__main__':
    main()
