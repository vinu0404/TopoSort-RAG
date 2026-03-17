"""
Usage:
    python dashboard.py

Then open dashboard.html in your browser at http://localhost:8080
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse


class DashboardHandler(SimpleHTTPRequestHandler):
    """HTTP handler that serves static files and provides API endpoints."""

    TABLE_NAMES = [
        'users', 'sessions', 'conversations', 'messages',
        'agent_executions', 'documents', 'conversation_summaries',
        'user_long_term_memory', 'hitl_requests', 'user_connections',
        'personas', 'web_scrape_collections', 'web_scrape_urls',
        'scheduled_jobs', 'scheduled_job_steps', 'scheduled_job_runs',
        'scheduled_job_step_results', 'artifacts',
    ]

    SNAPSHOT_SQL = """
        SELECT json_build_object(
            'users', (SELECT json_agg(row_to_json(t)) FROM users t),
            'sessions', (SELECT json_agg(row_to_json(t)) FROM sessions t),
            'conversations', (SELECT json_agg(row_to_json(t)) FROM conversations t),
            'messages', (SELECT json_agg(row_to_json(t)) FROM (SELECT * FROM messages ORDER BY created_at ASC, CASE role WHEN 'user' THEN 0 WHEN 'system' THEN 1 ELSE 2 END) t),
            'agent_executions', (SELECT json_agg(row_to_json(t)) FROM agent_executions t),
            'documents', (SELECT json_agg(row_to_json(t)) FROM documents t),
            'conversation_summaries', (SELECT json_agg(row_to_json(t)) FROM conversation_summaries t),
            'user_long_term_memory', (SELECT json_agg(row_to_json(t)) FROM user_long_term_memory t),
            'hitl_requests', (SELECT json_agg(row_to_json(t)) FROM hitl_requests t),
            'user_connections', (SELECT json_agg(row_to_json(t)) FROM user_connections t),
            'personas', (SELECT json_agg(row_to_json(t)) FROM personas t),
            'web_scrape_collections', (SELECT json_agg(row_to_json(t)) FROM web_scrape_collections t),
            'web_scrape_urls', (SELECT json_agg(row_to_json(t)) FROM web_scrape_urls t),
            'scheduled_jobs', (SELECT json_agg(row_to_json(t)) FROM scheduled_jobs t),
            'scheduled_job_steps', (SELECT json_agg(row_to_json(t)) FROM scheduled_job_steps t),
            'scheduled_job_runs', (SELECT json_agg(row_to_json(t)) FROM scheduled_job_runs t),
            'scheduled_job_step_results', (SELECT json_agg(row_to_json(t)) FROM scheduled_job_step_results t),
            'artifacts', (SELECT json_agg(row_to_json(t)) FROM artifacts t)
        ) AS data;
    """
    
    def end_headers(self):
        """Add CORS headers to allow browser access."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests for API endpoints and static files."""
        parsed = urlparse(self.path)
        
        if parsed.path == '/api/db-snapshot':
            self.handle_db_snapshot()
        
        # API endpoint: fetch Qdrant collections
        elif parsed.path == '/api/qdrant-collections':
            self.handle_qdrant_collections()
        
        # API endpoint: get container status
        elif parsed.path == '/api/container-status':
            self.handle_container_status()
        
        # Serve static files (HTML, CSS, JS)
        else:
            super().do_GET()
    
    def handle_db_snapshot(self):
        """Fetch all tables from PostgreSQL via Docker or DATABASE_URL fallback."""
        try:
            container = self.find_container('postgres')
            if container:
                print(f"Found PostgreSQL container: {container}")
                output = self._fetch_db_snapshot_via_docker(container)
                data = self._parse_snapshot_json(output)
                self._normalize_snapshot(data)
                print(f"Successfully parsed data with {sum(len(v) for v in data.values())} total rows")
                self.send_json_response(data)
                return

            database_url = self.get_config_value('DATABASE_URL')
            if not database_url:
                self.send_error_response(500, "DATABASE_URL is missing and no PostgreSQL container was found")
                return

            print("No PostgreSQL container found. Using DATABASE_URL fallback.")
            data = self._fetch_db_snapshot_via_database_url(database_url)
            self._normalize_snapshot(data)
            print(f"Successfully fetched data with {sum(len(v) for v in data.values())} total rows")
            self.send_json_response(data)
        
        except subprocess.TimeoutExpired:
            print("ERROR: Database query timeout")
            self.send_error_response(504, "Database query timeout")
        
        except UnicodeDecodeError as e:
            error_msg = f"Unicode encoding error: {str(e)}"
            print(f"ERROR: {error_msg}")
            self.send_error_response(500, error_msg)
        
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            self.send_error_response(500, error_msg)
    
    def handle_qdrant_collections(self):
        """Fetch Qdrant collections via localhost or QDRANT_URL fallback."""
        try:
            import urllib.request
            
            qdrant_url = 'http://localhost:6333'
            if not self.find_container('qdrant'):
                qdrant_url = (self.get_config_value('QDRANT_URL') or '').rstrip('/')
                if not qdrant_url:
                    self.send_error_response(500, "QDRANT_URL is missing and no Qdrant container was found")
                    return

            try:
                req = urllib.request.Request(f'{qdrant_url}/collections')
                api_key = self.get_config_value('QDRANT_API_KEY')
                if api_key:
                    req.add_header('api-key', api_key)
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read())
                    self.send_json_response(data)
            
            except Exception as e:
                self.send_json_response({
                    'status': 'error',
                    'message': f'Could not connect to Qdrant API: {str(e)}',
                    'collections': []
                })
        
        except Exception as e:
            self.send_error_response(500, f"Error: {str(e)}")
    
    def handle_container_status(self):
        """Check if PostgreSQL and Qdrant containers are running."""
        try:
            postgres = self.find_container('postgres')
            qdrant = self.find_container('qdrant')
            database_url = self.get_config_value('DATABASE_URL')
            qdrant_url = self.get_config_value('QDRANT_URL')
            
            self.send_json_response({
                'postgres': {
                    'running': postgres is not None,
                    'container': postgres,
                    'fallback_configured': bool(database_url),
                },
                'qdrant': {
                    'running': qdrant is not None,
                    'container': qdrant,
                    'fallback_configured': bool(qdrant_url),
                },
            })
        
        except Exception as e:
            self.send_error_response(500, f"Error: {str(e)}")
    
    def find_container(self, name_pattern):
        """Find a running Docker container by name pattern."""
        try:
            cmd = ['docker', 'ps', '--format', '{{.Names}}']
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=5,
                encoding='utf-8',
                errors='replace'
            )
            
            if result.returncode != 0:
                return None
            
            containers = result.stdout.strip().split('\n')
            for container in containers:
                if name_pattern.lower() in container.lower():
                    return container
            
            return None
        
        except Exception:
            return None

    def get_config_value(self, key):
        """Read config value from process env or project .env file."""
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

    def _fetch_db_snapshot_via_docker(self, container):
        cmd = [
            'docker', 'exec', container,
            'psql', '-U', 'postgres', '-d', 'mrag',
            '-c', self.SNAPSHOT_SQL,
            '-t', '-A'
        ]

        print(f"Executing: {' '.join(cmd[:4])}...")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            encoding='utf-8',
            errors='replace'
        )

        if result.returncode != 0:
            raise RuntimeError(f"PostgreSQL error: {result.stderr}")

        output = result.stdout.strip()
        return output.split('\n')[0] if '\n' in output else output

    def _fetch_db_snapshot_via_database_url(self, database_url):
        import asyncio
        import asyncpg

        dsn = database_url.replace('postgresql+asyncpg://', 'postgresql://')

        async def run_query():
            conn = await asyncpg.connect(dsn)
            try:
                raw = await conn.fetchval(self.SNAPSHOT_SQL)
                if raw is None:
                    return {}
                if isinstance(raw, str):
                    return json.loads(raw)
                if isinstance(raw, dict):
                    return raw
                return dict(raw)
            finally:
                await conn.close()

        return asyncio.run(run_query())

    def _parse_snapshot_json(self, output):
        print(f"Received {len(output)} characters of data")
        try:
            return json.loads(output)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"JSON parse error: {str(e)}; First 200 chars: {output[:200]}")

    def _normalize_snapshot(self, data):
        for table in self.TABLE_NAMES:
            if data.get(table) is None:
                data[table] = []
    
    def send_json_response(self, data):
        """Send a JSON response."""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def send_error_response(self, code, message):
        """Send an error response."""
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'error': message}).encode())


def main():
    """Start the dashboard server."""
    port = 8080
    
    print("=" * 60)
    print("MRAG Database Dashboard Server")
    print("=" * 60)
    print(f"Starting server on http://localhost:{port}")
    print(f"\nOpen dashboard.html in your browser, or visit:")
    print(f"  http://localhost:{port}/dashboard.html")
    print("\nAPI Endpoints:")
    print(f"  GET /api/db-snapshot          - Fetch all PostgreSQL tables")
    print(f"  GET /api/qdrant-collections   - Fetch Qdrant collections")
    print(f"  GET /api/container-status     - Check container status")
    print("\nPress Ctrl+C to stop")
    print("=" * 60)
    
    server = HTTPServer(('localhost', port), DashboardHandler)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        server.shutdown()


if __name__ == '__main__':
    main()