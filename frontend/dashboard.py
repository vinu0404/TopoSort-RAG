"""
Usage:
    python dashboard.py

Then open dashboard.html in your browser at http://localhost:8080
"""

import json
import subprocess
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse


class DashboardHandler(SimpleHTTPRequestHandler):
    """HTTP handler that serves static files and provides API endpoints."""
    
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
        """Fetch all tables from PostgreSQL via Docker."""
        try:
            container = self.find_container('postgres')
            if not container:
                self.send_error_response(404, "PostgreSQL container not found")
                return
            
            print(f"Found PostgreSQL container: {container}")
            
            sql_query = """
            SELECT json_build_object(
                'users', (SELECT json_agg(row_to_json(t)) FROM users t),
                'sessions', (SELECT json_agg(row_to_json(t)) FROM sessions t),
                'conversations', (SELECT json_agg(row_to_json(t)) FROM conversations t),
                'messages', (SELECT json_agg(row_to_json(t)) FROM messages t),
                'agent_executions', (SELECT json_agg(row_to_json(t)) FROM agent_executions t),
                'documents', (SELECT json_agg(row_to_json(t)) FROM documents t),
                'conversation_summaries', (SELECT json_agg(row_to_json(t)) FROM conversation_summaries t),
                'user_long_term_memory', (SELECT json_agg(row_to_json(t)) FROM user_long_term_memory t),
                'hitl_requests', (SELECT json_agg(row_to_json(t)) FROM hitl_requests t)
            ) AS data;
            """
            
            cmd = [
                'docker', 'exec', container,
                'psql', '-U', 'postgres', '-d', 'mrag',
                '-c', sql_query,
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
                error_msg = f"PostgreSQL error: {result.stderr}"
                print(f"ERROR: {error_msg}")
                self.send_error_response(500, error_msg)
                return
            

            output = result.stdout.strip()
            output = output.split('\n')[0] if '\n' in output else output
            
            print(f"Received {len(output)} characters of data")
            
            try:
                data = json.loads(output)
                
                for table in ['users', 'sessions', 'conversations', 'messages', 
                             'agent_executions', 'documents', 'conversation_summaries',
                             'user_long_term_memory', 'hitl_requests']:
                    if data.get(table) is None:
                        data[table] = []
                
                print(f"Successfully parsed data with {sum(len(v) for v in data.values())} total rows")
                self.send_json_response(data)
            
            except json.JSONDecodeError as e:
                error_msg = f"JSON parse error: {str(e)}\nFirst 200 chars: {output[:200]}"
                print(f"ERROR: {error_msg}")
                self.send_error_response(500, error_msg)
        
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
        """Fetch Qdrant collections info via Docker."""
        try:
            container = self.find_container('qdrant')
            if not container:
                self.send_error_response(404, "Qdrant container not found")
                return
            
            import urllib.request
            
            try:
                req = urllib.request.Request('http://localhost:6333/collections')
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
            
            self.send_json_response({
                'postgres': {
                    'running': postgres is not None,
                    'container': postgres
                },
                'qdrant': {
                    'running': qdrant is not None,
                    'container': qdrant
                }
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