import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse



class CalibrationUploadServer:
    """Simple web server for uploading calibration YAML files."""

    def __init__(self, port: int = 8080):
        self._port = port

    def _make_handler(self):
        class Handler(BaseHTTPRequestHandler):
            UPLOAD_PAGE = b"""<!DOCTYPE html>
<html>
<head>
<title>Calibration Upload</title>
<style>
  body { font-family: sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; }
  h1 { color: #333; }
  form { margin: 20px 0; padding: 20px; border: 1px solid #ccc; border-radius: 8px; }
  label { display: block; margin: 10px 0 5px; font-weight: bold; }
  input[type=text] { width: 100%; padding: 8px; box-sizing: border-box; }
  input[type=file] { margin: 10px 0; }
  button { background: #2563eb; color: white; border: none; padding: 10px 20px;
           border-radius: 4px; cursor: pointer; font-size: 16px; margin-top: 10px; }
  button:hover { background: #1d4ed8; }
  .success { color: #16a34a; font-weight: bold; }
  .error { color: #dc2626; font-weight: bold; }
  .files { margin-top: 20px; }
  .files li { margin: 4px 0; }
</style>
</head>
<body>
<h1>Calibration File Upload</h1>
<form method="POST" action="/upload" enctype="multipart/form-data">
  <label for="name">Calibration name:</label>
  <input type="text" id="name" name="name" placeholder="e.g. objdetect, 0, 1" required>
  <p style="color:#666; margin-top:4px;">File will be saved as <code>calibration_[name].yml</code></p>
  <label for="file">Calibration file (.yml):</label>
  <input type="file" id="file" name="file" accept=".yml,.yaml" required>
  <br>
  <button type="submit">Upload</button>
</form>
<div class="files">
  <h3>Existing calibration files:</h3>
  <ul id="filelist">FILELIST</ul>
</div>
</body>
</html>"""

            def do_GET(self):
                file_list = ""
                base_dir = "."
                for f in sorted(os.listdir(base_dir)):
                    if f.startswith("calibration_") and f.endswith(".yml"):
                        file_list += f"<li>{f}</li>"
                if not file_list:
                    file_list = "<li><em>None found</em></li>"

                page = self.UPLOAD_PAGE.replace(b"FILELIST", file_list.encode())
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)

            def do_POST(self):
                if self.path != "/upload":
                    self.send_error(404)
                    return

                content_type = self.headers.get("Content-Type", "")
                if "multipart/form-data" not in content_type:
                    self._respond(400, "Expected multipart/form-data")
                    return

                boundary = content_type.split("boundary=")[-1].encode()
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)

                name = self._extract_field(body, boundary, "name")
                file_data = self._extract_file(body, boundary, "file")

                if not name:
                    self._respond(400, "Missing calibration name")
                    return
                if file_data is None:
                    self._respond(400, "Missing file")
                    return

                # Sanitize name
                safe_name = "".join(c for c in name if c.isalnum() or c in "_-")
                if not safe_name:
                    self._respond(400, "Invalid name")
                    return

                filename = f"calibration_{safe_name}.yml"
                filepath = filename
                with open(filepath, "wb") as f:
                    f.write(file_data)

                print(f"Calibration uploaded: {filename} ({len(file_data)} bytes), restarting...")
                self._respond(200, f"Saved as {filename} ({len(file_data)} bytes). Restarting...")
                os._exit(0)

            def _respond(self, code, message):
                css_class = "success" if code == 200 else "error"
                body = f'<html><body style="font-family:sans-serif;max-width:600px;margin:40px auto;padding:0 20px;"><p class="{css_class}">{message}</p></body></html>'
                self.send_response(code)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body.encode())

            def _extract_field(self, body, boundary, field_name):
                parts = body.split(b"--" + boundary)
                for part in parts:
                    header_end = part.find(b"\r\n\r\n")
                    if header_end < 0:
                        continue
                    header = part[:header_end].decode(errors="replace")
                    if f'name="{field_name}"' in header and "filename" not in header:
                        value = part[header_end+4:]
                        if value.endswith(b"\r\n"):
                            value = value[:-2]
                        return value.decode().strip()
                return None

            def _extract_file(self, body, boundary, field_name):
                parts = body.split(b"--" + boundary)
                for part in parts:
                    header_end = part.find(b"\r\n\r\n")
                    if header_end < 0:
                        continue
                    header = part[:header_end].decode(errors="replace")
                    if f'name="{field_name}"' in header and "filename" in header:
                        data = part[header_end+4:]
                        if data.endswith(b"\r\n"):
                            data = data[:-2]
                        return data
                return None

            def log_message(self, format, *args):
                pass

        return Handler

    def start(self):
        def run():
            server = ThreadingHTTPServer(("", self._port), self._make_handler())
            print(f"Calibration upload server at http://0.0.0.0:{self._port}")
            server.serve_forever()
        threading.Thread(target=run, daemon=True).start()
