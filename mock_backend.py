"""Mock audio synthesis backend."""
import logging
import threading
import socket
import json
import base64
import click

# Configure logging
LOGGER = logging.getLogger(__name__)

# based on EECS 485 project 4 - map reduce


class MockBackend:
    """Audio synthesis backend that receives recipes and sends MP3 audio."""

    def __init__(self, host, port):
        """Initialize the backend and start listening for frontend connections."""
        LOGGER.info("Starting MockBackend on %s:%s", host, port)
        self.host = host
        self.port = port
        self.shutdown_flag = False

        # Start TCP server in a thread
        self.thread = threading.Thread(target=self._run_server)
        self.thread.daemon = False
        self.thread.start()
        self.thread.join()

    def _run_server(self):
        """Run the TCP server to listen for recipe requests from frontend."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.host, self.port))
            sock.listen(5)
            sock.settimeout(1)
            LOGGER.info("Backend server listening on %s:%s", self.host, self.port)

            while not self.shutdown_flag:
                try:
                    client_socket, client_addr = sock.accept()
                    LOGGER.info("Frontend connected from %s", client_addr)
                    self._handle_client(client_socket)
                except socket.timeout:
                    continue
                except Exception as e:
                    LOGGER.error("Server error: %s", e)

    def _handle_client(self, client_socket):
        """Handle a client connection from the frontend."""
        try:
            with client_socket:
                # Receive the recipe request
                message_chunks = []
                while True:
                    try:
                        data = client_socket.recv(4096)
                        if not data:
                            break
                        message_chunks.append(data)
                    except socket.timeout:
                        break

                if message_chunks:
                    message_bytes = b''.join(message_chunks)
                    message_str = message_bytes.decode("utf-8")

                    try:
                        request = json.loads(message_str)
                        LOGGER.info("Received request: %s", request)

                        if request.get("message_type") == "generate_audio":
                            recipe = request.get("recipe", {})
                            request_id = request.get("request_id")
                            self._generate_and_send_audio(
                                client_socket, recipe, request_id
                            )
                        elif request.get("message_type") == "shutdown":
                            self.shutdown_flag = True
                    except json.JSONDecodeError as e:
                        LOGGER.error("JSON decode error: %s", e)
        except Exception as e:
            LOGGER.error("Client handling error: %s", e)

    def _generate_and_send_audio(self, client_socket, recipe, request_id):
        """Generate mock audio from recipe and send MP3 to client."""
        LOGGER.info("Generating audio for recipe: %s", recipe)

        # Validate recipe
        if not isinstance(recipe, dict):
            self._send_error_response(client_socket, request_id, "Invalid recipe format")
            return


        # Create a mock MP3 file (minimal valid MP3 structure)
        mp3_data = self._create_mock_mp3(recipe)

        # Send response with audio data
        response = {
            "message_type": "audio_response",
            "request_id": request_id,
            "status": "success",
            "audio_size": len(mp3_data),
            "audio_data": base64.b64encode(mp3_data).decode("utf-8"),
        }

        response_json = json.dumps(response)
        try:
            client_socket.sendall(response_json.encode("utf-8"))
            LOGGER.info("Sent audio response (size: %d bytes)", len(mp3_data))
        except Exception as e:
            LOGGER.error("Failed to send audio response: %s", e)

    def _send_error_response(self, client_socket, request_id, error_message):
        """Send an error response to the client."""
        response = {
            "message_type": "audio_response",
            "request_id": request_id,
            "status": "error",
            "error": error_message,
        }
        response_json = json.dumps(response)
        try:
            client_socket.sendall(response_json.encode("utf-8"))
        except Exception as e:
            LOGGER.error("Failed to send error response: %s", e)

    def _create_mock_mp3(self, recipe):
        """Create a minimal mock MP3 file."""
        # ID3v2 tag header
        id3_header = b'ID3'
        id3_version = b'\x03\x00'  # v2.3.0
        id3_flags = b'\x00'
        id3_size = b'\x00\x00\x00\x00'

        # Create simple MP3 frames from recipe
        # Use different frequencies for each instrument
        mp3_frames = b''
        instruments = list(recipe.keys())

        # Generate frame data based on recipe (mock implementation)
        for instrument in instruments:
            # Add a simple MP3 frame header (0xFFF sync word + frame header)
            mp3_frames += b'\xFF\xFB\x10\x00'  # Frame sync + minimal MPEG1 Layer3 header

        # Pad with some silence data
        mp3_frames += b'\x00' * 1024

        return id3_header + id3_version + id3_flags + id3_size + mp3_frames

    def shutdown(self):
        """Shutdown the backend."""
        self.shutdown_flag = True
        LOGGER.info("Backend shutting down")


@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=5000)
@click.option("--logfile", "logfile", default=None)
@click.option("--loglevel", "loglevel", default="info")
def main(host, port, logfile, loglevel):
    """Run the mock audio synthesis backend."""
    if logfile:
        handler = logging.FileHandler(logfile)
    else:
        handler = logging.StreamHandler()
    formatter = logging.Formatter(f"Backend:{port} [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(loglevel.upper())
    MockBackend(host, port)


if __name__ == "__main__":
    main()
