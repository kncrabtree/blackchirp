#!/usr/bin/env python3
"""
Simple TCP echo server for testing PythonTestHardware.

Listens on localhost:12345. For each connection:
- Accumulates received characters into a buffer (max 100 chars)
- When a newline is received, sends the buffer contents back (with newline)
  and clears the buffer
- Logs all activity to stdout

Usage:
    python3 echo_server.py [port]
"""

import socket
import sys
import threading


def handle_client(conn, addr):
    print(f"[CONNECTED] {addr}")
    buf = ""
    try:
        while True:
            data = conn.recv(1024)
            if not data:
                break

            text = data.decode("utf-8", errors="replace")
            print(f"[RECV {addr}] {text!r}")

            for ch in text:
                if ch == "\n":
                    response = buf + "\n"
                    print(f"[SEND {addr}] {response!r}")
                    conn.sendall(response.encode("utf-8"))
                    buf = ""
                else:
                    if len(buf) < 100:
                        buf += ch
                    # silently drop chars beyond 100

    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        print(f"[DISCONNECTED] {addr}")
        conn.close()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 12345
    host = "127.0.0.1"

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(5)
    print(f"[LISTENING] {host}:{port}")

    try:
        while True:
            conn, addr = server.accept()
            t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            t.start()
    except KeyboardInterrupt:
        print("\n[SHUTDOWN]")
    finally:
        server.close()


if __name__ == "__main__":
    main()
