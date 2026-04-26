import os
import signal
import subprocess
import sys
import threading
import time


def stream_output(proc: subprocess.Popen, name: str) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"[{name}] {line.rstrip()}", flush=True)


def terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def start_process(cmd: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def main() -> int:
    backend_port = os.getenv("BACKEND_PORT", "8000")
    frontend_port = os.getenv("PORT", "7860")
    host = os.getenv("HOST", "0.0.0.0")
    max_backend_restarts = int(os.getenv("MAX_BACKEND_RESTARTS", "6"))
    npm_bin = "npm.cmd" if os.name == "nt" else "npm"

    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "server.main:app",
        "--host",
        host,
        "--port",
        backend_port,
        "--workers",
        "1",
    ]
    frontend_cmd = [
        npm_bin,
        "--prefix",
        "spatial-saas",
        "run",
        "start",
        "--",
        "-H",
        host,
        "-p",
        frontend_port,
    ]

    backend_proc = start_process(backend_cmd)
    frontend_proc = start_process(frontend_cmd)

    backend_thread = threading.Thread(
        target=stream_output, args=(backend_proc, "backend"), daemon=True
    )
    frontend_thread = threading.Thread(
        target=stream_output, args=(frontend_proc, "frontend"), daemon=True
    )
    backend_thread.start()
    frontend_thread.start()

    stop_requested = False
    backend_restart_count = 0

    def handle_signal(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        terminate_process(frontend_proc)
        terminate_process(backend_proc)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        while True:
            if stop_requested:
                break

            backend_code = backend_proc.poll()
            frontend_code = frontend_proc.poll()
            if backend_code is not None:
                if stop_requested:
                    break
                backend_restart_count += 1
                if backend_restart_count > max_backend_restarts:
                    print(
                        f"Backend exited with code {backend_code} too many times; stopping.",
                        flush=True,
                    )
                    terminate_process(frontend_proc)
                    return backend_code

                backoff = min(2 * backend_restart_count, 15)
                print(
                    f"Backend exited with code {backend_code}; restarting in {backoff}s "
                    f"({backend_restart_count}/{max_backend_restarts}).",
                    flush=True,
                )
                time.sleep(backoff)
                backend_proc = start_process(backend_cmd)
                backend_thread = threading.Thread(
                    target=stream_output, args=(backend_proc, "backend"), daemon=True
                )
                backend_thread.start()
                continue
            if frontend_code is not None:
                print(f"Frontend exited with code {frontend_code}", flush=True)
                terminate_process(backend_proc)
                return frontend_code

            time.sleep(0.5)
    finally:
        terminate_process(frontend_proc)
        terminate_process(backend_proc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
