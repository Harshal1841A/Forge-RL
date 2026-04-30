import os, signal, subprocess, sys, threading, time
from pathlib import Path


def stream_output(proc, name):
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"[{name}] {line.rstrip()}", flush=True)


def terminate_process(proc):
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def start_process(cmd):
    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )


def find_frontend_dir():
    for name in ["spatial-saas", "spatial_saas"]:
        if Path(name).is_dir():
            return name
    return "spatial-saas"


def main():
    # FastAPI on 7860 — the OpenEnv / HuggingFace exposed port
    backend_port  = os.getenv("BACKEND_PORT",  "7860")
    # Next.js on 3000 — internal only
    frontend_port = os.getenv("FRONTEND_PORT", "3000")
    host          = os.getenv("HOST", "0.0.0.0")
    npm_bin       = "npm.cmd" if os.name == "nt" else "npm"
    frontend_dir  = find_frontend_dir()
    max_restarts  = int(os.getenv("MAX_BACKEND_RESTARTS", "6"))

    backend_cmd = [
        sys.executable, "-m", "uvicorn",
        "server.main:app",
        "--host", host,
        "--port", backend_port,
        "--workers", "1",
    ]
    frontend_cmd = [
        npm_bin, "--prefix", frontend_dir,
        "run", "start", "--",
        "-H", host, "-p", frontend_port,
    ]

    backend_proc  = start_process(backend_cmd)
    frontend_proc = start_process(frontend_cmd)

    threading.Thread(target=stream_output, args=(backend_proc, "backend"), daemon=True).start()
    threading.Thread(target=stream_output, args=(frontend_proc, "frontend"), daemon=True).start()

    stop_requested = False
    restart_count  = 0

    def handle_signal(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        terminate_process(backend_proc)
        terminate_process(frontend_proc)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT,  handle_signal)

    while not stop_requested:
        time.sleep(5)
        if frontend_proc.poll() is not None and not stop_requested:
            print("[supervisor] Frontend exited unexpectedly", flush=True)
            stop_requested = True

        if backend_proc.poll() is not None and not stop_requested:
            if restart_count < max_restarts:
                restart_count += 1
                print(f"[supervisor] Backend crashed — restart {restart_count}/{max_restarts}", flush=True)
                time.sleep(2)
                backend_proc = start_process(backend_cmd)
                threading.Thread(target=stream_output, args=(backend_proc, "backend"), daemon=True).start()
            else:
                print("[supervisor] Max backend restarts reached. Stopping.", flush=True)
                stop_requested = True

    terminate_process(backend_proc)
    terminate_process(frontend_proc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
