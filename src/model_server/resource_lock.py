"""Serialize model-router requests across Neo processes on Windows and Linux."""
from contextlib import contextmanager
import os
from pathlib import Path
import time


@contextmanager
def router_request_lock(path, timeout=600):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Never unlink a lock file: another process may still hold its inode.
    with open(path, 'a+b') as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                if os.name == 'nt':
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (11, 13, 35, 36):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError('Timed out waiting for the active model request to finish') from exc
                time.sleep(0.05)
        try:
            yield
        finally:
            if os.name == 'nt':
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
