"""Process-lifetime experiment lock, automatically released on process exit."""
from contextlib import contextmanager
import os
from pathlib import Path


class AlreadyRunningError(RuntimeError):
    pass


@contextmanager
def run_lock(folder):
    path = Path(folder)/'experiment.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a+b') as f:
        if path.stat().st_size == 0:
            f.write(b'0'); f.flush()
        f.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise AlreadyRunningError(f'Experiment already running: {path.parent}') from e
        try:
            yield
        finally:
            f.seek(0)
            if os.name == 'nt':
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
