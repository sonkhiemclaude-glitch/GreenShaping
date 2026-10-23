"""Resumable SFTP evidence downloads with bounded prefetch and verified publish."""
import fcntl
import hashlib
import os
import uuid
from pathlib import Path

from greenshaping.utils.evidence import sync_directory


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download_verified(sftp, source: str, target: Path, expected_sha256: str):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_name(target.name + '.download.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another evidence download owns this target; no concurrent append') from None
        return _download_locked(sftp, source, target, expected_sha256)


def _download_locked(sftp, source: str, target: Path, expected_sha256: str):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.partial')
    before = sftp.stat(source)
    if target.exists():
        if checksum(target) == expected_sha256:
            return
        if not partial.exists() and target.stat().st_size < before.st_size:
            os.replace(target, partial)  # preserve/resume a legacy interrupted download
        else:
            os.replace(target, target.with_name(target.name + '.unverified.' + uuid.uuid4().hex))
    if partial.exists() and partial.stat().st_size > before.st_size:
        raise ValueError('partial exceeds remote object size; preserve and inspect')
    offset = partial.stat().st_size if partial.exists() else 0
    with sftp.open(source, 'rb') as remote, partial.open('ab') as output:
        remote.seek(offset)
        if hasattr(remote, 'prefetch'):
            remote.prefetch(file_size=before.st_size, max_concurrent_requests=8)
        synced = offset
        try:
            while chunk := remote.read(1024 * 1024):
                output.write(chunk)
                offset += len(chunk)
                if offset - synced >= 8 * 1024**2:
                    output.flush()
                    os.fsync(output.fileno())
                    synced = offset
        finally:
            output.flush()
            os.fsync(output.fileno())
    after = sftp.stat(source)
    if (before.st_size, before.st_mtime) != (after.st_size, after.st_mtime):
        raise ValueError('remote evidence changed during transfer; no final file published')
    if partial.stat().st_size != before.st_size or checksum(partial) != expected_sha256:
        os.replace(partial, partial.with_name(partial.name + '.corrupt.' + uuid.uuid4().hex))
        raise ValueError('download checksum mismatch; failed copy preserved separately')
    os.replace(partial, target)
    sync_directory(target.parent)
