"""Incremental content-addressed backups with independent restore verification."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from greenshaping.utils.evidence import atomic_json, sync_directory


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def allowed(path: Path) -> bool:
    return not (path.is_symlink() or any(part.startswith('.') for part in path.parts)
                or path.suffix.lower() in {'.pem', '.key', '.token', '.secret', '.pyc'}
                or '__pycache__' in path.parts
                or (path.name.startswith('id_') and not path.name.endswith('.pub')))


def snapshot(sources: dict[str, Path], destination: Path, name: str) -> dict:
    if not name or Path(name).name != name or name.startswith('.'):
        raise ValueError('invalid snapshot name')
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    objects = destination / 'objects'
    objects.mkdir(exist_ok=True)
    manifest = {'created_epoch': time.time(), 'files': {}, 'excluded': [], 'unstable': []}
    for label, source in sources.items():
        if not label or Path(label).name != label or label.startswith('.'):
            raise ValueError('invalid source label')
        source = Path(source).resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        if source == destination or source in destination.parents:
            raise ValueError('backup destination must be outside source')
        paths = sorted(source.rglob('*')) if source.is_dir() else [source]
        for path in paths:
            relative = path.relative_to(source) if source.is_dir() else Path(path.name)
            key = str(Path(label) / relative)
            if not allowed(relative) or path.is_symlink():
                manifest['excluded'].append(key)
                continue
            if not path.is_file():
                continue
            if shutil.disk_usage(destination).free < path.stat().st_size + 512 * 1024**2:
                raise OSError('insufficient backup disk space; no evidence deleted')
            before = path.stat()
            checksum = sha256(path)
            obj = objects / checksum
            if not obj.exists():
                fd, tempname = tempfile.mkstemp(prefix='.copy-', dir=objects)
                temp = Path(tempname)
                try:
                    with os.fdopen(fd, 'wb') as output, path.open('rb') as input_stream:
                        shutil.copyfileobj(input_stream, output, 1024 * 1024)
                        output.flush()
                        os.fsync(output.fileno())
                    if sha256(temp) != checksum:
                        manifest['unstable'].append(key)
                        continue
                    try:
                        os.link(temp, obj)
                    except FileExistsError:
                        pass
                    sync_directory(objects)
                finally:
                    temp.unlink(missing_ok=True)
            if sha256(obj) != checksum:
                raise ValueError('existing backup object is corrupt: ' + checksum)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                manifest['unstable'].append(key)
                continue
            manifest['files'][key] = {'sha256': checksum, 'bytes': before.st_size}
    manifest['complete'] = not manifest['unstable']
    atomic_json(destination / (name + '.json'), manifest, exclusive=True)
    return verify(destination, name)


def verify(destination: Path, name: str, restore_to: Path | None = None) -> dict:
    destination = Path(destination)
    manifest = json.loads((destination / (name + '.json')).read_text())
    for name, spec in manifest['files'].items():
        key = Path(name)
        checksum = spec['sha256']
        if (key.is_absolute() or '..' in key.parts or len(checksum) != 64
                or any(c not in '0123456789abcdef' for c in checksum)):
            raise ValueError('unsafe backup manifest')
        obj = destination / 'objects' / checksum
        if obj.is_symlink() or obj.stat().st_size != spec['bytes'] or sha256(obj) != checksum:
            raise ValueError('backup checksum mismatch: ' + name)
        if restore_to is not None:
            target = Path(restore_to) / key
            target.parent.mkdir(parents=True, exist_ok=True)
            # Never overwrite a restored or unrelated result.
            with target.open('xb') as output, obj.open('rb') as input_stream:
                shutil.copyfileobj(input_stream, output)
                output.flush()
                os.fsync(output.fileno())
            sync_directory(target.parent)
    return {'complete': manifest['complete'], 'verified_files': len(manifest['files']),
            'excluded_files': len(manifest['excluded']), 'unstable': manifest['unstable'],
            'restore_tested': restore_to is not None}
