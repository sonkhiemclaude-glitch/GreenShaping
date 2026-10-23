"""Crash-consistent, append-only session evidence on a local POSIX filesystem.

Committed files are authoritative; progress JSON is a rebuildable cache. A
started session without a commit is incomplete, never an automatic retry.
fsync cannot protect against loss of the entire device: use a second host copy.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value, *, exclusive: bool = False) -> None:
    """Durable publish; exclusive mode refuses to replace previous evidence."""
    path = Path(path)
    data = json.dumps(value, indent=2, allow_nan=False).encode() + b'\n'
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temp, path)
            temp.unlink()
        else:
            os.replace(temp, path)
        sync_directory(path.parent)
    finally:
        temp.unlink(missing_ok=True)


def digest(value) -> str:
    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError('ambiguous duplicate JSON key')
            result[key] = item
        return result

    # JSON converts integer site IDs to strings. Canonicalize after that
    # conversion so sites 2 and 10 keep the same hash after a disk round trip.
    normalized = json.loads(json.dumps(value, allow_nan=False), object_pairs_hook=unique)
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, allow_nan=False).encode()).hexdigest()


class SessionLedger:
    """One immutable intent and commit per acquisition; names are run-local IDs."""

    def __init__(self, root: Path, protocol: dict, *, minimum_free_bytes: int = 512 * 1024**2):
        self.root = Path(root)
        self.minimum_free_bytes = minimum_free_bytes
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ('started', 'committed'):
            (self.root / name).mkdir(exist_ok=True)
        sync_directory(self.root)
        frozen = {'protocol': protocol, 'sha256': digest(protocol)}
        try:
            atomic_json(self.root / 'protocol.json', frozen, exclusive=True)
        except FileExistsError:
            if json.loads((self.root / 'protocol.json').read_text()) != frozen:
                raise ValueError('protocol changed: use a new run directory') from None
        sync_directory(self.root.parent)

    def path(self, kind: str, session: str) -> Path:
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,160}', session):
            raise ValueError('unsafe session ID')
        return self.root / kind / (session + '.json')

    def start(self, session: str, metadata: dict) -> None:
        if shutil.disk_usage(self.root).free < self.minimum_free_bytes:
            raise OSError('insufficient free disk: acquisition not started')
        atomic_json(self.path('started', session), {'session': session, 'metadata': metadata}, exclusive=True)

    def commit(self, session: str, record: dict) -> None:
        intent = json.loads(self.path('started', session).read_text())
        value = {'session': session, 'intent_sha256': digest(intent), 'record': record, 'sha256': digest(record)}
        atomic_json(self.path('committed', session), value, exclusive=True)

    def audit(self) -> dict:
        frozen = json.loads((self.root / 'protocol.json').read_text())
        if frozen['sha256'] != digest(frozen['protocol']):
            raise ValueError('protocol checksum mismatch')
        started = {p.stem: p for p in (self.root / 'started').glob('*.json')}
        committed, passed = [], 0
        for path in sorted((self.root / 'committed').glob('*.json')):
            item = json.loads(path.read_text())
            intent = json.loads(started[path.stem].read_text())
            if (item['session'] != path.stem or intent['session'] != path.stem
                    or item['intent_sha256'] != digest(intent) or item['sha256'] != digest(item['record'])):
                raise ValueError('session checksum mismatch: ' + path.stem)
            committed.append(path.stem)
            passed += item['record'].get('passed') is True
        return {'committed': len(committed), 'passed': passed,
                'incomplete': sorted(set(started) - set(committed)), 'checksum_verified': True,
                'automatic_replay_allowed': False}

    def checkpoint(self) -> dict:
        result = self.audit()
        atomic_json(self.root / 'checkpoint.json', result)
        return result
