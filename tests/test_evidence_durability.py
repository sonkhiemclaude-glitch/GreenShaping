"""Failure drills for evidence publication, recovery and immutable acquisitions."""
import json
import os
import subprocess
import sys

import pytest

from greenshaping.utils.evidence import SessionLedger, atomic_json


def test_interrupted_session_and_lost_checkpoint_are_not_success_or_retry(tmp_path):
    ledger = SessionLedger(tmp_path, {'seed': 7})
    ledger.start('s0', {'question': 1})
    ledger.commit('s0', {'passed': True, 'bytes': 19})
    ledger.start('s1', {'question': 2})
    atomic_json(tmp_path / 'checkpoint.json', {'untrusted': 'stale cache'})
    restored = SessionLedger(tmp_path, {'seed': 7})
    assert restored.checkpoint() == {'committed': 1, 'passed': 1, 'incomplete': ['s1'],
                                     'checksum_verified': True, 'automatic_replay_allowed': False}
    with pytest.raises(FileExistsError):
        restored.start('s1', {'question': 2})
    with pytest.raises(FileExistsError):
        restored.commit('s0', {'passed': False})
    with pytest.raises(ValueError, match='protocol changed'):
        SessionLedger(tmp_path, {'seed': 8})


def test_sudden_process_exit_keeps_commit_and_records_pending_session(tmp_path):
    code = """from pathlib import Path
from greenshaping.utils.evidence import SessionLedger
import os,sys
l=SessionLedger(Path(sys.argv[1]),{'seed':1})
l.start('done',{});l.commit('done',{'passed':True})
l.start('interrupted',{});os._exit(29)
"""
    p = subprocess.run([sys.executable, '-c', code, str(tmp_path)], env=os.environ.copy())
    assert p.returncode == 29
    assert SessionLedger(tmp_path, {'seed': 1}).audit()['incomplete'] == ['interrupted']


def test_tampering_is_detected(tmp_path):
    ledger = SessionLedger(tmp_path, {})
    ledger.start('s0', {})
    ledger.commit('s0', {'passed': False})
    p = tmp_path / 'committed/s0.json'
    value = json.loads(p.read_text())
    value['record']['passed'] = True
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='checksum mismatch'):
        ledger.audit()


def test_numeric_site_keys_have_stable_hashes_after_json_round_trip(tmp_path):
    from greenshaping.utils.evidence import digest

    ledger = SessionLedger(tmp_path, {})
    record = {'passed': True, 'per_site': {i: {'bytes': i * 10} for i in range(20)}}
    ledger.start('n20', {})
    ledger.commit('n20', record)
    assert ledger.audit()['passed'] == 1
    assert digest(record) == digest(json.loads(json.dumps(record)))
    with pytest.raises(ValueError, match='duplicate JSON key'):
        digest({1: 'a', '1': 'b'})


def test_publication_failure_preserves_previous_checkpoint(tmp_path, monkeypatch):
    p = tmp_path / 'progress.json'
    atomic_json(p, {'sessions': 1})
    monkeypatch.setattr(os, 'replace', lambda *a: (_ for _ in ()).throw(OSError('injected disk failure')))
    with pytest.raises(OSError):
        atomic_json(p, {'sessions': 2})
    assert json.loads(p.read_text()) == {'sessions': 1}
    assert list(tmp_path.glob('.progress*')) == []


def test_low_disk_prevents_new_measurement_without_fabricated_result(tmp_path):
    ledger = SessionLedger(tmp_path, {}, minimum_free_bytes=2**63)
    with pytest.raises(OSError, match='free disk'):
        ledger.start('s0', {})
    assert ledger.audit()['committed'] == 0
    assert ledger.audit()['incomplete'] == []


def test_incremental_backup_restore_and_corrupt_replica_detection(tmp_path):
    from greenshaping.utils.evidence_backup import snapshot, verify

    source = tmp_path / 'source'
    source.mkdir()
    (source / 'metric.json').write_text('{"passed":true}')
    (source / '.env').write_text('excluded')
    (source / 'server.key').write_text('excluded')
    (source / 'alias').symlink_to(source / 'metric.json')
    destination = tmp_path / 'backup'
    result = snapshot({'experiment': source}, destination, 'first')
    assert result['complete'] and result['verified_files'] == 1
    snapshot({'experiment': source}, destination, 'second')
    assert len(list((destination / 'objects').iterdir())) == 1  # deduplicated
    restored = tmp_path / 'restored'
    assert verify(destination, 'first', restored)['restore_tested']
    assert (restored / 'experiment/metric.json').read_bytes() == (source / 'metric.json').read_bytes()
    with pytest.raises(FileExistsError):
        verify(destination, 'first', restored)
    next((destination / 'objects').iterdir()).write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='checksum mismatch'):
        verify(destination, 'second')
