import hashlib
import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location('higgs_download', Path(__file__).parents[1] / 'scripts/prepare_higgs_source.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_resume_after_truncated_transfer(tmp_path):
    part, target = tmp_path / 'archive.part', tmp_path / 'archive.gz'
    payload = b'abcdef'
    calls = []

    def transfer(argv):
        calls.append(argv)
        if len(calls) == 1:
            part.write_bytes(payload[:3])
            return subprocess.CompletedProcess(argv, 18)
        assert part.read_bytes() == payload[:3]
        part.write_bytes(payload)
        return subprocess.CompletedProcess(argv, 0)

    with patch.object(module.subprocess, 'run', side_effect=transfer):
        module.download_archive(part, target, hashlib.sha256(payload).hexdigest())
    assert len(calls) == 2 and target.read_bytes() == payload
    assert all(argv[argv.index('--continue-at') + 1] == '-' for argv in calls)


def test_bad_hash_never_published(tmp_path):
    part, target = tmp_path / 'archive.part', tmp_path / 'archive.gz'
    part.write_bytes(b'bad')
    with patch.object(module.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)):
        with pytest.raises(ValueError, match='SHA256'):
            module.download_archive(part, target, hashlib.sha256(b'good').hexdigest())
    assert part.exists() and not target.exists()


def test_retries_bounded(tmp_path):
    with patch.object(module.subprocess, 'run', return_value=subprocess.CompletedProcess([], 18)) as run:
        with pytest.raises(RuntimeError, match='three bounded'):
            module.download_archive(tmp_path / 'partial', tmp_path / 'archive')
    assert run.call_count == 3


def test_nonresumable_server_preserves_partial_then_restarts(tmp_path):
    part, target = tmp_path / 'archive.part', tmp_path / 'archive.gz'
    part.write_bytes(b'abc')
    calls = []

    def transfer(argv):
        calls.append(argv)
        if len(calls) == 1:
            return subprocess.CompletedProcess(argv, 33)
        assert not part.exists()
        assert part.with_name(part.name + '.before_restart').read_bytes() == b'abc'
        part.write_bytes(b'abcdef')
        return subprocess.CompletedProcess(argv, 0)

    with (patch.object(module.subprocess, 'run', side_effect=transfer),
          patch.object(module.shutil, 'disk_usage', return_value=SimpleNamespace(free=5_000_000_000))):
        module.download_archive(part, target, hashlib.sha256(b'abcdef').hexdigest())
    assert target.read_bytes() == b'abcdef'


def test_nonresumable_restart_respects_low_disk_and_retains_original_partial(tmp_path):
    part, target = tmp_path / 'archive.part', tmp_path / 'archive.gz'
    part.write_bytes(b'abc')
    with (patch.object(module.subprocess, 'run', return_value=subprocess.CompletedProcess([], 33)) as run,
          patch.object(module.shutil, 'disk_usage', return_value=SimpleNamespace(free=100_000_000))):
        with pytest.raises(RuntimeError, match='4GB free'):
            module.download_archive(part, target)
    assert run.call_count == 1 and part.read_bytes() == b'abc' and not target.exists()
