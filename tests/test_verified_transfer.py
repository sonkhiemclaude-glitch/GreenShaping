import hashlib
import io
from types import SimpleNamespace

import pytest

from greenshaping.utils.transfer import download_verified


class Remote:
    def __init__(self, data, fail=False):
        self.data, self.fail = data, fail
        self.resume_offset = None

    def stat(self, source):
        return SimpleNamespace(st_size=len(self.data), st_mtime=1)

    def open(self, source, mode):
        parent = self
        class Stream(io.BytesIO):
            def seek(self, offset, *args):
                parent.resume_offset = offset
                return super().seek(offset, *args)

            def read(self, size=-1):
                if parent.fail and self.tell() > 0:
                    raise OSError('injected disconnect')
                return super().read(min(size, 1024))
        return Stream(self.data)


def test_disconnected_copy_resumes_and_only_publishes_matching_bytes(tmp_path):
    data = b'packet trace' * 1000
    expected = hashlib.sha256(data).hexdigest()
    target = tmp_path / 'evidence.tar.gz'
    with pytest.raises(OSError, match='disconnect'):
        download_verified(Remote(data, fail=True), '/evidence', target, expected)
    assert not target.exists() and target.with_name(target.name + '.partial').stat().st_size == 1024
    remote = Remote(data)
    download_verified(remote, '/evidence', target, expected)
    assert remote.resume_offset == 1024 and target.read_bytes() == data
    assert not target.with_name(target.name + '.partial').exists()


def test_wrong_checksum_never_replaces_a_verified_name(tmp_path):
    target = tmp_path / 'evidence'
    with pytest.raises(ValueError, match='checksum'):
        download_verified(Remote(b'wrong'), '/evidence', target, hashlib.sha256(b'right').hexdigest())
    assert not target.exists()
    assert list(tmp_path.glob('evidence.partial.corrupt.*'))


def test_concurrent_writer_is_rejected_before_remote_read_or_partial_append(tmp_path):
    import fcntl
    target=tmp_path/'evidence'
    partial=tmp_path/'evidence.partial';partial.write_bytes(b'valid prefix')
    with (tmp_path/'evidence.download.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='concurrent append'):
            download_verified(Remote(b'data'),'/evidence',target,'0'*64)
    assert partial.read_bytes()==b'valid prefix' and not target.exists()
