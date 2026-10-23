from types import SimpleNamespace
import pytest
from greenshaping.utils import storage_budget as s


def test_budget_rejects_growth_before_writing_and_keeps_existing_evidence(tmp_path,monkeypatch):
    p=tmp_path/'commit.json';p.write_bytes(b'evidence')
    monkeypatch.setattr(s.shutil,'disk_usage',lambda _:SimpleNamespace(free=1000))
    assert s.check(tmp_path,max_bytes=20,reserve_bytes=100,upcoming_bytes=5)['output_bytes']==8
    with pytest.raises(RuntimeError,match='output budget'):s.check(tmp_path,max_bytes=10,reserve_bytes=100,upcoming_bytes=5)
    with pytest.raises(RuntimeError,match='reserve'):s.check(tmp_path,max_bytes=20,reserve_bytes=999,upcoming_bytes=5)
    assert p.read_bytes()==b'evidence'
