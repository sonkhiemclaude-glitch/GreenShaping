import hashlib
import importlib.util
import io
import tarfile
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('restore',Path(__file__).parents[1]/'scripts/restore_compacted_evidence.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_archive_restore_verifies_hash_and_refuses_overwrite(tmp_path):
    p=tmp_path/'archive.tar.gz';data=b'complete raw trace'
    with tarfile.open(p,'w:gz') as t:
        info=tarfile.TarInfo('results/raw.pcap');info.size=len(data);t.addfile(info,io.BytesIO(data))
    row={'archive':str(p),'archive_sha256':m.sha(p),'member':'results/raw.pcap','size':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    out=tmp_path/'raw.pcap';m.restore(row,out);assert out.read_bytes()==data
    out.write_bytes(b'other evidence')
    with pytest.raises(FileExistsError):m.restore(row,out)
    assert out.read_bytes()==b'other evidence'
