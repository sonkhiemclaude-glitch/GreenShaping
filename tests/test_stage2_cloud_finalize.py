import importlib.util
import json
import sys
from pathlib import Path


def test_incomplete_acquisition_cannot_be_reported_as_complete(tmp_path,monkeypatch):
    p=Path(__file__).parents[1]/'scripts/finalize_stage2_cloud.py'
    spec=importlib.util.spec_from_file_location('finalize_cloud',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    (tmp_path/'queue').mkdir();(tmp_path/'results').mkdir()
    (tmp_path/'queue/state.json').write_text('{"queue_status":"idle_complete"}')
    f=tmp_path/'results/failure.json';f.write_text('{"error":"intentional fixture"}')
    (tmp_path/'results/manifest.json').write_text(json.dumps({'failure.json':m.digest(f)}))
    monkeypatch.setattr(sys,'argv',['finalize','--root',str(tmp_path),'--deadline-epoch','1'])
    assert m.main()==0
    d=json.loads((tmp_path/'review/summary.json').read_text())
    assert not d['acquisition_complete'] and not d['cleanup_verified']
    assert not d['all_capture_lifecycle_coverage_verified']
    assert d['sessions']==0 and not d['main_paper_ready']
