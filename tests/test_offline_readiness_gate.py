import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def test_notification_gate_requires_verified_checks_same_source_backup_and_disk(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('offline_gate', Path(__file__).parents[1] / 'scripts/watch_offline_cloud_gate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    app, results, exports = [tmp_path / n for n in ['app', 'results', 'exports']]
    for path in [app / 'src', app / 'scripts', results, exports]:
        path.mkdir(parents=True)
    (results / 'readiness.json').write_text(json.dumps({'software_resilience_verified': True}))
    (results / 'source_sha256.json').write_text('{}')
    (exports / 'ec2_84_metrics_verified.json').write_text(json.dumps({'cross_host_verified': True}))
    (exports / 'mac_complete_controller_copy_verified.json').write_text(json.dumps({'cross_host_copy_verified': True}))
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda p: SimpleNamespace(free=100))
    gates, failed = module.inspect(results, app, exports, 101)
    assert not failed and not gates['storage_at_least_10GiB']
    gates, failed = module.inspect(results, app, exports, 100)
    assert all(gates.values()) and not failed
    (app / 'src/changed.py').write_text('changed source')
    gates, failed = module.inspect(results, app, exports, 100)
    assert failed and not gates['frozen_source_matches_prepared_batch']
