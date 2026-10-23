"""Wait for this project's data job, then run two finite TRAIN planner/tool checks."""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--evidence', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
state = {'stage': 'waiting_for_data', 'cloud_used': False, 'test_executed': False}
try:
    deadline = time.monotonic() + 14400
    while True:
        (a.out / 'status.json').write_text(json.dumps(state, indent=2))
        path = a.evidence / 'status.json'
        source = json.loads(path.read_text()) if path.exists() else {}
        if source.get('stage') == 'completed' and source.get('reference_preparation_pass'):
            break
        if source.get('stage') in ('failed', 'completed_with_findings'):
            raise RuntimeError('data prerequisite failed; planner not launched')
        if time.monotonic() > deadline:
            raise TimeoutError('data wait deadline reached; planner not launched')
        time.sleep(10)
    with urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=30) as response:
        (a.out / 'ollama_model_inventory.json').write_bytes(response.read())
    state['stage'] = 'train_planner_tool_validation'
    (a.out / 'status.json').write_text(json.dumps(state, indent=2))
    outcomes = {}
    for model, name in [('mistral-small3.1:24b', 'mistral'), ('qwen2.5:14b-instruct-q4_K_M', 'qwen14b')]:
        result = subprocess.run([sys.executable, 'scripts/validate_higgs_planner.py', '--bank',
                                 str(a.evidence / 'banks/n20/train.jsonl'), '--store', str(a.evidence / 'store'),
                                 '--out', str(a.out / name), '--sites', '20', '--model', model], timeout=4000)
        outcomes[name] = result.returncode
        if result.returncode not in (0, 2):
            raise RuntimeError('planner execution failed: ' + name)
    state.update(stage='completed' if all(v == 0 for v in outcomes.values()) else 'completed_with_findings',
                 outcomes=outcomes, large_run_ready=False)
except Exception as exc:
    state.update(stage='failed', error=repr(exc))
    raise
finally:
    (a.out / 'status.json').write_text(json.dumps(state, indent=2))
