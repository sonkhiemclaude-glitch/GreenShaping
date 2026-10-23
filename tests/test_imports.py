import importlib

MODULES = [
    "greenshaping", "greenshaping.cli", "greenshaping.data.schemas", "greenshaping.data.synthetic",
    "greenshaping.data.adapters", "greenshaping.shaping.padding", "greenshaping.shaping.rounds",
    "greenshaping.shaping.dp_release", "greenshaping.shaping.pipeline", "greenshaping.energy.cost_model",
    "greenshaping.energy.fit", "greenshaping.energy.rapl", "greenshaping.energy.meter", "greenshaping.energy.carbon",
    "greenshaping.attack.features", "greenshaping.attack.classifiers", "greenshaping.attack.evaluation",
    "greenshaping.controller.grid", "greenshaping.controller.selection", "greenshaping.analysis.bounds",
    "greenshaping.metrics.metrics", "greenshaping.metrics.confidence_intervals", "greenshaping.baselines",
    "greenshaping.experiments.runner", "greenshaping.experiments.plot_results", "greenshaping.testbed.site_server",
    "greenshaping.testbed.orchestrator", "greenshaping.testbed.driver", "greenshaping.testbed.framing",
    "greenshaping.testbed.question_bank", "greenshaping.testbed.capture_commands", "greenshaping.testbed.transport",
    "greenshaping.utils.provenance",
]


def test_all_modules_import():
    for m in MODULES:
        importlib.import_module(m)


def test_synthetic_label_constant():
    import greenshaping
    assert greenshaping.SYNTHETIC_LABEL == "synthetic smoke-test output, not a paper result"
