# GreenShaping

Source code and de-identified evaluation data for GreenShaping, a toolkit for metadata protection in federated analytics. Physical energy and packet-level privacy are not certified by these results.

## What is available

- `src/greenshaping/`: padding, rounds, DP release, public-contract admission, workflow envelopes, local TLS adapters, analysis and workload generation.
- `scripts/` and `tests/`: reproduction/analysis scripts and the current regression suite. Operational launch recipes require a separately authorised deployment; shipped cloud configs are an offline profile with every host reserved.
- [Research data](data/README.md): HIGGS provenance/TRAIN banks, Mac calibration, 168 EC2 sessions, 48 model checks, synthetic sensitivity/admission tables, and three local TLS phases with all failures retained.
- [Reuse terms](LICENSING.md): an original-code licence has not yet been selected; the upstream HIGGS licence remains CC BY 4.0.

## Main evidence and limits

The public-data EC2 campaign completed 168/168 reference-correct sessions across seven modes. A separate final local TLS comparison completed 96/96 positive sessions and reported all 30 deliberately invalid sessions while retaining fixed cover. Against six-tick periodic cover, the envelope used about 33.4% fewer observed loopback IPv4 bytes at similar internal tool p95 (about 0.610 s); against ten ticks the reduction was 53.9% but tool p95 was 1.649 times as long. These are scripted workloads on one host, not distributed LLM-agent or joule measurements. The initial transport failure and repair are documented separately, without replacing a failed run by a successful retry. Formal results are conditional on their public bounds and observer assumptions. The reserved original final-test bank has not been scored.

## Run locally

Python >=3.11 and a TLS-1.3-capable SSL runtime are required. Reuse an existing environment, or install this package and development dependencies in your own isolated environment:

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/reproduce_public_results.py
python scripts/reproduce_public_attackers.py
```

The regression recorded before publication has 257 passes and one skip for optional Torch. The public-data checkers verify hashes, rederive packet/session costs and all metadata feature rows, and reproduce the 56 frozen retrospective classifier scores. Results are descriptive; software tests are not evidence of privacy.

To rerun synthetic studies, use a scratch checkout/output directory: benchmark scripts write to their dated output folders. Do not overwrite the preserved published results. A fresh TLS acquisition requires scoped capture permission, bounded storage and new run identifiers; see `scripts/benchmark_local_contract_tls.py`. No acquisition, model download, cloud provisioning or external message is triggered by importing the package or running its unit tests.
