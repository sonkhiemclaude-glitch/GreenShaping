# GreenShaping

Research code and de-identified evaluation data for **Green Shaping: Energy-Aware Design and Validation of Metadata Protection for LLM-Orchestrated Federated Analytics**. This is a research draft, not a claim of measured energy or certified packet-level privacy.

## What is available

- `src/greenshaping/`: padding, rounds, DP release, public-contract admission, workflow envelopes, local TLS adapters, analysis and workload generation.
- `scripts/` and `tests/`: reproduction/analysis scripts and the current regression suite. Operational launch recipes require a separately authorised deployment; shipped cloud configs are an offline profile with every host reserved.
- [Research data](data/README.md): HIGGS provenance/TRAIN banks, Mac calibration, 168 EC2 sessions, 48 model checks, synthetic sensitivity/admission tables, and three local TLS phases with all failures retained.
- [Manuscript](manuscript/): LaTeX source, figure assets and the latest review PDF. Funding and acknowledgements follow the authors' instructions. CRediT allocation remains proposed until every author verifies their actual roles; competing-interest and AI-use confirmations are pending.
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

Build the review paper with `cd manuscript` followed by `tectonic --keep-logs main.tex`. To rerun synthetic studies, use a scratch checkout/output directory: benchmark scripts write to their dated output folders. Do not overwrite the preserved published results. A fresh TLS acquisition requires scoped capture permission, bounded storage and new run identifiers; see `scripts/benchmark_local_contract_tls.py`. No acquisition, model download, cloud provisioning or external message is triggered by importing the package or running its unit tests.

## Authors

1. Lam Nguyen Tran Thanh — ORCID 0000-0003-0991-6157
2. Thanh Pham Chi — ORCID 0000-0002-5674-7985
3. Son Ha Xuan — ORCID 0000-0002-5336-0566
4. Nguyen Ngoc Phien — ORCID 0000-0001-9152-6208, corresponding author (nguyenngocphien@tdtu.edu.vn)
5. Vo Hong Khanh — ORCID 0009-0000-2881-0943
6. Nguyen Minh Triet — ORCID 0009-0009-4235-584X
7. Duy Pham Dang Xuan — ORCID 0009-0009-0116-1619
8. Tung Nguyen Quang — ORCID 0009-0003-5046-5535

## Funding and acknowledgements

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

We acknowledge the support of time and facilities provided by HCMC University of Technology and Engineering (HCM-UTE) for this study. *This project was also undertaken with the assistance of computing resources from RACE (RMIT Advanced Computing Ecosystem).*

## Versioned snapshot

The publication snapshot is tagged `paper-artifact-2026-10-23`. Cite that tag or its commit when reproducing these tables; later `main` changes may differ. No archival DOI has been assigned.
