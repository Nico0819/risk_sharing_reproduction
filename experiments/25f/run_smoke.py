"""Run 25F by file path or with ``python -m experiments.25f.run_smoke``."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.baseline import run_smoke as shared  # noqa: E402
from protocol.paper_spec import Treatment  # noqa: E402


def run_experiment(**kwargs):
    # Cover insurance introduction and withdrawal without changing the paper's
    # unresolved assignment probabilities. The caller must supply those.
    kwargs.setdefault("completed_period_limit", 31)
    kwargs.setdefault("request_limit", 400)
    return shared.run_experiment(treatment=Treatment.FORCED_INSURANCE_25, **kwargs)


def main(argv=None):
    return shared.main(argv, treatment=Treatment.FORCED_INSURANCE_25)


if __name__ == "__main__":
    raise SystemExit(main())
