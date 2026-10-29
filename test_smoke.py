"""End-to-end smoke tests on 400 tweets with a tiny random BERT (CPU, no download)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "tests" / "tiny_bert"
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")


def setup_module(_):
    if not (TINY / "config.json").exists():
        subprocess.check_call([sys.executable, str(ROOT / "tests" / "make_tiny_bert.py")])


def run(*args):
    cmd = [sys.executable, *map(str, args)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


@pytest.mark.parametrize("spec", [["cb"], ["bb"], ["cb", "--balance", "smote"], ["hyphix", "--fusion", "gate"],
                                  ["hyphix", "--fusion", "concat", "--balance", "smote"], ["cnn"], ["bilstm"], ["bert"]])
def test_train(spec):
    name = "pytest_" + "_".join(s.strip("-") for s in spec)
    run(ROOT / "scripts" / "train.py", "--model", *spec, "--pretrained", TINY, "--limit", 400, "--epochs", 1,
        "--run-name", name, "--eval-sets", "validation")
    assert (ROOT / "runs" / name / "metrics_validation.json").exists()
    assert (ROOT / "runs" / name / "confusion_validation.png").exists()


def test_shap():
    run(ROOT / "scripts" / "explain_shap.py", "--run", ROOT / "runs" / "pytest_cb_balance_smote", "--pretrained", TINY,
        "--n-samples", 12, "--max-evals", 32)
    assert (ROOT / "runs" / "pytest_cb_balance_smote" / "shap_validation" / "shap_global_ranking.csv").exists()


def test_smote_counts():
    import numpy as np
    from hyphix.smote import apply_smote, check_against_imblearn

    rng = np.random.default_rng(0)
    y = np.array([0] * 30 + [1] * 60 + [2] * 45)
    cls = rng.normal(size=(len(y), 8)).astype(np.float32)
    seq = rng.normal(size=(len(y), 5, 8)).astype(np.float16)
    mask = np.ones((len(y), 5), dtype=np.int64)
    out = apply_smote(seq, mask, cls, y, k=5, seed=1)
    assert np.bincount(np.concatenate([y, out["y"]])).tolist() == [60, 60, 60]
    assert check_against_imblearn(cls, y)["identical_counts"]
    # every synthetic sample lies on the segment between its base and neighbour
    i = 0
    b, n, lam = out["base"][i], out["neighbour"][i], out["lam"][i]
    assert np.allclose(out["cls"][i], cls[b] + lam * (cls[n] - cls[b]), atol=1e-5)


def test_statistics():
    from hyphix.evaluation import compute_metrics, mcnemar_test, wilson_interval

    lo, hi = wilson_interval(967, 999)
    assert round(lo, 3) == 0.955 and round(hi, 3) == 0.977
    m = compute_metrics([0, 1, 2, 3] * 25, [0, 1, 2, 3] * 25, n_bootstrap=50)
    assert m["accuracy"] == 1.0
    assert mcnemar_test([0, 1, 1], [0, 1, 0], [0, 0, 1])["discordant"] == 2
