"""PhysFormer loss: rPPG-Toolbox's PhysFormerLossComputer.py is not distributed here (GPL-3.0 ancestry, see
THIRD_PARTY_NOTICES.md); tools/fetch_physformer_loss.py fetches it and PhysFormerLossAdapter.py applies the
KR-rPPG change on top of it.

Without the fetched file: the adapter and the trainer package import, PhysFormer stops with fetch instructions,
the target heart rate is limited to 40-179 bpm before the upstream call (checked with a stand-in), the adapter
runs only sha256-verified bytes, and the fetch script writes nothing that fails the sha256 check. With the file
(default location, or KR_RPPG_PHYSFORMER_LOSS): the adapter reproduces, on seeded random inputs, the loss of the
KR-rPPG manuscript runs (tests/data/physformer_loss_reference.json: numbers generated with the former in-repo
implementation; no code of it is stored). Without CUDA the upstream code, which hard-codes .cuda(), runs on the
CPU through a test-only stand-in.
"""

import hashlib
import json
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
TOOLBOX = REPO / "third_party" / "rPPG-Toolbox"
REFERENCE = json.loads((Path(__file__).with_name("data") / "physformer_loss_reference.json").read_text())
sys.path.insert(0, str(TOOLBOX))
sys.path.insert(0, str(REPO / "tools"))

import fetch_physformer_loss as fetch  # noqa: E402
from neural_methods.loss import PhysFormerLossAdapter as adapter  # noqa: E402

STAND_IN = b"class TorchLossComputer:\n    SOURCE = 'stand-in for the pinned upstream file'\n"


def _upstream_present() -> bool:
    try:
        return adapter.upstream_file().is_file()
    except adapter.UpstreamLossMissing:
        return True  # misconfigured KR_RPPG_PHYSFORMER_LOSS: let the reproduction tests fail with the message


UPSTREAM_PRESENT = _upstream_present()


@pytest.fixture
def no_location_env(monkeypatch):
    """Simulate a machine on which KR_RPPG_PHYSFORMER_LOSS is not set."""
    monkeypatch.delenv(adapter.LOCATION_ENV, raising=False)


@pytest.fixture
def isolated_upstream_module():
    """Run with no upstream module loaded and restore the previous state afterwards."""
    saved = sys.modules.pop(adapter.UPSTREAM_MODULE, None)
    adapter.upstream_loss_computer.cache_clear()
    yield
    sys.modules.pop(adapter.UPSTREAM_MODULE, None)
    if saved is not None:
        sys.modules[adapter.UPSTREAM_MODULE] = saved
    adapter.upstream_loss_computer.cache_clear()


def test_pins_are_consistent():
    assert re.fullmatch(r"[0-9a-f]{40}", adapter.UPSTREAM_COMMIT)
    assert re.fullmatch(r"[0-9a-f]{64}", adapter.UPSTREAM_SHA256)
    assert adapter.UPSTREAM_URL.endswith(f"/{adapter.UPSTREAM_COMMIT}/{adapter.UPSTREAM_RELPATH}")
    assert adapter.UPSTREAM_URL.startswith("https://raw.githubusercontent.com/ubicomplab/rPPG-Toolbox/")
    assert adapter.UPSTREAM_FILE == (TOOLBOX / adapter.UPSTREAM_RELPATH).resolve()
    assert (adapter.HR_MIN_BPM, adapter.HR_MAX_BPM) == (40.0, 179.0)
    assert REFERENCE["upstream_sha256"] == adapter.UPSTREAM_SHA256
    assert adapter.UPSTREAM_COMMIT in (TOOLBOX / "NOTICE.md").read_text()
    assert adapter.LOCATION_ENV in (REPO / "README.md").read_text()


def test_fetched_file_is_git_ignored():
    ignored = (REPO / ".gitignore").read_text().splitlines()
    assert {"PhysFormerLossComputer.py", "PhysFormerLossComputer.py.part", "RythmFormerLossComputer.py"} <= set(ignored)


def test_no_file_of_gpl_ancestry_is_tracked_by_git():
    if not (REPO / ".git").exists() or shutil.which("git") is None:
        pytest.skip("not a git checkout")
    git = ["git", "-c", f"safe.directory={REPO}", "--no-optional-locks", "-C", str(REPO)]
    tracked = subprocess.run(git + ["ls-files", "--", "*PhysFormerLossComputer.py*", "*RythmFormerLossComputer.py*"],
                             capture_output=True, text=True, check=True).stdout
    assert tracked == ""
    default = f"third_party/rPPG-Toolbox/{adapter.UPSTREAM_RELPATH}"
    assert subprocess.run(git + ["check-ignore", "-q", "--no-index", default]).returncode == 0


def test_missing_upstream_file_stops_with_fetch_instructions(tmp_path):
    with pytest.raises(adapter.UpstreamLossMissing, match="tools/fetch_physformer_loss.py"):
        adapter.check_upstream_loss(tmp_path / "PhysFormerLossComputer.py")


def test_other_upstream_file_is_rejected(tmp_path):
    path = tmp_path / "PhysFormerLossComputer.py"
    path.write_text("# not the pinned upstream file\n")
    with pytest.raises(adapter.UpstreamLossMissing, match="sha256"):
        adapter.check_upstream_loss(path)


def test_location_from_the_environment(monkeypatch, tmp_path):
    path = tmp_path / "elsewhere" / "PhysFormerLossComputer.py"
    monkeypatch.setenv(adapter.LOCATION_ENV, str(path))
    assert adapter.upstream_file() == path
    with pytest.raises(adapter.UpstreamLossMissing) as missing:
        adapter.check_upstream_loss()
    message = str(missing.value)
    assert "not distribute" in message and f"--dest {path}" in message and f"Looked for: {path}" in message
    monkeypatch.setenv(adapter.LOCATION_ENV, "relative/PhysFormerLossComputer.py")
    with pytest.raises(adapter.UpstreamLossMissing, match="absolute path"):
        adapter.upstream_file()
    monkeypatch.setenv(adapter.LOCATION_ENV, "  ")
    assert adapter.upstream_file() == adapter.UPSTREAM_FILE


def test_physformer_trainer_stops_before_training_without_the_file(monkeypatch, tmp_path, no_location_env):
    trainers = pytest.importorskip("neural_methods.trainer")  # other trainers import without the file
    assert hasattr(trainers, "PhysnetTrainer") and hasattr(trainers, "PhysFormerTrainer")
    monkeypatch.setattr(adapter, "UPSTREAM_FILE", tmp_path / "PhysFormerLossComputer.py")
    with pytest.raises(adapter.UpstreamLossMissing, match="not distribute"):
        trainers.PhysFormerTrainer.PhysFormerTrainer(config=None, data_loader=None)


def test_the_module_is_built_from_the_verified_bytes(monkeypatch, tmp_path, isolated_upstream_module):
    path = tmp_path / "PhysFormerLossComputer.py"
    path.write_bytes(STAND_IN)
    monkeypatch.setattr(adapter, "UPSTREAM_SHA256", hashlib.sha256(STAND_IN).hexdigest())
    monkeypatch.setenv(adapter.LOCATION_ENV, str(path))
    cls = adapter.upstream_loss_computer()
    assert cls.SOURCE == "stand-in for the pinned upstream file"
    module = sys.modules[adapter.UPSTREAM_MODULE]
    assert cls.__module__ == adapter.UPSTREAM_MODULE and module.__file__ == str(path)
    assert not (tmp_path / "__pycache__").exists()  # no bytecode cache next to the file
    path.write_bytes(b"class TorchLossComputer:\n    SOURCE = 'edited'\n")
    adapter.upstream_loss_computer.cache_clear()
    with pytest.raises(adapter.UpstreamLossMissing, match="sha256"):  # an edited file is refused at once
        adapter.upstream_loss_computer()


def test_a_module_imported_without_the_check_is_refused(monkeypatch, tmp_path, isolated_upstream_module):
    path = tmp_path / "PhysFormerLossComputer.py"
    path.write_bytes(STAND_IN)
    monkeypatch.setattr(adapter, "UPSTREAM_SHA256", hashlib.sha256(STAND_IN).hexdigest())
    monkeypatch.setenv(adapter.LOCATION_ENV, str(path))
    unchecked = types.ModuleType(adapter.UPSTREAM_MODULE)
    unchecked.__file__ = str(path)
    sys.modules[adapter.UPSTREAM_MODULE] = unchecked
    with pytest.raises(adapter.UpstreamLossMissing, match="without the sha256 check"):
        adapter.upstream_loss_computer()


@pytest.mark.parametrize("target, bounded", [(25.0, 40.0), (39.99, 40.0), (40.0, 40.0), (72.5, 72.5),
                                             (179.0, 179.0), (179.5, 179.0), (250.0, 179.0)])
def test_target_is_limited_to_the_grid_before_the_upstream_call(monkeypatch, target, bounded):
    calls = []

    class StandIn:
        @staticmethod
        def cross_entropy_power_spectrum_DLDL_softmax2(*args):
            calls.append(args)
            return "kl", "ce", "error"

    monkeypatch.setattr(adapter, "upstream_loss_computer", lambda: StandIn)
    rppg = torch.zeros(160)
    assert adapter.dldl_losses(rppg, torch.tensor(target), 30.0, std=1.0) == ("kl", "ce", "error")
    (passed_rppg, passed_hr, fs, std), = calls
    assert passed_rppg is rppg and fs == 30.0 and std == 1.0
    assert float(passed_hr) == pytest.approx(bounded)


def _toolbox_with_pins(tmp_path, sha256):
    """A toolbox folder holding only a copy of the adapter whose pinned sha256 is replaced."""
    toolbox = tmp_path / "toolbox"
    target = toolbox / fetch.ADAPTER_RELPATH
    target.parent.mkdir(parents=True)
    text = (TOOLBOX / fetch.ADAPTER_RELPATH).read_text()
    target.write_text(text.replace(adapter.UPSTREAM_SHA256, sha256))
    return toolbox, toolbox / adapter.UPSTREAM_RELPATH


def test_fetch_script_writes_only_a_verified_file(tmp_path, monkeypatch, capsys, no_location_env):
    good = b"# stand-in for the pinned upstream file\n"
    toolbox, destination = _toolbox_with_pins(tmp_path, hashlib.sha256(good).hexdigest())
    downloads = []
    payload = {"data": b"tampered\n"}

    def fake_download(url, timeout):
        downloads.append(url)
        return payload["data"]

    monkeypatch.setattr(fetch, "download", fake_download)
    args = ["--toolbox", str(toolbox)]

    assert fetch.main(args + ["--check"]) == 1
    assert fetch.main(args) == 1
    assert not destination.exists() and not list(destination.parent.glob("*.part"))
    assert downloads == [adapter.UPSTREAM_URL]

    payload["data"] = good
    assert fetch.main(args) == 0 and destination.read_bytes() == good
    assert fetch.main(args + ["--check"]) == 0
    assert fetch.main(args) == 0 and len(downloads) == 2  # already verified: no second download

    destination.write_bytes(b"edited\n")
    assert fetch.main(args) == 1 and destination.read_bytes() == b"edited\n"  # not replaced without --force
    wrong = tmp_path / "wrong.py"
    wrong.write_bytes(b"other\n")
    assert fetch.main(args + ["--force", "--from-file", str(wrong)]) == 1
    assert destination.read_bytes() == b"edited\n"
    source = tmp_path / "copy.py"
    source.write_bytes(good)
    assert fetch.main(args + ["--force", "--from-file", str(source)]) == 0 and destination.read_bytes() == good
    assert len(downloads) == 2
    assert fetch.main(["--toolbox", str(tmp_path / "nothing")]) == 2
    capsys.readouterr()


def test_fetch_script_dest_and_environment(tmp_path, monkeypatch, capsys, no_location_env):
    good = b"# stand-in for the pinned upstream file\n"
    toolbox, default = _toolbox_with_pins(tmp_path, hashlib.sha256(good).hexdigest())
    monkeypatch.setattr(fetch, "download", lambda url, timeout: good)
    outside = (tmp_path / "outside" / "PhysFormerLossComputer.py").resolve()
    args = ["--toolbox", str(toolbox)]

    assert fetch.main(args + ["--dest", str(outside)]) == 0
    assert outside.read_bytes() == good and not default.exists()
    assert f"export {adapter.LOCATION_ENV}={outside}" in capsys.readouterr().out
    monkeypatch.setenv(adapter.LOCATION_ENV, str(outside))
    assert fetch.main(args + ["--check"]) == 0  # the location the toolbox reads is now the env path
    assert "export" not in capsys.readouterr().out and not default.exists()
    monkeypatch.setenv(adapter.LOCATION_ENV, "relative.py")
    assert fetch.main(args + ["--check"]) == 2
    capsys.readouterr()


def test_fetch_script_leaves_no_partial_file_when_the_write_fails(tmp_path, monkeypatch, capsys, no_location_env):
    good = b"# stand-in for the pinned upstream file\n"
    toolbox, destination = _toolbox_with_pins(tmp_path, hashlib.sha256(good).hexdigest())
    monkeypatch.setattr(fetch, "download", lambda url, timeout: good)

    def failing_replace(src, dst):
        raise OSError("simulated write failure")

    monkeypatch.setattr(fetch.os, "replace", failing_replace)
    assert fetch.main(["--toolbox", str(toolbox)]) == 1
    assert not destination.exists() and not list(destination.parent.glob("*.part"))
    assert "nothing installed" in capsys.readouterr().err


class _CpuTorch:
    """`torch` as seen by the upstream loss on a machine without CUDA: its hard-coded torch.device('cuda') and
    torch.cuda.FloatTensor mean the CPU (together with Tensor.cuda patched to a no-op). Test-only."""

    def __init__(self, real):
        self._real = real
        self.cuda = types.SimpleNamespace(FloatTensor=real.FloatTensor)

    def device(self, *args, **kwargs):
        return self._real.device("cpu")

    def __getattr__(self, name):
        return getattr(self._real, name)


def _input(case):
    x = np.random.default_rng(case["seed"]).standard_normal(REFERENCE["length"])
    if case["sine_bpm"] is not None:
        t = np.arange(REFERENCE["length"]) / REFERENCE["fs"]
        x = x + REFERENCE["sine_amplitude"] * np.sin(2 * np.pi * case["sine_bpm"] / 60.0 * t)
    x = ((x - x.mean()) / x.std()).astype(np.float32)
    assert hashlib.sha256(x.tobytes()).hexdigest() == case["input_sha256"], "input recipe no longer reproducible"
    return x


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: f"seed{c['seed']}-hr{c['target_bpm']}")
def test_adapter_reproduces_the_loss_of_the_manuscript_runs(case, monkeypatch):
    if not UPSTREAM_PRESENT:
        pytest.skip("upstream PhysFormerLossComputer.py not fetched (python tools/fetch_physformer_loss.py)")
    upstream_module = sys.modules[adapter.upstream_loss_computer().__module__]
    assert Path(upstream_module.__file__) == adapter.upstream_file()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        monkeypatch.setattr(upstream_module, "torch", _CpuTorch(torch))
        monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)

    x = torch.tensor(_input(case), device=device, requires_grad=True)
    hr = torch.tensor(case["target_bpm"], dtype=torch.float32, device=device)
    kl, ce, error = adapter.dldl_losses(x, hr, REFERENCE["fs"], std=REFERENCE["std"])
    (kl + ce).backward()
    grad = x.grad.detach().cpu().numpy().astype(np.float64)
    probe = np.random.default_rng(REFERENCE["probe_seed"]).standard_normal(REFERENCE["length"]).astype(np.float32)

    assert float(error.detach()) == case["hr_bin_error"]
    assert float(kl.detach()) == pytest.approx(case["kl"], rel=1e-5)
    assert float(ce.detach()) == pytest.approx(case["ce"], rel=1e-5)
    assert float(np.linalg.norm(grad)) == pytest.approx(case["grad_l2"], rel=1e-4)
    assert float(grad @ probe.astype(np.float64)) == pytest.approx(case["grad_probe"], rel=1e-3, abs=1e-7)


def test_reference_covers_targets_outside_the_grid():
    targets = [case["target_bpm"] for case in REFERENCE["cases"]]
    assert min(targets) < adapter.HR_MIN_BPM and max(targets) > adapter.HR_MAX_BPM
    assert any(adapter.HR_MIN_BPM <= t <= adapter.HR_MAX_BPM for t in targets)
    assert len({case["input_sha256"] for case in REFERENCE["cases"]}) == len(REFERENCE["cases"])
