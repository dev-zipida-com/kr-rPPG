"""Smoke tests: every benchmark architecture imports and runs a forward pass on random input with random
weights, in the shapes used by the benchmark configs. No trained weights are distributed with this repository.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "third_party" / "rPPG-Toolbox"))

MODELS = ("physnet", "physformer", "deepphys", "tscan", "efficientphys")


def build(model: str):
    """Model + random input with the shapes used by the benchmark configs (batch of 1, CPU)."""
    torch.manual_seed(0)
    if model == "physnet":
        from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
        return PhysNet_padding_Encoder_Decoder_MAX(frames=128), (torch.rand(1, 3, 128, 72, 72),), (1, 128)
    if model == "physformer":
        from neural_methods.model.PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp
        net = ViT_ST_ST_Compact3_TDC_gra_sharp(image_size=(160, 128, 128), patches=(4,) * 3, dim=96, ff_dim=144,
                                               num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7)
        return net, (torch.rand(1, 3, 160, 128, 128), 0.7), (1, 160)
    if model == "deepphys":
        from neural_methods.model.DeepPhys import DeepPhys
        return DeepPhys(img_size=72), (torch.rand(20, 6, 72, 72),), (20, 1)
    if model == "tscan":
        from neural_methods.model.TS_CAN import TSCAN
        return TSCAN(frame_depth=20, img_size=36), (torch.rand(20, 6, 36, 36),), (20, 1)
    if model == "efficientphys":
        from neural_methods.model.EfficientPhys import EfficientPhys
        # the trainer appends the last frame once, so N + 1 frames give N outputs (N multiple of frame_depth)
        return EfficientPhys(frame_depth=10, img_size=72), (torch.rand(11, 3, 72, 72),), (10, 1)
    raise ValueError(model)


def _run(net, inputs):
    net.eval()
    with torch.no_grad():
        out = net(*inputs)
    return out[0] if isinstance(out, tuple) else out


@pytest.mark.parametrize("model", MODELS)
def test_forward_pass_on_random_input(model):
    net, inputs, shape = build(model)
    out = _run(net, inputs)
    assert tuple(out.shape) == shape
    assert torch.isfinite(out).all()
