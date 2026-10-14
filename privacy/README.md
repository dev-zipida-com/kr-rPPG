# Privacy: BFDT reference implementation

`bfdt_transform.py` is an independent re-implementation of the block-wise frequency-domain transform
(BFDT) of Huang, Ou & Xiong (Biomed. Signal Process. Control 108, 107893, 2025): every frame is split
into 8 x 8 blocks, a 2-D DCT is applied per block, only the lowest-frequency coefficients (zig-zag order)
are kept, and the frame is reconstructed. The KR-rPPG release uses keep ratio 0.02, which for 8 x 8 blocks
keeps only the DC coefficient (the block mean). The release builder (`release_builder/krrppg_release/bfdt.py`)
uses a fast path for that setting; its tests check it against this reference.

The privacy evaluation of the manuscript (Table 9, closed-set ArcFace re-identification) is described in the
paper. It needs the original, untransformed videos, so it cannot be reproduced with the released data, and its
code is not part of this repository. The protection of the released data relies on the BFDT transform together
with controlled access under the data use agreement (DUA).
