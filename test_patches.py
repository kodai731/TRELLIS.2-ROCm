"""
Test patches in isolation without loading the full TRELLIS pipeline.
Runs in ~5 seconds.
"""
import os
os.environ['GPU_ARCHS'] = 'gfx1100'

venv_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin")
if venv_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = venv_bin + ":" + os.environ.get("PATH", "")

import patch_gated_models
patch_gated_models.apply()

import torch


def test_grid_sample_3d():
    from flex_gemm.ops.grid_sample import grid_sample_3d

    G, C = 16, 6
    coords_list = []
    feats_list = []
    for x in range(G):
        for y in range(G):
            for z in range(G):
                if (x - G // 2) ** 2 + (y - G // 2) ** 2 + (z - G // 2) ** 2 < (G // 3) ** 2:
                    coords_list.append([0, x, y, z])
                    feats_list.append([float(x) / G, float(y) / G, float(z) / G, 1.0, 0.5, 0.3])

    N = len(coords_list)
    coords = torch.tensor(coords_list, dtype=torch.int32, device='cuda')
    feats = torch.tensor(feats_list, dtype=torch.float32, device='cuda')

    center = torch.tensor([[G / 2.0, G / 2.0, G / 2.0]], device='cuda')
    query = center.unsqueeze(0)
    shape = torch.Size([1, C, G, G, G])

    out = grid_sample_3d(feats, coords, shape, query, mode='trilinear')
    assert out.shape == (1, 1, C), f"Expected (1, 1, {C}), got {out.shape}"
    assert not torch.isnan(out).any()
    assert out.abs().max() > 0.01, f"Output is near-zero: {out}"

    R = G // 3
    L = 100
    angles = torch.rand(L, 2, device='cuda') * 2 * 3.14159
    radii = torch.rand(L, device='cuda') * R * 0.8
    cx, cy, cz = G / 2.0, G / 2.0, G / 2.0
    query_inside = torch.stack([
        cx + radii * angles[:, 0].cos() * angles[:, 1].sin(),
        cy + radii * angles[:, 0].sin() * angles[:, 1].sin(),
        cz + radii * angles[:, 1].cos(),
    ], dim=-1).unsqueeze(0)
    out_inside = grid_sample_3d(feats, coords, shape, query_inside, mode='trilinear')
    assert out_inside.shape == (1, L, C)
    assert not torch.isnan(out_inside).any()
    nonzero_ratio = (out_inside.abs() > 0.01).float().mean().item()
    assert nonzero_ratio > 0.8, f"Too many near-zero outputs inside sphere: {nonzero_ratio:.1%}"
    print(f"  grid_sample_3d: OK (N={N} voxels, center={out[0,0,:3].tolist()}, inside_nonzero={nonzero_ratio:.0%})")


def test_flash_attn_varlen():
    from flash_attn import flash_attn_varlen_qkvpacked_func

    S, H, D = 64, 8, 64
    qkv = torch.randn(S, 3, H, D, device='cuda', dtype=torch.float16)
    cu_seqlens = torch.tensor([0, S], device='cuda', dtype=torch.int32)

    out = flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, S)
    assert out.shape == (S, H, D), f"Expected ({S}, {H}, {D}), got {out.shape}"
    assert not torch.isnan(out).any()
    print(f"  flash_attn varlen: OK (shape={out.shape})")


def test_birefnet_dtype():
    from trellis2.pipelines.rembg.BiRefNet import BiRefNet
    assert BiRefNet.__call__ is patch_gated_models._patched_birefnet_call
    print("  birefnet dtype patch: OK")


def test_dinov3_timm():
    from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor
    assert DinoV3FeatureExtractor is patch_gated_models.DinoV3FeatureExtractorTimm
    print("  dinov3 timm patch: OK")


if __name__ == "__main__":
    print("Testing patches...")
    test_grid_sample_3d()
    test_flash_attn_varlen()
    test_birefnet_dtype()
    test_dinov3_timm()
    print("All patch tests passed.")
