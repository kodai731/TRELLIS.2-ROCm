"""
Monkey-patch for ROCm compatibility with TRELLIS.2.

Usage:
    import patch_gated_models
    patch_gated_models.apply()

Patches:
  - DinoV3: facebook/dinov3-vitl16-pretrain-lvd1689m -> timm (same weights, non-gated)
  - RMBG:   briaai/RMBG-2.0 -> ZhengPeng7/BiRefNet (same arch, MIT, non-gated)
  - BiRefNet: dtype-safe __call__ (fp16 model + float32 input)
  - grid_sample: Triton indice_weighed_sum -> pure PyTorch (gfx1100 Triton autotuner workaround)

To revert: simply remove the apply() call or delete this file.
"""

from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image

TIMM_MODEL_MAP = {
    "facebook/dinov3-vitl16-pretrain-lvd1689m": "vit_large_patch16_dinov3.lvd1689m",
}

REMBG_MODEL_MAP = {
    "briaai/RMBG-2.0": "ZhengPeng7/BiRefNet",
}


class DinoV3FeatureExtractorTimm:
    def __init__(self, model_name: str, image_size=512):
        import timm

        self.model_name = model_name
        timm_name = TIMM_MODEL_MAP.get(model_name, model_name)
        self.model = timm.create_model(timm_name, pretrained=True)
        self.model.norm = nn.Identity()
        self.model.eval()
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def to(self, device):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(next(self.model.parameters()).dtype)
        hidden_states = self.model.forward_features(image)
        return F.layer_norm(hidden_states, hidden_states.shape[-1:])

    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image)
            image = [i.resize((self.image_size, self.image_size), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        image = self.transform(image).cuda()
        return self.extract_features(image)


_original_birefnet_init = None
_original_birefnet_call = None


def _patched_birefnet_init(self, model_name="ZhengPeng7/BiRefNet"):
    mapped_name = REMBG_MODEL_MAP.get(model_name, model_name)
    _original_birefnet_init(self, mapped_name)


def _patched_birefnet_call(self, image):
    from torchvision import transforms as T

    image_size = image.size
    input_images = self.transform_image(image).unsqueeze(0).to("cuda")

    model_dtype = next(self.model.parameters()).dtype
    input_images = input_images.to(dtype=model_dtype)

    with torch.no_grad():
        preds = self.model(input_images)[-1].sigmoid().cpu().float()

    pred = preds[0].squeeze()
    pred_pil = T.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    return image


def _patch_grid_sample_3d():
    from flex_gemm.ops.grid_sample.grid_sample import GridSample3dFunction

    @staticmethod
    def _trilinear_fwd_gpu(ctx, feats, coords, shape, query_pts):
        N = coords.shape[0]
        B, L = query_pts.shape[:2]
        C_feat = feats.shape[1]
        _, W, H, D = shape[-4:]

        coord_keys = coords[:, 1].long() * (H * D) + coords[:, 2].long() * D + coords[:, 3].long()
        sorted_keys, sort_perm = coord_keys.sort()
        inv_perm = torch.empty_like(sort_perm)
        inv_perm[sort_perm] = torch.arange(N, device=feats.device)
        sorted_feats = feats[sort_perm]

        HD = int(H * D)

        neigh_offsets = torch.tensor([
            [-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5],
            [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5],
            [0.5, -0.5, -0.5], [0.5, -0.5, 0.5],
            [0.5, 0.5, -0.5], [0.5, 0.5, 0.5],
        ], device=feats.device)

        q_flat = query_pts.reshape(-1, 3)
        M = q_flat.shape[0]

        neigh_pts = (q_flat.unsqueeze(1) + neigh_offsets.unsqueeze(0)).int()

        weights = torch.prod(
            1.0 - torch.abs(neigh_pts.float() + 0.5 - q_flat.unsqueeze(1)),
            dim=-1,
        )

        nx = neigh_pts[..., 0].clamp(0, W - 1).long()
        ny = neigh_pts[..., 1].clamp(0, H - 1).long()
        nz = neigh_pts[..., 2].clamp(0, D - 1).long()
        flat_query = nx * HD + ny * D + nz

        flat_query_flat = flat_query.reshape(-1)
        search_pos = torch.searchsorted(sorted_keys, flat_query_flat)
        search_pos = search_pos.clamp(0, N - 1)
        found = sorted_keys[search_pos] == flat_query_flat

        valid = found.reshape(M, 8)
        weights = weights * valid.float()

        safe_pos = torch.where(found, search_pos, torch.zeros_like(search_pos))
        gathered = sorted_feats[safe_pos.reshape(-1)].reshape(M, 8, C_feat)

        weight_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
        out = (weights.unsqueeze(-1) * gathered).sum(dim=1) / weight_sum

        ctx.save_for_backward(torch.empty(0), torch.empty(0))
        ctx.N = N
        ctx.C = C_feat
        return out.reshape(B, L, C_feat)

    GridSample3dFunction._trilinear_fwd = _trilinear_fwd_gpu
    print("[patch] grid_sample_3d trilinear -> pure PyTorch (searchsorted)")


def apply():
    global _original_birefnet_init, _original_birefnet_call

    from trellis2.modules import image_feature_extractor
    image_feature_extractor.DinoV3FeatureExtractor = DinoV3FeatureExtractorTimm
    print("[patch] DinoV3 -> timm backend")

    from trellis2.pipelines.rembg.BiRefNet import BiRefNet
    _original_birefnet_init = BiRefNet.__init__
    _original_birefnet_call = BiRefNet.__call__
    BiRefNet.__init__ = _patched_birefnet_init
    BiRefNet.__call__ = _patched_birefnet_call
    print("[patch] RMBG-2.0 -> ZhengPeng7/BiRefNet (dtype-safe)")

    _patch_grid_sample_3d()
