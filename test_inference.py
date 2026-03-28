import os
os.environ['GPU_ARCHS'] = 'gfx1100'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

venv_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin")
if venv_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = venv_bin + ":" + os.environ.get("PATH", "")

import patch_gated_models
patch_gated_models.apply()

import sys
import torch
from PIL import Image
from trellis2 import pipelines

FULL_MODE = "--full" in sys.argv

print("Loading pipeline...")
pipeline = pipelines.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()
print(f"Pipeline loaded. GPU memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

image = Image.open("assets/example_image/T.png")
print(f"Input image: {image.size}")

if FULL_MODE:
    print("Running inference (full)...")
    mesh = pipeline.run(image)[0]
    print(f"Mesh generated: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
    print("Full inference test passed.")
else:
    print("Running inference (smoke test: sparse structure sampling)...")
    cond = pipeline.get_cond([image], resolution=512)
    sparse = pipeline.sample_sparse_structure(cond, 64)
    print(f"Sparse structure sampled: shape={sparse.shape}")
    print("Smoke test passed.")
