"""
TRELLIS.2 mesh generation worker.

Called as a subprocess from the gRPC server's model_trellis.py.

Usage (one-shot):
    python trellis_worker.py --image input.png --output output.glb

Usage (daemon):
    python trellis_worker.py --daemon
    Reads JSON requests from stdin, writes results to stdout.
"""

import argparse
import json
import os
import sys
import time

os.environ["GPU_ARCHS"] = "gfx1100"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

venv_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin")
if venv_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = venv_bin + ":" + os.environ.get("PATH", "")

READY_MARKER = "__READY__"
SHUTDOWN_COMMAND = "__SHUTDOWN__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daemon", action="store_true", help="Run as persistent daemon")
    parser.add_argument("--image", help="Input image path (one-shot mode)")
    parser.add_argument("--output", help="Output GLB path (one-shot mode)")
    parser.add_argument("--target-faces", type=int, default=150000)
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_pipeline_cpu():
    import patch_gated_models
    patch_gated_models.apply()

    from trellis2 import pipelines

    pipeline = pipelines.from_pretrained("microsoft/TRELLIS.2-4B")
    return pipeline


def load_pipeline():
    pipeline = load_pipeline_cpu()
    pipeline.cuda()
    return pipeline


def generate_mesh(pipeline, image_path: str, seed: int):
    import torch
    from PIL import Image

    image = Image.open(image_path)

    if seed == 0:
        seed = torch.randint(0, 2**31, (1,)).item()

    mesh = pipeline.run(image, seed=seed)[0]
    return mesh


def export_glb(mesh, output_path: str, target_faces: int, texture_size: int):
    import o_voxel

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=target_faces,
        texture_size=texture_size,
        remesh=False,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
        use_tqdm=True,
    )
    glb.export(output_path)


def count_mesh_stats(output_path: str) -> tuple[int, int]:
    import trimesh
    mesh = trimesh.load(output_path, force="mesh")
    return len(mesh.vertices), len(mesh.faces)


def _process_request(pipeline, request: dict) -> dict:
    import torch

    image_path = request["image"]
    output_path = request["output"]
    target_faces = request.get("target_faces", 150000)
    texture_size = request.get("texture_size", 2048)
    seed = request.get("seed", 0)

    t0 = time.monotonic()
    pipeline.cuda()

    t1 = time.monotonic()
    mesh = generate_mesh(pipeline, image_path, seed)
    t_generate = time.monotonic() - t1

    t2 = time.monotonic()
    export_glb(mesh, output_path, target_faces, texture_size)
    t_export = time.monotonic() - t2

    vertex_count, face_count = count_mesh_stats(output_path)

    pipeline.cpu()
    torch.cuda.empty_cache()

    total_ms = (time.monotonic() - t0) * 1000.0
    return {
        "status": "ok",
        "output": output_path,
        "vertex_count": vertex_count,
        "face_count": face_count,
        "total_ms": total_ms,
        "load_ms": 0.0,
        "generate_ms": t_generate * 1000.0,
        "export_ms": t_export * 1000.0,
    }


def daemon_main():
    t0 = time.monotonic()
    pipeline = load_pipeline_cpu()
    load_ms = (time.monotonic() - t0) * 1000.0

    sys.stderr.write(f"TRELLIS pipeline loaded to CPU in {load_ms:.0f}ms\n")
    sys.stderr.flush()

    print(READY_MARKER, flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == SHUTDOWN_COMMAND:
            break

        try:
            request = json.loads(line)
            result = _process_request(pipeline, request)
        except Exception as e:
            result = {"status": "error", "message": str(e)}

        print(f"__RESULT__{json.dumps(result)}__END_RESULT__", flush=True)


def main():
    args = parse_args()

    if args.daemon:
        daemon_main()
        return

    if not args.image or not args.output:
        print("Error: --image and --output required in one-shot mode", file=sys.stderr)
        sys.exit(1)

    t0 = time.monotonic()
    pipeline = load_pipeline()
    t_load = time.monotonic() - t0

    t1 = time.monotonic()
    mesh = generate_mesh(pipeline, args.image, args.seed)
    t_generate = time.monotonic() - t1

    t2 = time.monotonic()
    export_glb(mesh, args.output, args.target_faces, args.texture_size)
    t_export = time.monotonic() - t2

    vertex_count, face_count = count_mesh_stats(args.output)
    total_ms = (time.monotonic() - t0) * 1000.0

    result = {
        "status": "ok",
        "output": args.output,
        "vertex_count": vertex_count,
        "face_count": face_count,
        "total_ms": total_ms,
        "load_ms": t_load * 1000.0,
        "generate_ms": t_generate * 1000.0,
        "export_ms": t_export * 1000.0,
    }
    print(f"\n__RESULT__{json.dumps(result)}__END_RESULT__")


if __name__ == "__main__":
    main()
