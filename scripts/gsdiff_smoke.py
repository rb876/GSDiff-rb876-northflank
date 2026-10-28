#!/usr/bin/env python3
"""One original, unconstrained GSDiff sample; no dataset, guidance, or FID/KID."""

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def checkpoint_path(explicit, repo, stage, pattern):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = sorted({p.resolve()
                         for base in (repo / "outputs", repo / "scripts" / "outputs")
                         if base.exists()
                         for p in base.rglob(pattern)
                         if stage in p.parts})
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one {stage} checkpoint, found {candidates}; "
                           "provide the explicit checkpoint argument.")
    return candidates[0]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/workspace/GSDiff"))
    parser.add_argument("--node-checkpoint")
    parser.add_argument("--edge-checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    if not (repo / "gsdiff" / "house_nn1.py").is_file():
        raise FileNotFoundError(f"GSDiff source not found in {repo}")
    sys.path.insert(0, str(repo))

    import cv2
    import networkx as nx
    import numpy as np
    import PIL
    import torch
    from PIL import Image, ImageDraw
    from gsdiff.house_nn1 import HeterHouseModel
    from gsdiff.house_nn2 import EdgeModel
    from gsdiff.utils import (
        edges_to_coordinates,
        get_cycle_basis_and_semantic_3_semansimplified,
        inverse_normalize_and_remove_padding_100_4testing,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing a CPU fallback.")
    device = torch.device("cuda:0")
    node_path = checkpoint_path(args.node_checkpoint, repo, "structure-1", "model100*.pt")
    edge_path = checkpoint_path(args.edge_checkpoint, repo, "structure-2", "model_stage2_best_061000.pt")
    started = datetime.now(timezone.utc)
    output = args.output_dir or Path("/workspace/results") / started.strftime("smoke-%Y%m%dT%H%M%S-%fZ")
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    wall_start = time.perf_counter()
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    metadata = {
        "started_utc": started.isoformat(), "seed": args.seed,
        "repo": str(repo), "repo_commit": commit,
        "script_sha256": sha256(Path(__file__).resolve()),
        "checkpoints": {
            "node": {"path": str(node_path), "sha256": sha256(node_path)},
            "edge": {"path": str(edge_path), "sha256": sha256(edge_path)},
        },
        "environment": {
            "python": platform.python_version(), "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "numpy": np.__version__, "networkx": nx.__version__,
            "opencv": cv2.__version__, "pillow": PIL.__version__,
            "gpu": torch.cuda.get_device_name(device),
            "gpu_capability": list(torch.cuda.get_device_capability(device)),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
        "sampler": {"mode": "unconstrained", "steps": 1000, "batch_size": 1,
                    "slots": 53, "state_dtype": "float64", "resolution": 512,
                    "merge_points": False, "align_points": False,
                    "eval_mode": True, "inference_mode": True},
        "units": "drawing coordinates; no physical scale assigned",
    }
    print(f"Output: {output}\nGPU: {metadata['environment']['gpu']}", flush=True)
    node = HeterHouseModel()
    edge = EdgeModel()
    node.load_state_dict(torch.load(node_path, map_location="cpu", weights_only=True), strict=True)
    edge.load_state_dict(torch.load(edge_path, map_location="cpu", weights_only=True), strict=True)
    node.to(device).eval().requires_grad_(False)
    edge.to(device).eval().requires_grad_(False)
    # Seed after parameter initialization/loading; the seed controls sampling itself.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    steps = 1000
    alpha_bar = lambda t: math.cos(t / 1.000 * math.pi / 2) ** 2
    betas = np.array([min(1 - alpha_bar((i + 1) / steps) / alpha_bar(i / steps), 0.999)
                      for i in range(steps)], dtype=np.float64)
    alphas = 1.0 - betas
    cumulative = np.cumprod(alphas)
    previous = np.append(1.0, cumulative[:-1])
    # Same float64 coefficients as scripts/test_main.py, transferred once.
    reciprocal = torch.tensor(np.sqrt(1.0 / cumulative), device=device)
    reciprocal_minus_one = torch.tensor(np.sqrt(1.0 / cumulative - 1), device=device)
    variance = torch.tensor(betas * (1.0 - previous) / (1.0 - cumulative), device=device)
    coefficient1 = torch.tensor(betas * np.sqrt(previous) / (1.0 - cumulative), device=device)
    coefficient2 = torch.tensor((1.0 - previous) * np.sqrt(alphas) / (1.0 - cumulative), device=device)
    node_mask = torch.ones((1, 53, 53), dtype=torch.uint8, device=device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    generation_start = time.perf_counter()

    with torch.inference_mode():
        state = torch.randn((1, 53, 10), device=device, dtype=torch.float64)
        for step in range(steps - 1, -1, -1):
            timestep = torch.tensor([step], device=device)
            prediction = torch.cat(node(state, node_mask, timestep), dim=2)
            x0 = (reciprocal[timestep][:, None, None].expand_as(state) * state
                  - reciprocal_minus_one[timestep][:, None, None].expand_as(state) * prediction)
            x0[:, :, :2] = x0[:, :, :2].clamp(-1, 1)
            x0[:, :, 2:9] = x0[:, :, 2:9] >= 0.5
            x0[:, :, 9:10] = x0[:, :, 9:10] >= 0.75
            mean = (coefficient1[timestep][:, None, None].expand_as(state) * x0
                    + coefficient2[timestep][:, None, None].expand_as(state) * state)
            state = mean + torch.sqrt(variance[timestep][:, None, None].expand_as(state)) * torch.randn_like(state)
            if (steps - step) % 200 == 0:
                torch.cuda.synchronize(device)
                print(f"DDPM {steps - step}/{steps}; elapsed {time.perf_counter() - generation_start:.1f}s", flush=True)

        coordinates, semantics = inverse_normalize_and_remove_padding_100_4testing(
            [state[:, :, :2]], [state[:, :, 2:9]], [state[:, :, 9:10].reshape(-1)], resolution=512,
        )
        coordinates, semantics = coordinates[0], semantics[0]
        count = coordinates.shape[1]
        if count < 3:
            np.savez_compressed(output / "invalid_nodes.npz", normalized_nodes=state.cpu().numpy())
            raise RuntimeError(f"Only {count} junctions generated; invalid sample saved in {output}")
        corners = torch.zeros((1, 53, 2), dtype=torch.float64, device=device)
        corners[:, :count] = (torch.tensor(coordinates, dtype=torch.float64, device=device) - 256) / 256
        node_semantics = torch.zeros((1, 53, 7), dtype=torch.float64, device=device)
        node_semantics[:, :count] = torch.tensor(semantics, dtype=torch.float64, device=device)
        edge_mask = torch.zeros((1, 53, 53), dtype=torch.bool, device=device)
        edge_mask[:, :count, :count] = True
        padding_mask = torch.zeros((1, 53, 1), dtype=torch.uint8, device=device)
        padding_mask[:, :count] = 1
        logits, _, _ = edge(corners, edge_mask, padding_mask, node_semantics)
        # Equivalent to original one_hot -> edges_remove_padding -> class-one matrix.
        adjacency = logits.softmax(dim=2).argmax(dim=2).reshape(53, 53)[:count, :count].cpu().numpy()
        normalized_nodes = state.cpu().numpy()
    torch.cuda.synchronize(device)
    metadata["generation_seconds"] = time.perf_counter() - generation_start
    metadata["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    metadata["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved(device)

    semantic_indices = np.where(semantics == 1, np.arange(7), 99999)
    points = [tuple(row) for row in np.concatenate((coordinates, semantic_indices), axis=-1)[0].tolist()]
    upper_adjacency = np.triu(adjacency)
    raw_edge_indices = np.argwhere(upper_adjacency == 1)
    edges = edges_to_coordinates(upper_adjacency.reshape(-1), points)
    raw_edges = list(edges)
    # Save the original prediction before the upstream cycle routine mutates edges.
    np.savez_compressed(output / "raw_prediction.npz", normalized_nodes=normalized_nodes,
                        coordinates=coordinates[0], semantics=semantics[0],
                        predicted_adjacency=adjacency, raw_edge_indices=raw_edge_indices)
    reverse, polygons, labels = get_cycle_basis_and_semantic_3_semansimplified(points, edges)
    areas = [abs(sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1]
                     for i in range(len(p)))) / 2 for p in polygons]
    renderable = bool(polygons) and len(polygons) == len(labels) and all(
        len(p) >= 3 and area > 0 for p, area in zip(polygons, areas)
    )
    metadata.update({"junctions": count, "raw_edges": len(raw_edges),
                     "postprocessed_edges": len(edges), "rooms": len(polygons),
                     "nonempty_room_render_check": renderable,
                     "validity_note": "Nonempty positive-area rendering is not a complete structural-validity check.",
                     "elapsed_seconds": time.perf_counter() - wall_start})
    graph = {"metadata": metadata, "output_points_test": points,
             "raw_output_edges_test": raw_edges, "raw_edge_indices": raw_edge_indices.tolist(),
             "output_edges_test": edges, "d_rev_test": reverse,
             "simple_cycles_test": polygons, "simple_cycles_semantics_test": labels}
    def json_default(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    (output / "graph.json").write_text(json.dumps(graph, indent=2, default=json_default, allow_nan=False) + "\n")
    if not renderable:
        raise RuntimeError(f"No valid nonempty room rendering; raw prediction and graph saved in {output}")

    colors = {0: (244, 241, 222), 1: (234, 182, 159), 2: (107, 112, 92),
              3: (224, 122, 95), 4: (95, 121, 123), 5: (242, 204, 143), 6: (0, 0, 0)}
    image = Image.new("RGB", (512, 512), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for polygon, label in zip(polygons, labels):
        draw.polygon([(point[0], point[1]) for point in polygon],
                     fill=colors[int(label)], outline=None)
    # Same aa_scale=1 wall rendering as the original script.
    for polygon in polygons:
        for index in range(len(polygon) - 1):
            p1, p2 = polygon[index], polygon[index + 1]
            draw.rectangle((p1[0] - 3, p1[1] - 3, p1[0] + 3, p1[1] + 3), fill=(150, 150, 150))
            draw.line((p1[0], p1[1], p2[0], p2[1]), fill=(150, 150, 150), width=7)
    if image.getextrema() == ((255, 255), (255, 255), (255, 255)):
        raise RuntimeError(f"Unexpected blank rendering; graph saved in {output}")
    image.save(output / "floorplan.png")
    print(f"Saved {output}: {count} junctions, {len(edges)} edges, {len(polygons)} rooms; "
          f"generation={metadata['generation_seconds']:.2f}s; "
          f"peak GPU allocated={metadata['peak_gpu_allocated_bytes'] / 1024**3:.3f} GiB; "
          f"total={time.perf_counter() - wall_start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
