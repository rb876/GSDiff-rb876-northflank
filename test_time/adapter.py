"""Thin adapter around the original unconstrained RPLAN models and sampler math.

Source: scripts/test_main.py, especially lines 40-65, 197-247 and 317-387.
The upstream script has no importable sampler and runs dataset evaluation at import.
Both models support a batch dimension with 53 padded junction slots. Batch helpers
accept lists of [1, 53, 10] states and bound neural forwards by ``batch_size``;
room extraction still runs independently for each graph.
"""

from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import platform

import numpy as np
import torch

from .types import FloorplanSample, Room


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GSDiffAdapter:
    steps = 1000
    variant = "unconstrained_rplan"

    def __init__(self, repo=None, node_checkpoint=None, edge_checkpoint=None, device="cuda:0", variant="unconstrained_rplan"):
        if variant != self.variant:
            raise ValueError("v1 supports only variant='unconstrained_rplan'")
        self.repo = Path(repo or Path(__file__).resolve().parents[1]).resolve()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Run on Northflank, or explicitly use --device cpu.")
        paths = {
            "node": Path(node_checkpoint) if node_checkpoint else self.repo / "outputs/structure-1/model1000000.pt",
            "edge": Path(edge_checkpoint) if edge_checkpoint else self.repo / "outputs/structure-2/model_stage2_best_061000.pt",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing checkpoint(s): " + ", ".join(missing)
                                    + ". See test_time/README.md for the official download and Northflank paths.")
        from gsdiff.house_nn1 import HeterHouseModel
        from gsdiff.house_nn2 import EdgeModel
        from gsdiff import utils
        self.utils = utils
        self.node = HeterHouseModel()
        self.edge = EdgeModel()
        for name, model in (("node", self.node), ("edge", self.edge)):
            model.load_state_dict(torch.load(paths[name], map_location="cpu", weights_only=True), strict=True)
            model.to(self.device).eval().requires_grad_(False)
        self.metadata = {
            "variant": self.variant, "steps": self.steps,
            "checkpoints": {name: {"path": str(path), "sha256": file_hash(path)} for name, path in paths.items()},
            "node_shape": [1, 53, 10], "state_dtype": "float64",
            "drawing_resolution": 512, "coordinate_unit": "drawing_pixel",
            "merge_points": False, "align_points": False,
            "edge_rng_policy": "Clone current diffusion RNG for edge interpolation; common random numbers within a guided step.",
            "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                            "numpy": np.__version__, "cuda": torch.version.cuda,
                            "device": str(self.device),
                            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None},
        }
        alpha_bar = lambda t: math.cos(t / 1.000 * math.pi / 2) ** 2
        betas = np.array([min(1 - alpha_bar((i + 1) / self.steps) / alpha_bar(i / self.steps), 0.999)
                          for i in range(self.steps)], dtype=np.float64)
        alphas = 1.0 - betas
        cumulative = np.cumprod(alphas)
        previous = np.append(1.0, cumulative[:-1])
        coefficients = {
            "reciprocal": np.sqrt(1.0 / cumulative),
            "reciprocal_minus_one": np.sqrt(1.0 / cumulative - 1),
            "variance": betas * (1.0 - previous) / (1.0 - cumulative),
            "coefficient1": betas * np.sqrt(previous) / (1.0 - cumulative),
            "coefficient2": (1.0 - previous) * np.sqrt(alphas) / (1.0 - cumulative),
        }
        for name, values in coefficients.items():
            setattr(self, name, torch.tensor(values, device=self.device))
        self.node_mask = torch.ones((1, 53, 53), dtype=torch.uint8, device=self.device)
        self.counters = {"node_calls": 0, "edge_calls": 0, "graph_completions": 0,
                         "node_samples": 0, "edge_samples": 0}
        self.new_rng(0)

    def new_rng(self, seed):
        self.active_rng = torch.Generator(device=self.device).manual_seed(seed)
        return self.active_rng

    def initial_state(self, rng):
        return torch.randn((1, 53, 10), device=self.device, dtype=torch.float64, generator=rng)

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def predict_x0(self, state, t):
        """Predict clean states in one forward, accepting [B, 53, 10]."""
        count = state.shape[0]
        timestep = torch.full((count,), t, dtype=torch.long, device=self.device)
        prediction = torch.cat(self.node(state, self.node_mask.expand(count, -1, -1), timestep), dim=2)
        self.counters["node_calls"] += 1
        self.counters["node_samples"] += count
        x0 = (self.reciprocal[timestep][:, None, None].expand_as(state) * state
              - self.reciprocal_minus_one[timestep][:, None, None].expand_as(state) * prediction)
        x0[:, :, :2] = x0[:, :, :2].clamp(-1, 1)
        x0[:, :, 2:9] = x0[:, :, 2:9] >= 0.5
        x0[:, :, 9:10] = x0[:, :, 9:10] >= 0.75
        return x0

    @staticmethod
    def _check_batch(states, batch_size):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if any(state.shape != (1, 53, 10) for state in states):
            raise ValueError("Each state must have shape [1, 53, 10]")

    @torch.no_grad()
    def predict_x0_batch(self, states, t, batch_size=10):
        """Predict independent states with at most batch_size items per forward."""
        self._check_batch(states, batch_size)
        results = []
        for start in range(0, len(states), batch_size):
            prediction = self.predict_x0(torch.cat(states[start:start + batch_size], dim=0), t)
            results.extend(prediction.split(1, dim=0))
        return results

    @torch.no_grad()
    def sample_transition(self, state, t, rng, x0=None):
        if x0 is None:
            x0 = self.predict_x0(state, t)
        timestep = torch.tensor([t], device=self.device)
        mean = (self.coefficient1[timestep][:, None, None].expand_as(state) * x0
                + self.coefficient2[timestep][:, None, None].expand_as(state) * state)
        noise = torch.randn(state.shape, dtype=state.dtype, device=state.device, generator=rng)
        return mean + torch.sqrt(self.variance[timestep][:, None, None].expand_as(state)) * noise

    @torch.no_grad()
    def complete_graph(self, state, rng=None):
        return self.complete_graph_batch([state], [self.active_rng if rng is None else rng], batch_size=1)[0]

    @torch.no_grad()
    def predict_graph_tensors(self, states, rngs, batch_size=10):
        """Decode nodes and predict edges without transferring geometry to CPU.

        Valid junctions are compacted in their original order, matching the
        reference decoder. Room extraction and checks are left to gpu_graph and
        gpu_constraints. RNG state cloning preserves common proposal edge draws.
        """
        self._check_batch(states, batch_size)
        if len(states) != len(rngs):
            raise ValueError("Provide one trajectory RNG per state")
        parts = {name: [] for name in ("xy", "semantics", "node_mask", "adjacency")}
        positions = torch.arange(53, device=self.device)[None, :]
        for start in range(0, len(states), batch_size):
            state = torch.cat(states[start:start + batch_size], dim=0)
            size = state.shape[0]
            finite = torch.isfinite(state).all(dim=(1, 2))
            valid = (state[:, :, 9] == 0) & finite[:, None]
            order = torch.where(valid, positions, positions + 53).argsort(dim=1)
            count = valid.sum(dim=1)
            node_mask = positions < count[:, None]
            clean = torch.nan_to_num(state)
            xy = torch.trunc(clean[:, :, :2] * 256 + 256).gather(1, order[:, :, None].expand(-1, -1, 2))
            semantics = clean[:, :, 2:9].round().gather(1, order[:, :, None].expand(-1, -1, 7))
            xy = torch.where(node_mask[:, :, None], xy, 0)
            semantics = torch.where(node_mask[:, :, None], semantics, 0)
            adjacency = torch.zeros((size, 53, 53), dtype=torch.bool, device=self.device)
            eligible = (count >= 3).nonzero(as_tuple=False).flatten()
            if eligible.numel():
                masks = node_mask[eligible]
                corners = torch.where(masks[:, :, None], (xy[eligible] - 256) / 256, 0)
                interpolation = [torch.rand((1, 2809, 1), device=self.device,
                                            generator=torch.Generator(device=self.device).set_state(rng.get_state()))
                                 for rng in rngs[start:start + batch_size]]
                logits, _, _ = self.edge(corners, masks[:, :, None] & masks[:, None, :],
                                         masks[:, :, None].to(torch.uint8), semantics[eligible],
                                         random_lambda=torch.cat(interpolation, dim=0)[eligible])
                predicted = logits.softmax(dim=2).argmax(dim=2).reshape(-1, 53, 53).bool()
                adjacency[eligible] = torch.triu(predicted & masks[:, :, None] & masks[:, None, :])
                self.counters["edge_calls"] += 1
                self.counters["edge_samples"] += eligible.numel()
            self.counters["graph_completions"] += size
            for name, tensor in (("xy", xy), ("semantics", semantics),
                                 ("node_mask", node_mask), ("adjacency", adjacency)):
                parts[name].append(tensor)
        if not states:
            return {"xy": torch.empty((0, 53, 2), dtype=torch.float64, device=self.device),
                    "semantics": torch.empty((0, 53, 7), dtype=torch.float64, device=self.device),
                    "node_mask": torch.empty((0, 53), dtype=torch.bool, device=self.device),
                    "adjacency": torch.empty((0, 53, 53), dtype=torch.bool, device=self.device)}
        return {name: torch.cat(tensors, dim=0) for name, tensors in parts.items()}

    @torch.no_grad()
    def complete_graph_batch(self, states, rngs, batch_size=10):
        """Batch edge prediction, retaining a separate RNG and padding per graph.

        Interpolation draws are made from cloned trajectory RNG states. Reusing
        a generator for proposals gives common edge randomness without advancing
        that generator or the global RNG, independently of microbatch boundaries.
        """
        self._check_batch(states, batch_size)
        if len(states) != len(rngs):
            raise ValueError("Provide one trajectory RNG per state")
        samples, prepared = [], []
        for state, rng in zip(states, rngs):
            self.counters["graph_completions"] += 1
            sample = FloorplanSample(metadata=dict(self.metadata))
            samples.append(sample)
            if not torch.isfinite(state).all().item():
                sample.diagnostics.append("nonfinite_node_state")
                continue
            sample.raw_nodes = state[0].cpu().tolist()
            coordinates, semantics = self.utils.inverse_normalize_and_remove_padding_100_4testing(
                [state[:, :, :2]], [state[:, :, 2:9]], [state[:, :, 9:10].reshape(-1)], resolution=512)
            coordinates, semantics = coordinates[0], semantics[0]
            count = coordinates.shape[1]
            sample.nodes = np.concatenate((coordinates, semantics), axis=-1)[0].tolist()
            if count < 3:
                sample.diagnostics.append("fewer_than_three_junctions")
                continue
            prepared.append((sample, coordinates, semantics, rng))

        for start in range(0, len(prepared), batch_size):
            chunk = prepared[start:start + batch_size]
            size = len(chunk)
            corners = torch.zeros((size, 53, 2), dtype=torch.float64, device=self.device)
            node_semantics = torch.zeros((size, 53, 7), dtype=torch.float64, device=self.device)
            edge_mask = torch.zeros((size, 53, 53), dtype=torch.bool, device=self.device)
            padding_mask = torch.zeros((size, 53, 1), dtype=torch.uint8, device=self.device)
            interpolation = []
            for index, (sample, coordinates, semantics, rng) in enumerate(chunk):
                count = coordinates.shape[1]
                corners[index:index + 1, :count] = (
                    torch.tensor(coordinates, dtype=torch.float64, device=self.device) - 256) / 256
                node_semantics[index:index + 1, :count] = torch.tensor(semantics, dtype=torch.float64, device=self.device)
                edge_mask[index, :count, :count] = True
                padding_mask[index, :count] = 1
                rng_state = rng.get_state()
                edge_rng = torch.Generator(device=self.device).set_state(rng_state)
                interpolation.append(torch.rand((1, 2809, 1), device=self.device, generator=edge_rng))
                sample.metadata["edge_rng_state_sha256"] = hashlib.sha256(rng_state.numpy().tobytes()).hexdigest()
            logits, _, _ = self.edge(corners, edge_mask, padding_mask, node_semantics,
                                     random_lambda=torch.cat(interpolation, dim=0))
            self.counters["edge_calls"] += 1
            self.counters["edge_samples"] += size
            adjacency = logits.softmax(dim=2).argmax(dim=2).reshape(size, 53, 53).cpu().numpy()
            for index, (sample, coordinates, semantics, _) in enumerate(chunk):
                count = coordinates.shape[1]
                self._finish_graph(sample, coordinates, semantics, adjacency[index, :count, :count])
        return samples

    def _finish_graph(self, sample, coordinates, semantics, adjacency):
        sample.raw_edges = np.argwhere(np.triu(adjacency) == 1).tolist()
        semantic_indices = np.where(semantics == 1, np.arange(7), 99999)
        points = [tuple(row) for row in np.concatenate((coordinates, semantic_indices), axis=-1)[0].tolist()]
        edges = self.utils.edges_to_coordinates(np.triu(adjacency).reshape(-1), points)
        point_index = {point: index for index, point in enumerate(points)}
        import networkx as nx
        try:
            _, polygons, labels = self.utils.get_cycle_basis_and_semantic_3_semansimplified(points, edges)
            sample.rooms = [Room(int(label), [[float(p[0]), float(p[1])] for p in polygon])
                            for polygon, label in zip(polygons, labels)]
            if len(polygons) != len(labels):
                sample.diagnostics.append("room_label_count_mismatch")
        except (ValueError, KeyError, IndexError, nx.NetworkXException) as error:
            sample.diagnostics.append(f"room_extraction_failed: {type(error).__name__}: {error}")
        sample.edges = [[point_index[start], point_index[end]] for start, end in edges]
        return sample

    def sample_baseline(self, seed, scenario=None):
        rng = self.new_rng(seed)
        state = self.initial_state(rng)
        for t in range(self.steps - 1, -1, -1):
            state = self.sample_transition(state, t, rng)
        sample = self.complete_graph(state)
        sample.metadata.update(seed=seed, method="baseline")
        if scenario is not None:
            sample.scenario = asdict(scenario)
        return sample

    def parameter_hash(self):
        """Used once before/after an experiment to verify frozen parameters."""
        digest = hashlib.sha256()
        for model in (self.node, self.edge):
            for name, tensor in model.state_dict().items():
                digest.update(name.encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
