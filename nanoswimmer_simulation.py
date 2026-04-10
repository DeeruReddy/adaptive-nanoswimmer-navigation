from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


SEED = 7


def sphere_bgf(x, y, r=0.5):
    d2 = x**2 + y**2
    val = np.where(d2 <= r**2, 1.0, 1.0 - 0.02 * d2)
    return np.clip(val, 0.0, 1.0)


def matyas_bgf(x, y, r=0.5):
    d2 = x**2 + y**2
    val = np.where(d2 <= r**2, 1.0, 1.0 - 0.26 * d2 - (-0.48 * x * y))
    val = val / 100.0
    return np.clip(val, 0.0, 1.0)


def ackley_bgf(x, y, r=0.5):
    d2 = x**2 + y**2
    inside = d2 <= r**2
    a = 4.0 / 3.0
    b = 10.0
    c = 2 * math.pi
    val_out = a * (
        -np.exp(-np.sqrt(2 * (x**2 + y**2)) / b)
        - np.exp((np.cos(c * x) + np.cos(c * y)) / 2.0) / np.e
        + 1.0
    )
    val = np.where(inside, 1.0, val_out)
    val = (val - np.min(val)) / (np.max(val) - np.min(val) + 1e-9)
    return np.clip(val, 0.0, 1.0)


def easom_bgf(x, y, r=0.5):
    d2 = x**2 + y**2
    inside = d2 <= r**2
    val_out = 0.01 + 0.99 * np.cos(3 * x) * np.cos(3 * y) * np.exp(-(9 * x**2 + 9 * y**2))
    val = np.where(inside, 1.0, val_out)
    val = (val - np.min(val)) / (np.max(val) - np.min(val) + 1e-9)
    return np.clip(val, 0.0, 1.0)


BGF_FUNCTIONS = {
    "Sphere": sphere_bgf,
    "Matyas": matyas_bgf,
    "Ackley": ackley_bgf,
    "Easom": easom_bgf,
}


@dataclass
class NanoSwimmer:
    pos: np.ndarray
    idx: int
    alive: bool = True
    fitness: float = 0.0
    detected: bool = False
    role: str = "active"
    detected_as: str = "none"
    path: list = field(default_factory=list)

    def __post_init__(self):
        self.pos = np.array(self.pos, dtype=float)
        self.path = [tuple(self.pos)]

    def record(self):
        self.path.append(tuple(self.pos))


def clip_to_bounds(p, bound=5.0):
    return np.clip(p, -bound, bound)


def sample_bgf_at_positions(func, positions):
    x = positions[:, 0]
    y = positions[:, 1]
    return func(x, y)


def continuous_to_index(p, size=201, bound=5.0):
    x, y = p
    ix = int(((x + bound) / (2 * bound)) * (size - 1))
    iy = int(((y + bound) / (2 * bound)) * (size - 1))
    return max(0, min(size - 1, ix)), max(0, min(size - 1, iy))


def index_to_continuous(ix, iy, size=201, bound=5.0):
    x = (ix / (size - 1)) * (2 * bound) - bound
    y = (iy / (size - 1)) * (2 * bound) - bound
    return np.array([x, y], dtype=float)


def generate_vascular_mask(size=201, occupancy=0.5, fractal_seed=SEED):
    rng = np.random.default_rng(fractal_seed)
    target_nodes = int(size * size * occupancy)
    mask = np.zeros((size, size), dtype=bool)
    start = (rng.integers(0, size // 12 + 1), rng.integers(0, size // 12 + 1))
    mask[start] = True
    frontier = {start}

    while np.count_nonzero(mask) < target_nodes and frontier:
        current = frontier.pop()
        neighbors = []
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = current[0] + di, current[1] + dj
            if 0 <= ni < size and 0 <= nj < size and not mask[ni, nj]:
                neighbors.append((ni, nj))

        if not neighbors:
            continue

        scores = []
        for ni, nj in neighbors:
            toward_target = ni + nj
            randomness = rng.random()
            scores.append(toward_target + randomness * size * 0.2)

        nbr = neighbors[int(np.argmax(scores))]
        mask[nbr] = True
        frontier.add(current)
        frontier.add(nbr)

    mask[: max(3, size // 14), : max(3, size // 14)] = True
    target_band = slice(max(0, size // 2 - 5), min(size, size // 2 + 6))
    mask[target_band, target_band] = True
    return mask


def snap_to_vessel(pos, vessel_mask, bound=5.0, search_radius=4):
    size = vessel_mask.shape[0]
    ix, iy = continuous_to_index(pos, size=size, bound=bound)
    if vessel_mask[iy, ix]:
        return clip_to_bounds(pos, bound=bound)

    best = None
    best_dist = None
    for radius in range(1, search_radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx = ix + dx
                ny = iy + dy
                if 0 <= nx < size and 0 <= ny < size and vessel_mask[ny, nx]:
                    dist = dx * dx + dy * dy
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best = (nx, ny)
        if best is not None:
            break

    if best is None:
        return clip_to_bounds(pos, bound=bound)
    return index_to_continuous(best[0], best[1], size=size, bound=bound)


def estimate_local_gradient(func, pos, delta=0.05):
    x, y = pos
    f0 = func(x, y)
    fx = func(x + delta, y)
    fy = func(x, y + delta)
    gx = (fx - f0) / (delta + 1e-9)
    gy = (fy - f0) / (delta + 1e-9)
    angle = np.arctan2(gy, gx)
    return angle, float(f0)


def get_elite_direction(agents, func, elite_count):
    coords = np.array([a.pos for a in agents])
    vals = sample_bgf_at_positions(func, coords)
    for agent, value in zip(agents, vals):
        agent.fitness = float(value)

    sorted_agents = sorted(agents, key=lambda agent: agent.fitness, reverse=True)
    elites = sorted_agents[: max(1, elite_count)]
    angles = []
    weights = []
    for elite in elites:
        ang, _ = estimate_local_gradient(func, elite.pos)
        angles.append(ang)
        weights.append(elite.fitness + 1e-6)

    vecs = np.array([[math.cos(angle), math.sin(angle)] for angle in angles])
    mean_vec = np.sum(np.diag(weights) @ vecs, axis=0) / np.sum(weights)
    return math.atan2(mean_vec[1], mean_vec[0])


def get_chase_direction(agents, func, disadvantaged_count):
    coords = np.array([a.pos for a in agents])
    vals = np.array([a.fitness for a in agents])
    idxs = np.argsort(vals)[: max(1, disadvantaged_count)]
    low_coords = coords[idxs]

    if len(low_coords) <= 1:
        centroid = np.mean(coords, axis=0)
        if len(low_coords) == 0:
            return 0.0
        return math.atan2(centroid[1] - low_coords[0][1], centroid[0] - low_coords[0][0])

    pairwise = np.sum((low_coords[:, None, :] - low_coords[None, :, :]) ** 2, axis=2)
    adjacency = pairwise <= 0.3**2
    seen = set()
    clusters = []
    for idx in range(len(low_coords)):
        if idx in seen:
            continue
        stack = [idx]
        cluster = []
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            cluster.append(cur)
            neighbors = np.where(adjacency[cur])[0].tolist()
            stack.extend(neighbors)
        clusters.append(cluster)

    main_cluster = max(clusters, key=len)
    cluster_points = low_coords[main_cluster]
    cluster_centroid = np.mean(cluster_points, axis=0)
    elites_idx = np.argsort(vals)[-max(1, int(len(agents) * 0.2)) :]
    elite_centroid = np.mean(coords[elites_idx], axis=0)
    return math.atan2(
        elite_centroid[1] - cluster_centroid[1],
        elite_centroid[0] - cluster_centroid[0],
    )


class IARSSimulation:
    def __init__(
        self,
        bgf_name="Ackley",
        n_agents=60,
        grid_size=201,
        vessel_occupancy=0.5,
        so_steps=10,
        so_step_length=0.05,
        v_ma=0.5,
        elite_rate=0.35,
        disadvantage_ratio=0.2,
        target_radius=0.5,
        target_center=(0.0, 0.0),
        bound=5.0,
        seed=SEED,
    ):
        self.bgf_name = bgf_name
        self.func = BGF_FUNCTIONS[bgf_name]
        self.n_agents = n_agents
        self.grid_size = grid_size
        self.vessel_occupancy = vessel_occupancy
        self.so_steps = so_steps
        self.so_step_length = so_step_length
        self.v_ma = v_ma
        self.elite_rate = elite_rate
        self.disadvantage_ratio = disadvantage_ratio
        self.target_radius = target_radius
        self.target_center = np.array(target_center, dtype=float)
        self.bound = bound
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        self.vessel_mask = generate_vascular_mask(
            size=grid_size,
            occupancy=vessel_occupancy,
            fractal_seed=seed,
        )
        self.agents = []
        for i in range(n_agents):
            x = np.random.uniform(-5.0, -4.0)
            y = np.random.uniform(-5.0, -4.0)
            pos = snap_to_vessel(np.array([x, y]), self.vessel_mask, bound=self.bound)
            self.agents.append(NanoSwimmer(pos, i))

        if self.agents:
            coords = np.array([agent.pos for agent in self.agents])
            vals = sample_bgf_at_positions(self.func, coords)
            for agent, value in zip(self.agents, vals):
                agent.fitness = float(value)

        self.iteration = 0
        self.detection_history = []
        self.last_behavior = "forage"
        self.last_explanation = (
            "Elite swimmers have the highest current local fitness among alive swimmers, "
            "but that does not guarantee the shortest route through the vessel network."
        )

        self._assign_roles()

    def _assign_roles(self):
        alive_agents = [agent for agent in self.agents if agent.alive]
        for agent in self.agents:
            if not agent.detected:
                agent.role = "active"

        if not alive_agents:
            return

        ranked = sorted(alive_agents, key=lambda agent: agent.fitness, reverse=True)
        elite_count = max(1, int(self.elite_rate * len(alive_agents)))
        weak_count = max(1, int(self.disadvantage_ratio * len(alive_agents)))

        for agent in ranked[:elite_count]:
            agent.role = "elite"

        for agent in ranked[-weak_count:]:
            agent.role = "weak"

    def _group_summary(self):
        alive_agents = [agent for agent in self.agents if agent.alive]
        if not alive_agents:
            detected_as = [agent.detected_as for agent in self.agents if agent.detected]
            return {
                "elite_ids": set(),
                "weak_ids": set(),
                "alive_count": 0,
                "elite_count": 0,
                "weak_count": 0,
                "best_fitness": 0.0,
                "mean_fitness": 0.0,
                "elite_mean_fitness": 0.0,
                "weak_mean_fitness": 0.0,
                "mean_distance": 0.0,
                "elite_mean_distance": 0.0,
                "weak_mean_distance": 0.0,
                "detected_from_elite": detected_as.count("elite"),
                "detected_from_weak": detected_as.count("weak"),
                "detected_from_other": detected_as.count("active"),
            }

        elite_agents = [agent for agent in alive_agents if agent.role == "elite"]
        weak_agents = [agent for agent in alive_agents if agent.role == "weak"]
        alive_fitness = np.array([agent.fitness for agent in alive_agents], dtype=float)
        alive_distances = np.array(
            [np.linalg.norm(agent.pos - self.target_center) for agent in alive_agents],
            dtype=float,
        )
        detected_as = [agent.detected_as for agent in self.agents if agent.detected]

        return {
            "elite_ids": {agent.idx for agent in elite_agents},
            "weak_ids": {agent.idx for agent in weak_agents},
            "alive_count": len(alive_agents),
            "elite_count": len(elite_agents),
            "weak_count": len(weak_agents),
            "best_fitness": float(np.max(alive_fitness)),
            "mean_fitness": float(np.mean(alive_fitness)),
            "elite_mean_fitness": float(np.mean([agent.fitness for agent in elite_agents]))
            if elite_agents
            else 0.0,
            "weak_mean_fitness": float(np.mean([agent.fitness for agent in weak_agents]))
            if weak_agents
            else 0.0,
            "mean_distance": float(np.mean(alive_distances)),
            "elite_mean_distance": float(
                np.mean([np.linalg.norm(agent.pos - self.target_center) for agent in elite_agents])
            )
            if elite_agents
            else 0.0,
            "weak_mean_distance": float(
                np.mean([np.linalg.norm(agent.pos - self.target_center) for agent in weak_agents])
            )
            if weak_agents
            else 0.0,
            "detected_from_elite": detected_as.count("elite"),
            "detected_from_weak": detected_as.count("weak"),
            "detected_from_other": detected_as.count("active"),
        }

    def _build_explanation(self, group_summary):
        if group_summary["alive_count"] == 0:
            return "No active swimmers remain. Final detections are now grouped by the role each swimmer had when it reached the target."

        reasons = [
            "Elite means highest current local fitness among alive swimmers, not guaranteed first arrival.",
        ]

        if (
            group_summary["weak_count"] > 0
            and group_summary["elite_count"] > 0
            and group_summary["weak_mean_distance"] < group_summary["elite_mean_distance"]
        ):
            reasons.append(
                "Weak swimmers are currently closer to the target on average, so path geometry is helping them."
            )

        if group_summary["detected_from_weak"] > 0:
            reasons.append(
                "Some weak swimmers already reached the target, which shows route access can dominate local fitness rank."
            )

        if group_summary["detected_from_other"] > 0:
            reasons.append(
                "Some middle-group swimmers also reached the target, so elite and weak are not the only successful paths."
            )

        reasons.append(
            "Vessel constraints, shared drive direction, and random motion can delay an elite swimmer even when its fitness stays high."
        )
        return " ".join(reasons)

    def _detect(self, agent):
        if np.linalg.norm(agent.pos - self.target_center) <= self.target_radius:
            agent.detected = True
            agent.alive = False
            agent.detected_as = agent.role
            agent.role = "detected"

    def step(self):
        alive_agents = [agent for agent in self.agents if agent.alive]
        if not alive_agents:
            return self.snapshot(done=True)

        for agent in alive_agents:
            for _ in range(self.so_steps):
                flow_bias = np.array([0.02, 0.02])
                jitter = np.random.normal(scale=self.so_step_length, size=2)
                candidate = clip_to_bounds(agent.pos + flow_bias + jitter, bound=self.bound)
                agent.pos = snap_to_vessel(candidate, self.vessel_mask, bound=self.bound)
            agent.record()
            self._detect(agent)

        alive_agents = [agent for agent in self.agents if agent.alive]
        if not alive_agents:
            self.detection_history.append(self.n_agents)
            self.iteration += 1
            return self.snapshot(done=True)

        coords = np.array([agent.pos for agent in alive_agents])
        vals = sample_bgf_at_positions(self.func, coords)
        for agent, value in zip(alive_agents, vals):
            agent.fitness = float(value)
        self._assign_roles()

        elite_count = max(1, int(self.elite_rate * len(alive_agents)))
        theta_f = get_elite_direction(alive_agents, self.func, elite_count)
        disadvantaged_count = max(1, int(self.disadvantage_ratio * len(alive_agents)))
        theta_c = get_chase_direction(alive_agents, self.func, disadvantaged_count)
        thetas = np.array([estimate_local_gradient(self.func, agent.pos)[0] for agent in alive_agents])

        def circ_err(angle_a, angle_b):
            diff = (angle_a - angle_b + np.pi) % (2 * np.pi) - np.pi
            return diff**2

        delta_et = np.sum([circ_err(theta, theta_f) for theta in thetas])
        delta_lf = np.sum([circ_err(theta, theta_c) for theta in thetas])
        pet = delta_lf / (delta_et + delta_lf + 1e-9)
        plf = delta_et / (delta_et + delta_lf + 1e-9)
        self.last_behavior = "forage" if random.random() < plf else "chase"
        drive_theta = theta_f if self.last_behavior == "forage" else theta_c

        dt = 0.5
        step = self.v_ma * dt
        for agent in alive_agents:
            noise = np.random.normal(scale=0.02, size=2)
            move = np.array([math.cos(drive_theta), math.sin(drive_theta)]) * step + noise
            candidate = clip_to_bounds(agent.pos + move, bound=self.bound)
            agent.pos = snap_to_vessel(candidate, self.vessel_mask, bound=self.bound)
            agent.record()
            self._detect(agent)

        alive_agents = [agent for agent in self.agents if agent.alive]
        if alive_agents:
            coords = np.array([agent.pos for agent in alive_agents])
            vals = sample_bgf_at_positions(self.func, coords)
            for agent, value in zip(alive_agents, vals):
                agent.fitness = float(value)
        self._assign_roles()

        detected_count = sum(1 for agent in self.agents if agent.detected)
        self.detection_history.append(detected_count)
        self.iteration += 1
        return self.snapshot(done=detected_count == self.n_agents)

    def snapshot(self, done=False):
        group_summary = self._group_summary()
        self.last_explanation = self._build_explanation(group_summary)
        swimmers = []
        for agent in self.agents:
            group = "detected" if agent.detected else agent.role
            swimmers.append(
                {
                    "id": agent.idx,
                    "x": round(float(agent.pos[0]), 4),
                    "y": round(float(agent.pos[1]), 4),
                    "alive": agent.alive,
                    "detected": agent.detected,
                    "group": group,
                    "fitness": round(float(agent.fitness), 6),
                    "path": [[round(px, 4), round(py, 4)] for px, py in agent.path[-25:]],
                }
            )

        return {
            "iteration": self.iteration,
            "bgf": self.bgf_name,
            "behavior": self.last_behavior,
            "done": done,
            "detected_count": int(sum(1 for agent in self.agents if agent.detected)),
            "n_agents": self.n_agents,
            "target_center": [float(self.target_center[0]), float(self.target_center[1])],
            "target_radius": float(self.target_radius),
            "bound": float(self.bound),
            "alive_count": group_summary["alive_count"],
            "elite_count": group_summary["elite_count"],
            "weak_count": group_summary["weak_count"],
            "best_fitness": round(group_summary["best_fitness"], 6),
            "mean_fitness": round(group_summary["mean_fitness"], 6),
            "elite_mean_fitness": round(group_summary["elite_mean_fitness"], 6),
            "weak_mean_fitness": round(group_summary["weak_mean_fitness"], 6),
            "mean_distance": round(group_summary["mean_distance"], 6),
            "elite_mean_distance": round(group_summary["elite_mean_distance"], 6),
            "weak_mean_distance": round(group_summary["weak_mean_distance"], 6),
            "detected_from_elite": int(group_summary["detected_from_elite"]),
            "detected_from_weak": int(group_summary["detected_from_weak"]),
            "detected_from_other": int(group_summary["detected_from_other"]),
            "explanation": self.last_explanation,
            "vessel_mask": self.vessel_mask.astype(int).tolist() if self.iteration == 0 else None,
            "swimmers": swimmers,
        }


def build_simulation(payload: Optional[Dict[str, Any]] = None) -> IARSSimulation:
    payload = payload or {}
    return IARSSimulation(
        bgf_name=payload.get("bgf", "Ackley"),
        n_agents=int(payload.get("n_agents", 60)),
        grid_size=int(payload.get("grid_size", 201)),
        vessel_occupancy=float(payload.get("vessel_occupancy", 0.5)),
        so_steps=int(payload.get("so_steps", 8)),
        so_step_length=float(payload.get("so_step_length", 0.05)),
        v_ma=float(payload.get("v_ma", 0.8)),
        elite_rate=float(payload.get("elite_rate", 0.35)),
        disadvantage_ratio=float(payload.get("disadvantage_ratio", 0.2)),
        target_radius=float(payload.get("target_radius", 0.6)),
        seed=int(payload.get("seed", 7)),
    )


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>IARS Nanoswimmer Simulator</title>
  <style>
    :root {
      --bg: #f3efe4;
      --panel: rgba(255, 252, 246, 0.9);
      --ink: #17313e;
      --accent: #c75146;
      --accent-2: #1b7f79;
      --line: #d2c7b3;
      --target: #8f1d21;
      --elite: #1768ac;
      --weak: #c47f00;
      --active: #5f6b73;
      --detected: #8f1d21;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #fff7df 0%, transparent 28%),
        radial-gradient(circle at bottom right, #dceee6 0%, transparent 32%),
        linear-gradient(135deg, #efe8d7, #f8f4ec);
      min-height: 100vh;
    }

    .layout {
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 20px 36px;
    }

    .hero {
      display: flex;
      justify-content: space-between;
      gap: 20px;
      align-items: end;
      margin-bottom: 18px;
    }

    h1 {
      margin: 0;
      font-size: clamp(2rem, 4vw, 3.4rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }

    .subtitle {
      margin-top: 10px;
      max-width: 56ch;
      font-size: 1rem;
    }

    .panel {
      background: var(--panel);
      border: 1px solid rgba(23, 49, 62, 0.12);
      border-radius: 20px;
      box-shadow: 0 18px 40px rgba(33, 48, 39, 0.08);
      backdrop-filter: blur(12px);
    }

    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      align-items: end;
      padding: 16px;
      margin-bottom: 18px;
    }

    .field {
      display: flex;
      flex-direction: column;
      gap: 6px;
      min-width: 180px;
    }

    label {
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    select, button {
      border-radius: 12px;
      border: 1px solid var(--line);
      padding: 12px 14px;
      font: inherit;
      color: var(--ink);
      background: white;
    }

    button {
      cursor: pointer;
      min-width: 140px;
      transition: transform 120ms ease, background 120ms ease;
    }

    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }

    button.secondary {
      background: var(--accent-2);
      border-color: var(--accent-2);
      color: white;
    }

    button:hover { transform: translateY(-1px); }

    .viz {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 280px;
      gap: 18px;
    }

    canvas {
      width: 100%;
      aspect-ratio: 1 / 1;
      border-radius: 24px;
      border: 1px solid rgba(23, 49, 62, 0.08);
      background: linear-gradient(180deg, rgba(255,255,255,0.65), rgba(240,247,242,0.9));
      display: block;
    }

    .stats {
      padding: 18px;
      display: grid;
      gap: 14px;
      align-content: start;
    }

    .stat {
      border-top: 1px solid rgba(23, 49, 62, 0.12);
      padding-top: 12px;
    }

    .stat:first-child {
      border-top: 0;
      padding-top: 0;
    }

    .legend {
      display: grid;
      gap: 8px;
    }

    .legend-row {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 0.93rem;
    }

    .swatch {
      width: 14px;
      height: 14px;
      border-radius: 999px;
      border: 1px solid rgba(23, 49, 62, 0.18);
      flex: 0 0 auto;
    }

    .value {
      font-size: 2rem;
      font-weight: 700;
      line-height: 1;
    }

    .note {
      font-size: 0.93rem;
      color: rgba(23, 49, 62, 0.78);
    }

    @media (max-width: 920px) {
      .viz { grid-template-columns: 1fr; }
      .hero { align-items: start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <div class="hero">
      <div>
        <h1>IARS Nanoswimmer Simulator</h1>
        <div class="subtitle">
          The simulator now runs from the Python file instead of the notebook.
          Use the notebook for setup, graphs, and analysis, and use this page
          for interactive swarm visualization.
        </div>
      </div>
    </div>

    <div class="panel controls">
      <div class="field">
        <label for="bgfSelect">BGF Function</label>
        <select id="bgfSelect"></select>
      </div>
      <button id="startBtn" class="primary">Start</button>
      <button id="stepBtn" class="secondary">Step Once</button>
    </div>

    <div class="viz">
      <div class="panel">
        <canvas id="simCanvas" width="820" height="820"></canvas>
      </div>
      <div class="panel stats">
        <div class="stat">
          <div class="note">BGF Function</div>
          <div id="bgfName" class="value" style="font-size:1.35rem">Ackley</div>
        </div>
        <div class="stat">
          <div class="note">Iteration</div>
          <div id="iteration" class="value">0</div>
        </div>
        <div class="stat">
          <div class="note">Detected Swimmers</div>
          <div id="detected" class="value">0</div>
        </div>
        <div class="stat">
          <div class="note">Alive / Elite / Weak</div>
          <div id="groupCounts" class="value" style="font-size:1.15rem">0 / 0 / 0</div>
        </div>
        <div class="stat">
          <div class="note">Mean Fitness</div>
          <div id="meanFitness" class="value" style="font-size:1.35rem">0.000000</div>
        </div>
        <div class="stat">
          <div class="note">Elite Mean Fitness</div>
          <div id="eliteMeanFitness" class="value" style="font-size:1.2rem">0.000000</div>
        </div>
        <div class="stat">
          <div class="note">Weak Mean Fitness</div>
          <div id="weakMeanFitness" class="value" style="font-size:1.2rem">0.000000</div>
        </div>
        <div class="stat">
          <div class="note">Best Current Fitness</div>
          <div id="bestFitness" class="value" style="font-size:1.2rem">0.000000</div>
        </div>
        <div class="stat">
          <div class="note">Mean Distance To Target</div>
          <div id="meanDistance" class="value" style="font-size:1.2rem">0.000000</div>
        </div>
        <div class="stat">
          <div class="note">Elite / Weak Distance</div>
          <div id="groupDistances" class="value" style="font-size:1.05rem">0.000000 / 0.000000</div>
        </div>
        <div class="stat">
          <div class="note">Detected As Elite / Weak / Other</div>
          <div id="detectedByRole" class="value" style="font-size:1.05rem">0 / 0 / 0</div>
        </div>
        <div class="stat">
          <div class="note">Behavior Mode</div>
          <div id="behavior" class="value" style="font-size:1.35rem">idle</div>
        </div>
        <div class="stat legend">
          <div class="note">Swimmer Colors</div>
          <div class="legend-row"><span class="swatch" style="background: var(--elite);"></span>Elite swimmers</div>
          <div class="legend-row"><span class="swatch" style="background: var(--weak);"></span>Weak swimmers</div>
          <div class="legend-row"><span class="swatch" style="background: var(--active);"></span>Other active swimmers</div>
          <div class="legend-row"><span class="swatch" style="background: var(--detected);"></span>Detected at target cell</div>
        </div>
        <div class="stat">
          <div class="note">Why Elites Can Lag</div>
          <div id="explanation" class="note">Elite swimmers are ranked by current local fitness, not guaranteed shortest arrival path.</div>
        </div>
        <div class="stat">
          <div id="status" class="note">Press Start to initialize a new simulation run.</div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const canvas = document.getElementById("simCanvas");
    const ctx = canvas.getContext("2d");
    const bgfSelect = document.getElementById("bgfSelect");
    const startBtn = document.getElementById("startBtn");
    const stepBtn = document.getElementById("stepBtn");
    const bgfNameEl = document.getElementById("bgfName");
    const iterationEl = document.getElementById("iteration");
    const detectedEl = document.getElementById("detected");
    const groupCountsEl = document.getElementById("groupCounts");
    const meanFitnessEl = document.getElementById("meanFitness");
    const eliteMeanFitnessEl = document.getElementById("eliteMeanFitness");
    const weakMeanFitnessEl = document.getElementById("weakMeanFitness");
    const bestFitnessEl = document.getElementById("bestFitness");
    const meanDistanceEl = document.getElementById("meanDistance");
    const groupDistancesEl = document.getElementById("groupDistances");
    const detectedByRoleEl = document.getElementById("detectedByRole");
    const behaviorEl = document.getElementById("behavior");
    const explanationEl = document.getElementById("explanation");
    const statusEl = document.getElementById("status");

    let simState = null;
    let timer = null;
    const pad = 28;

    function worldToCanvas(x, y, bound) {
      const span = canvas.width - pad * 2;
      const px = pad + ((x + bound) / (2 * bound)) * span;
      const py = canvas.height - pad - ((y + bound) / (2 * bound)) * span;
      return [px, py];
    }

    function drawMask(mask, bound) {
      if (!mask) return;
      const rows = mask.length;
      const cols = mask[0].length;
      const w = (canvas.width - pad * 2) / cols;
      const h = (canvas.height - pad * 2) / rows;
      ctx.fillStyle = "rgba(27, 127, 121, 0.22)";
      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cols; x++) {
          if (!mask[y][x]) continue;
          const px = pad + x * w;
          const py = canvas.height - pad - (y + 1) * h;
          ctx.fillRect(px, py, Math.ceil(w), Math.ceil(h));
        }
      }
    }

    function drawGrid() {
      ctx.strokeStyle = "rgba(23,49,62,0.08)";
      ctx.lineWidth = 1;
      for (let i = 0; i <= 10; i++) {
        const t = pad + ((canvas.width - pad * 2) / 10) * i;
        ctx.beginPath();
        ctx.moveTo(t, pad);
        ctx.lineTo(t, canvas.height - pad);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(pad, t);
        ctx.lineTo(canvas.width - pad, t);
        ctx.stroke();
      }
    }

    function drawTarget(center, radius, bound) {
      const [cx, cy] = worldToCanvas(center[0], center[1], bound);
      const span = canvas.width - pad * 2;
      const pxRadius = (radius / (2 * bound)) * span;

      ctx.beginPath();
      ctx.fillStyle = "rgba(143, 29, 33, 0.16)";
      ctx.arc(cx, cy, pxRadius * 2.4, 0, Math.PI * 2);
      ctx.fill();

      ctx.beginPath();
      ctx.fillStyle = "#8f1d21";
      ctx.arc(cx, cy, pxRadius, 0, Math.PI * 2);
      ctx.fill();
    }

    function drawSwimmers(swimmers, bound) {
      const swimmerColors = {
        elite: "#1768ac",
        weak: "#c47f00",
        active: "#5f6b73",
        detected: "#8f1d21",
      };

      swimmers.forEach((swimmer) => {
        const color = swimmerColors[swimmer.group] || swimmerColors.active;
        const trail = swimmer.path || [];
        if (trail.length > 1) {
          ctx.beginPath();
          ctx.strokeStyle = color + "66";
          ctx.lineWidth = 1.5;
          trail.forEach(([x, y], index) => {
            const [px, py] = worldToCanvas(x, y, bound);
            if (index === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
          });
          ctx.stroke();
        }

        const [px, py] = worldToCanvas(swimmer.x, swimmer.y, bound);
        ctx.beginPath();
        ctx.fillStyle = color;
        ctx.arc(px, py, swimmer.alive ? 4.5 : 3.5, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    function render(state) {
      if (!state) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const bound = state.bound || 5;
      drawGrid();
      drawMask(state.vessel_mask || simState?.vessel_mask, bound);
      drawTarget(state.target_center, state.target_radius, bound);
      drawSwimmers(state.swimmers, bound);

      bgfNameEl.textContent = state.bgf;
      iterationEl.textContent = state.iteration;
      detectedEl.textContent = `${state.detected_count}/${state.n_agents}`;
      groupCountsEl.textContent = `${state.alive_count} / ${state.elite_count} / ${state.weak_count}`;
      meanFitnessEl.textContent = Number(state.mean_fitness || 0).toFixed(6);
      eliteMeanFitnessEl.textContent = Number(state.elite_mean_fitness || 0).toFixed(6);
      weakMeanFitnessEl.textContent = Number(state.weak_mean_fitness || 0).toFixed(6);
      bestFitnessEl.textContent = Number(state.best_fitness || 0).toFixed(6);
      meanDistanceEl.textContent = Number(state.mean_distance || 0).toFixed(6);
      groupDistancesEl.textContent = `${Number(state.elite_mean_distance || 0).toFixed(6)} / ${Number(state.weak_mean_distance || 0).toFixed(6)}`;
      detectedByRoleEl.textContent = `${state.detected_from_elite || 0} / ${state.detected_from_weak || 0} / ${state.detected_from_other || 0}`;
      behaviorEl.textContent = state.behavior;
      explanationEl.textContent = state.explanation || "";
      statusEl.textContent = state.done
        ? `Completed for ${state.bgf}: all detected or no active swimmers remain.`
        : `Running ${state.bgf}: fitness values refresh at every movement step.`;
    }

    async function loadBgfs() {
      const response = await fetch("/api/bgfs");
      const data = await response.json();
      data.bgfs.forEach((name) => {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        bgfSelect.appendChild(option);
      });
      bgfSelect.value = "Ackley";
    }

    async function resetSimulation() {
      const response = await fetch("/api/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ bgf: bgfSelect.value }),
      });
      simState = await response.json();
      render(simState);
    }

    async function stepSimulation() {
      const response = await fetch("/api/step", { method: "POST" });
      simState = await response.json();
      render(simState);
      if (simState.done && timer) {
        clearInterval(timer);
        timer = null;
        startBtn.textContent = "Start";
      }
    }

    startBtn.addEventListener("click", async () => {
      if (timer) {
        clearInterval(timer);
        timer = null;
        startBtn.textContent = "Start";
        statusEl.textContent = "Paused.";
        return;
      }

      await resetSimulation();
      startBtn.textContent = "Pause";
      timer = setInterval(stepSimulation, 140);
    });

    stepBtn.addEventListener("click", async () => {
      if (!simState) {
        await resetSimulation();
      } else {
        await stepSimulation();
      }
    });

    loadBgfs().then(resetSimulation);
  </script>
</body>
</html>
"""


def create_app(default_payload: Optional[Dict[str, Any]] = None):
    try:
        from flask import Flask, jsonify, render_template_string, request
    except ImportError as exc:
        raise RuntimeError(
            "Flask is required to run the web simulator. Install it in the notebook "
            "environment first, then run `python nanoswimmer_simulation.py`."
        ) from exc

    app = Flask(__name__)
    state = {"simulation": build_simulation(default_payload)}

    @app.get("/")
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.get("/api/bgfs")
    def bgfs():
        return jsonify({"bgfs": list(BGF_FUNCTIONS.keys())})

    @app.post("/api/reset")
    def reset():
        payload = request.get_json(silent=True)
        state["simulation"] = build_simulation(payload)
        return jsonify(state["simulation"].snapshot(done=False))

    @app.post("/api/step")
    def step():
        if state["simulation"] is None:
            state["simulation"] = build_simulation(default_payload)
        return jsonify(state["simulation"].step())

    return app


def main():
    app = create_app()
    port = int(os.environ.get("PORT", "5000"))
    app.run(debug=False, use_reloader=False, port=port)


if __name__ == "__main__":
    main()
