from __future__ import annotations

import copy
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from mcevrp_data import build_extended_data


Node = str
Vehicle = str
Action = int


@dataclass
class SearchResult:
    best_solution: dict[Vehicle, list[Node]]
    best_cost: float
    best_trace: list[tuple[int, float]]
    iteration_stats: list[dict]
    update_stats: list[dict]


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def conjugate_gradient(fvp_fn, b: np.ndarray, nsteps: int = 10, residual_tol: float = 1e-10) -> np.ndarray:
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rdotr = float(r @ r)
    for _ in range(nsteps):
        z = fvp_fn(p)
        denom = float(p @ z) + 1e-12
        alpha = rdotr / denom
        x += alpha * p
        r -= alpha * z
        new_rdotr = float(r @ r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / (rdotr + 1e-12)
        p = r + beta * p
        rdotr = new_rdotr
    return x


def flatten_params(W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.concatenate([W.ravel(), b.ravel()])


def unflatten_params(theta: np.ndarray, state_dim: int, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    split = state_dim * action_dim
    W = theta[:split].reshape(state_dim, action_dim)
    b = theta[split:].reshape(action_dim)
    return W, b


class LinearTRPOPolicy:

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_kl: float = 0.02,
        cg_steps: int = 12,
        damping: float = 1e-2,
        line_search_steps: int = 10,
        line_search_decay: float = 0.7,
        seed: int = 7,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_kl = max_kl
        self.cg_steps = cg_steps
        self.damping = damping
        self.line_search_steps = line_search_steps
        self.line_search_decay = line_search_decay
        self.rng = np.random.default_rng(seed)

        self.W = 0.01 * self.rng.standard_normal((state_dim, action_dim))
        self.b = np.zeros(action_dim, dtype=float)
        self.value_w = np.zeros(state_dim, dtype=float)

    def probs(self, state: np.ndarray, theta: np.ndarray | None = None) -> np.ndarray:
        W, b = (self.W, self.b) if theta is None else unflatten_params(theta, self.state_dim, self.action_dim)
        logits = state @ W + b
        return softmax(logits)

    def sample_action(self, state: np.ndarray) -> tuple[Action, np.ndarray]:
        probs = self.probs(state)
        action = int(self.rng.choice(self.action_dim, p=probs))
        return action, probs

    def predict_values(self, states: np.ndarray) -> np.ndarray:
        return states @ self.value_w

    def fit_value(self, states: np.ndarray, returns: np.ndarray, ridge: float = 1e-4) -> None:
        xtx = states.T @ states + ridge * np.eye(states.shape[1])
        xty = states.T @ returns
        self.value_w = np.linalg.solve(xtx, xty)

    def grad_log_prob(self, state: np.ndarray, action: int, probs: np.ndarray | None = None) -> np.ndarray:
        probs = self.probs(state) if probs is None else probs
        diff = -probs
        diff[action] += 1.0
        grad_W = np.outer(state, diff)
        grad_b = diff
        return flatten_params(grad_W, grad_b)

    def surrogate_and_kl(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        old_probs_batch: np.ndarray,
        theta_new: np.ndarray,
    ) -> tuple[float, float]:
        obj_terms = []
        kl_terms = []
        for s, a, adv, old_probs in zip(states, actions, advantages, old_probs_batch):
            new_probs = self.probs(s, theta_new)
            ratio = new_probs[a] / max(old_probs[a], 1e-12)
            obj_terms.append(ratio * adv)
            kl_terms.append(np.sum(old_probs * (np.log(old_probs + 1e-12) - np.log(new_probs + 1e-12))))
        return float(np.mean(obj_terms)), float(np.mean(kl_terms))

    def update(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, old_probs_batch: np.ndarray) -> dict:
        returns = rewards.astype(float)
        self.fit_value(states, returns)
        baseline = self.predict_values(states)
        advantages = returns - baseline
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        grad = np.zeros(self.state_dim * self.action_dim + self.action_dim, dtype=float)
        fisher = np.zeros((grad.size, grad.size), dtype=float)
        for s, a, adv, old_probs in zip(states, actions, advantages, old_probs_batch):
            g = self.grad_log_prob(s, int(a), old_probs)
            grad += adv * g
            fisher += np.outer(g, g)
        grad /= len(states)
        fisher /= len(states)
        fisher += self.damping * np.eye(fisher.shape[0])

        def fvp_fn(v: np.ndarray) -> np.ndarray:
            return fisher @ v

        step_dir = conjugate_gradient(fvp_fn, grad, nsteps=self.cg_steps)
        shs = float(step_dir @ fvp_fn(step_dir))
        if shs <= 1e-12:
            return {"surrogate": 0.0, "kl": 0.0, "accepted": False}

        step_scale = math.sqrt(2.0 * self.max_kl / (shs + 1e-12))
        full_step = step_scale * step_dir

        theta_old = flatten_params(self.W, self.b)
        surrogate_old, _ = self.surrogate_and_kl(states, actions, advantages, old_probs_batch, theta_old)

        accepted = False
        best_theta = theta_old
        best_surrogate = surrogate_old
        best_kl = 0.0
        for step_idx in range(self.line_search_steps):
            frac = self.line_search_decay ** step_idx
            theta_candidate = theta_old + frac * full_step
            surrogate_new, kl_new = self.surrogate_and_kl(states, actions, advantages, old_probs_batch, theta_candidate)
            if kl_new <= self.max_kl and surrogate_new >= surrogate_old:
                accepted = True
                best_theta = theta_candidate
                best_surrogate = surrogate_new
                best_kl = kl_new
                break

        if accepted:
            self.W, self.b = unflatten_params(best_theta, self.state_dim, self.action_dim)
        return {"surrogate": float(best_surrogate), "kl": float(best_kl), "accepted": accepted}


class MCEVRPRouteEvaluator:
    def __init__(self, data):
        self.data = data
        self.depot = data["depot"]
        self.customers = set(data["customers"])
        self.stations = set(data["stations"])
        self.vehicles = data["vehicles"]
        self.products = data["products"]
        self.alpha = data["alpha"]
        self.beta = data["beta"]
        self.travel_time = data["travel_time"]
        self.route_cost = data["route_cost"]
        self.demand = data["demand"]
        self.vehicle_capacity = data["vehicle_capacity"]
        self.compartment_capacity = data["compartment_capacity"]
        self.battery_capacity = data["battery_capacity"]
        self.safety_soc = data["safety_soc"]
        self.charge_price = data["charge_price"]
        self.charge_time_weight = data["charge_time_weight"]
        self.tardiness_weight = data["tardiness_weight"]
        self.fixed_cost = data["fixed_cost"]
        self.service_time = data["service_time"]
        self.time_window = data["time_window"]
        self.rho = data["rho"]
        self.gamma = data["gamma"]
        self.breakpoint_names = sorted(data["breakpoints"])
        self.penalty = 1e4

    def charge_time_from_soc(self, station: Node, soc: float) -> float:
        points = sorted((self.rho[b], self.gamma[(station, b)]) for b in self.breakpoint_names)
        if soc <= points[0][0]:
            return points[0][1]
        if soc >= points[-1][0]:
            return points[-1][1]
        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
            if x0 <= soc <= x1:
                if abs(x1 - x0) < 1e-12:
                    return y1
                lam = (soc - x0) / (x1 - x0)
                return y0 + lam * (y1 - y0)
        return points[-1][1]

    def initial_solution(self) -> dict[Vehicle, list[Node]]:
        routes = {v: [] for v in self.vehicles}
        remaining_cap = {v: self.vehicle_capacity[v] for v in self.vehicles}
        sorted_customers = sorted(
            self.customers,
            key=lambda c: sum(self.demand[c][p] for p in self.products),
            reverse=True,
        )
        for cust in sorted_customers:
            demand_sum = sum(self.demand[cust][p] for p in self.products)
            feasible_vehicles = [
                v
                for v in self.vehicles
                if remaining_cap[v] >= demand_sum
                and self._route_product_load(routes[v] + [cust], "A") <= self.compartment_capacity[(v, "L1")]
                and self._route_product_load(routes[v] + [cust], "B") <= self.compartment_capacity[(v, "L2")]
            ]
            chosen = min(feasible_vehicles or self.vehicles, key=lambda v: remaining_cap[v], default=self.vehicles[0])
            routes[chosen].append(cust)
            remaining_cap[chosen] -= demand_sum
        return routes

    def _route_product_load(self, route: list[Node], product: str) -> float:
        return sum(self.demand[node][product] for node in route if node in self.customers)

    def _greedy_target_soc(self, vehicle: Vehicle, current_soc: float, suffix: Sequence[Node], remaining: dict[str, float]) -> float:
        if len(suffix) <= 1:
            return min(self.battery_capacity[vehicle], max(current_soc + 0.1, self.safety_soc[vehicle]))
        needed_energy = 0.0
        current = suffix[0]
        shadow_remaining = remaining.copy()
        for nxt in suffix[1:]:
            if current == nxt or (current, nxt) not in self.alpha:
                break
            arc_load = shadow_remaining["A"] + shadow_remaining["B"]
            needed_energy += self.alpha[(current, nxt)] + self.beta[(current, nxt)] * arc_load
            if nxt in self.customers:
                shadow_remaining["A"] -= self.demand[nxt]["A"]
                shadow_remaining["B"] -= self.demand[nxt]["B"]
            if nxt in self.stations:
                break
            current = nxt
        target = max(current_soc + 0.1, needed_energy + self.safety_soc[vehicle])
        return min(self.battery_capacity[vehicle], target)

    def evaluate(self, solution: dict[Vehicle, list[Node]]) -> tuple[float, dict]:
        total_cost = 0.0
        violations = 0
        tardiness_total = 0.0
        battery_violations = 0
        used_vehicles = 0
        visited_customers = []
        route_details = {}

        for vehicle, route in solution.items():
            if route:
                explicit_route = [self.depot] + route + [self.depot]
                used_vehicles += 1
                total_cost += self.fixed_cost[vehicle]
            else:
                route_details[vehicle] = {"detail": [], "charge_stops": 0}
                continue

            station_visits = [n for n in route if n in self.stations]
            repeated_station_visits = len(station_visits) - len(set(station_visits))
            if repeated_station_visits > 0:
                violations += repeated_station_visits
                total_cost += self.penalty * repeated_station_visits

            prod_a = self._route_product_load(route, "A")
            prod_b = self._route_product_load(route, "B")
            total_load = prod_a + prod_b

            if total_load > self.vehicle_capacity[vehicle] + 1e-9:
                violations += 1
                total_cost += self.penalty * (total_load - self.vehicle_capacity[vehicle])
            if prod_a > self.compartment_capacity[(vehicle, "L1")] + 1e-9:
                violations += 1
                total_cost += self.penalty * (prod_a - self.compartment_capacity[(vehicle, "L1")])
            if prod_b > self.compartment_capacity[(vehicle, "L2")] + 1e-9:
                violations += 1
                total_cost += self.penalty * (prod_b - self.compartment_capacity[(vehicle, "L2")])

            remaining = {"A": prod_a, "B": prod_b}
            soc = self.battery_capacity[vehicle]
            time = 0.0
            charge_stops = 0
            detail = []

            for idx in range(len(explicit_route) - 1):
                i, j = explicit_route[idx], explicit_route[idx + 1]
                if i == j or (i, j) not in self.alpha:
                    violations += 1
                    total_cost += self.penalty
                    continue
                arc_load = remaining["A"] + remaining["B"]
                travel_energy = self.alpha[(i, j)] + self.beta[(i, j)] * arc_load
                soc_after = soc - travel_energy
                total_cost += self.route_cost[(i, j)]
                time += self.travel_time[(i, j)]

                if soc_after < self.safety_soc[vehicle] - 1e-9:
                    battery_violations += 1
                    violations += 1
                    total_cost += self.penalty * (self.safety_soc[vehicle] - soc_after)
                soc = soc_after

                step_info = {
                    "from": i,
                    "to": j,
                    "time": time,
                    "soc": soc,
                    "load_a": remaining["A"],
                    "load_b": remaining["B"],
                }

                if j in self.customers:
                    visited_customers.append(j)
                    open_t, close_t = self.time_window[j]
                    if time < open_t:
                        time = open_t
                    tard = max(0.0, time - close_t)
                    tardiness_total += tard
                    total_cost += self.tardiness_weight * tard
                    time += self.service_time[j]
                    remaining["A"] -= self.demand[j]["A"]
                    remaining["B"] -= self.demand[j]["B"]
                    step_info["tardiness"] = tard

                elif j in self.stations:
                    charge_stops += 1
                    target_soc = self._greedy_target_soc(vehicle, soc, explicit_route[idx + 1 :], remaining)
                    if target_soc > soc + 1e-9:
                        charge_amount = target_soc - soc
                        charge_duration = self.charge_time_from_soc(j, target_soc) - self.charge_time_from_soc(j, soc)
                        soc = target_soc
                        time += charge_duration
                        total_cost += self.charge_price[j] * charge_amount
                        total_cost += self.charge_time_weight * charge_duration
                        step_info["charge_amount"] = charge_amount
                        step_info["charge_time"] = charge_duration
                    else:
                        violations += 1
                        total_cost += self.penalty * 0.1

                detail.append(step_info)
            route_details[vehicle] = {"detail": detail, "charge_stops": charge_stops}

        missing = len(self.customers - set(visited_customers))
        duplicates = max(0, len(visited_customers) - len(set(visited_customers)))
        if missing or duplicates:
            violations += missing + duplicates
            total_cost += self.penalty * (missing + duplicates)

        info = {
            "violations": violations,
            "battery_violations": battery_violations,
            "tardiness_total": tardiness_total,
            "used_vehicles": used_vehicles,
            "route_details": route_details,
        }
        return total_cost, info

    def verify_model_requirements(self, solution: dict[Vehicle, list[Node]]) -> dict:
        cost, info = self.evaluate(solution)
        route_customer_counts = {c: 0 for c in self.customers}
        station_repeat_violations = 0
        unknown_node_violations = 0
        vehicle_capacity_violations = 0
        compartment_capacity_violations = 0

        for vehicle, route in solution.items():
            seen_stations = set()
            prod_a = self._route_product_load(route, "A")
            prod_b = self._route_product_load(route, "B")
            total_load = prod_a + prod_b
            if total_load > self.vehicle_capacity[vehicle] + 1e-9:
                vehicle_capacity_violations += 1
            if prod_a > self.compartment_capacity[(vehicle, "L1")] + 1e-9:
                compartment_capacity_violations += 1
            if prod_b > self.compartment_capacity[(vehicle, "L2")] + 1e-9:
                compartment_capacity_violations += 1
            for node in route:
                if node in self.customers:
                    route_customer_counts[node] += 1
                elif node in self.stations:
                    if node in seen_stations:
                        station_repeat_violations += 1
                    seen_stations.add(node)
                else:
                    unknown_node_violations += 1

        missing_customers = sum(1 for c, cnt in route_customer_counts.items() if cnt == 0)
        duplicate_customers = sum(max(0, cnt - 1) for cnt in route_customer_counts.values())
        feasible = (
            info["violations"] == 0
            and missing_customers == 0
            and duplicate_customers == 0
            and station_repeat_violations == 0
            and unknown_node_violations == 0
            and vehicle_capacity_violations == 0
            and compartment_capacity_violations == 0
        )
        return {
            "feasible": feasible,
            "cost": cost,
            "violations": info["violations"],
            "battery_violations": info["battery_violations"],
            "tardiness_total": info["tardiness_total"],
            "used_vehicles": info["used_vehicles"],
            "missing_customers": missing_customers,
            "duplicate_customers": duplicate_customers,
            "station_repeat_violations": station_repeat_violations,
            "unknown_node_violations": unknown_node_violations,
            "vehicle_capacity_violations": vehicle_capacity_violations,
            "compartment_capacity_violations": compartment_capacity_violations,
        }


class TabuTRPOSearch:
    ACTION_NAMES = ["intra_swap", "inter_relocate", "inter_swap", "two_opt", "station_toggle"]

    def __init__(
        self,
        data,
        seed: int = 7,
        tabu_tenure: int = 12,
        neighborhood_samples: int = 20,
        update_every: int = 20,
        trpo_max_kl: float = 0.02,
        policy_mode: str = "trpo",
        trpo_top_actions: int = 3,
        intensify_every: int = 10,
        stagnation_limit: int = 20,
    ):
        self.data = data
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.evaluator = MCEVRPRouteEvaluator(data)
        self.tabu_tenure = tabu_tenure
        self.neighborhood_samples = neighborhood_samples
        self.update_every = update_every
        self.policy = LinearTRPOPolicy(
            state_dim=8,
            action_dim=len(self.ACTION_NAMES),
            max_kl=trpo_max_kl,
            seed=seed,
        )
        self.policy_mode = policy_mode
        self.trpo_top_actions = max(1, min(trpo_top_actions, len(self.ACTION_NAMES)))
        self.intensify_every = max(1, intensify_every)
        self.stagnation_limit = max(2, stagnation_limit)
        self.customer_list = list(data["customers"])
        self.station_list = list(data["stations"])
        self.vehicles = data["vehicles"]

    def _copy_solution(self, solution):
        return {v: list(route) for v, route in solution.items()}

    def _solution_signature(self, solution):
        return tuple((v, tuple(route)) for v, route in sorted(solution.items()))

    def _all_customer_positions(self, solution):
        positions = []
        for v, route in solution.items():
            for idx, node in enumerate(route):
                if node in self.evaluator.customers:
                    positions.append((v, idx, node))
        return positions

    def state_features(self, current_cost: float, current_info: dict, best_cost: float, iteration: int, max_iter: int) -> np.ndarray:
        violation_scale = max(1.0, len(self.customer_list))
        used_ratio = current_info["used_vehicles"] / max(1, len(self.vehicles))
        violation_ratio = current_info["violations"] / violation_scale
        battery_ratio = current_info["battery_violations"] / violation_scale
        tardiness_ratio = current_info["tardiness_total"] / 100.0
        progress = iteration / max(1, max_iter)
        gap = (current_cost - best_cost) / max(1.0, abs(best_cost))
        normalized_cost = current_cost / 100.0
        avg_customers_per_used = len(self.customer_list) / max(1, current_info["used_vehicles"])
        features = np.array(
            [
                normalized_cost,
                used_ratio,
                violation_ratio,
                battery_ratio,
                tardiness_ratio,
                progress,
                gap,
                avg_customers_per_used / max(1, len(self.customer_list)),
            ],
            dtype=float,
        )
        return features

    def move_signature(self, action: int, meta: dict) -> tuple:
        if action == 0:
            return ("swap", meta["vehicle"], tuple(sorted([meta["node_a"], meta["node_b"]])))
        if action == 1:
            return ("relocate", meta["node"], meta["from_vehicle"], meta["to_vehicle"])
        if action == 2:
            return ("inter_swap", tuple(sorted([meta["node_a"], meta["node_b"]])))
        if action == 3:
            return ("two_opt", meta["vehicle"], meta["start"], meta["end"])
        return ("station_toggle", meta["vehicle"], meta.get("index", -1), meta.get("station", ""))

    def neighbor_intra_swap(self, solution):
        candidates = []
        for v, route in solution.items():
            customer_positions = [idx for idx, n in enumerate(route) if n in self.evaluator.customers]
            if len(customer_positions) < 2:
                continue
            for _ in range(min(self.neighborhood_samples, len(customer_positions) * 2)):
                i, j = sorted(self.rng.sample(customer_positions, 2))
                cand = self._copy_solution(solution)
                cand[v][i], cand[v][j] = cand[v][j], cand[v][i]
                candidates.append((cand, {"vehicle": v, "node_a": route[i], "node_b": route[j]}))
        return candidates

    def neighbor_inter_relocate(self, solution):
        candidates = []
        positions = self._all_customer_positions(solution)
        if not positions:
            return candidates
        for _ in range(self.neighborhood_samples):
            from_v, from_idx, node = self.rng.choice(positions)
            to_v = self.rng.choice(self.vehicles)
            cand = self._copy_solution(solution)
            cand[from_v].pop(from_idx)
            if to_v == from_v:
                valid_positions = [idx for idx in range(len(cand[to_v]) + 1) if idx != from_idx]
                if not valid_positions:
                    continue
                insert_idx = self.rng.choice(valid_positions)
            else:
                insert_idx = self.rng.randint(0, len(cand[to_v]))
            cand[to_v].insert(insert_idx, node)
            candidates.append((cand, {"node": node, "from_vehicle": from_v, "to_vehicle": to_v}))
        return candidates

    def neighbor_inter_swap(self, solution):
        candidates = []
        positions = self._all_customer_positions(solution)
        if len(positions) < 2:
            return candidates
        for _ in range(self.neighborhood_samples):
            (v1, i1, n1), (v2, i2, n2) = self.rng.sample(positions, 2)
            if v1 == v2 and i1 == i2:
                continue
            cand = self._copy_solution(solution)
            cand[v1][i1], cand[v2][i2] = cand[v2][i2], cand[v1][i1]
            candidates.append((cand, {"node_a": n1, "node_b": n2}))
        return candidates

    def neighbor_two_opt(self, solution):
        candidates = []
        for v, route in solution.items():
            customer_indices = [idx for idx, n in enumerate(route) if n in self.evaluator.customers]
            if len(customer_indices) < 3:
                continue
            for _ in range(self.neighborhood_samples):
                i, j = sorted(self.rng.sample(customer_indices, 2))
                cand = self._copy_solution(solution)
                cand[v][i : j + 1] = reversed(cand[v][i : j + 1])
                candidates.append((cand, {"vehicle": v, "start": i, "end": j}))
        return candidates

    def neighbor_station_toggle(self, solution):
        candidates = []
        if not self.station_list:
            return candidates
        for station in self.station_list:
            for v, route in solution.items():
                station_indices = [idx for idx, n in enumerate(route) if n == station]
                if station_indices:
                    for idx in station_indices:
                        cand = self._copy_solution(solution)
                        cand[v].pop(idx)
                        candidates.append((cand, {"vehicle": v, "station": station, "index": idx}))
                if station not in route:
                    customer_positions = [idx for idx, n in enumerate(route) if n in self.evaluator.customers]
                    for idx in customer_positions:
                        if idx + 1 < len(route) and route[idx + 1] == station:
                            continue
                        cand = self._copy_solution(solution)
                        cand[v].insert(idx + 1, station)
                        candidates.append((cand, {"vehicle": v, "station": station, "index": idx + 1}))
        return candidates

    def generate_candidates(self, solution, action: int):
        generators = [
            self.neighbor_intra_swap,
            self.neighbor_inter_relocate,
            self.neighbor_inter_swap,
            self.neighbor_two_opt,
            self.neighbor_station_toggle,
        ]
        return generators[action](solution)

    def choose_action_set(self, state: np.ndarray, iteration: int, stagnation: int) -> tuple[list[int], np.ndarray]:
        if self.policy_mode == "trpo":
            probs = self.policy.probs(state)
            ranked = list(np.argsort(-probs))
            shortlisted = ranked[: self.trpo_top_actions]
            if iteration % self.intensify_every == 0 or stagnation >= self.stagnation_limit:
                shortlisted = ranked[: max(self.trpo_top_actions, len(self.ACTION_NAMES) - 1)]
            exploratory = int(self.np_rng.choice(len(self.ACTION_NAMES), p=probs))
            if exploratory not in shortlisted:
                shortlisted.append(exploratory)
            return shortlisted, probs
        if self.policy_mode == "plain":
            return list(range(len(self.ACTION_NAMES))), np.ones(len(self.ACTION_NAMES), dtype=float) / len(self.ACTION_NAMES)
        if self.policy_mode == "random":
            action = int(self.rng.randrange(len(self.ACTION_NAMES)))
            return [action], np.ones(len(self.ACTION_NAMES), dtype=float) / len(self.ACTION_NAMES)
        if self.policy_mode == "cyclic":
            action = (iteration - 1) % len(self.ACTION_NAMES)
            return [action], np.ones(len(self.ACTION_NAMES), dtype=float) / len(self.ACTION_NAMES)
        raise ValueError(f"Unknown policy_mode: {self.policy_mode}")

    def _candidate_rank(self, cand_cost: float, cand_info: dict) -> tuple:
        return (
            cand_info["violations"],
            cand_info["battery_violations"],
            round(cand_info["tardiness_total"], 8),
            cand_cost,
            cand_info["used_vehicles"],
        )

    def run(self, iterations: int = 120) -> SearchResult:
        current = self.evaluator.initial_solution()
        current_cost, current_info = self.evaluator.evaluate(current)
        best = self._copy_solution(current)
        best_cost = current_cost
        best_trace = [(0, best_cost)]
        stagnation = 0

        tabu = deque(maxlen=self.tabu_tenure)
        experience_states = []
        experience_actions = []
        experience_rewards = []
        experience_probs = []
        iteration_stats = []
        update_stats = []

        for iteration in range(1, iterations + 1):
            state = self.state_features(current_cost, current_info, best_cost, iteration, iterations)
            action_set, probs = self.choose_action_set(state, iteration, stagnation)

            evaluated = []
            seen_candidates = set()
            for alt_action in action_set:
                for cand, meta in self.generate_candidates(current, alt_action):
                    sig = self.move_signature(alt_action, meta)
                    sol_sig = self._solution_signature(cand)
                    if sol_sig in seen_candidates:
                        continue
                    seen_candidates.add(sol_sig)
                    cand_cost, cand_info = self.evaluator.evaluate(cand)
                    aspiration = cand_cost < best_cost
                    if sig in tabu and not aspiration:
                        continue
                    evaluated.append((cand_cost, cand_info, cand, sig, alt_action))

            if not evaluated:
                evaluated = []
                for alt_action in range(len(self.ACTION_NAMES)):
                    for cand, meta in self.generate_candidates(current, alt_action):
                        sig = self.move_signature(alt_action, meta)
                        sol_sig = self._solution_signature(cand)
                        if sol_sig in seen_candidates:
                            continue
                        seen_candidates.add(sol_sig)
                        cand_cost, cand_info = self.evaluator.evaluate(cand)
                        aspiration = cand_cost < best_cost
                        if sig in tabu and not aspiration:
                            continue
                        evaluated.append((cand_cost, cand_info, cand, sig, alt_action))
                    if evaluated:
                        break
                if not evaluated:
                    break
                cand_cost, cand_info, candidate, sig, chosen_action = min(
                    evaluated,
                    key=lambda x: self._candidate_rank(x[0], x[1]),
                )
            else:
                cand_cost, cand_info, candidate, sig, chosen_action = min(
                    evaluated,
                    key=lambda x: self._candidate_rank(x[0], x[1]),
                )

            reward = current_cost - cand_cost
            if cand_cost < best_cost:
                reward += 0.5 * (best_cost - cand_cost)
            if cand_info["violations"] == 0:
                reward += 0.1

            current = candidate
            current_cost = cand_cost
            current_info = cand_info
            tabu.append(sig)

            if current_cost < best_cost:
                best = self._copy_solution(current)
                best_cost = current_cost
                best_trace.append((iteration, best_cost))
                stagnation = 0
            else:
                stagnation += 1

            if stagnation >= self.stagnation_limit and self.policy_mode == "trpo":
                current = self._copy_solution(best)
                current_cost, current_info = self.evaluator.evaluate(current)
                tabu.clear()
                stagnation = 0

            experience_states.append(state)
            experience_actions.append(chosen_action)
            experience_rewards.append(reward)
            experience_probs.append(probs)
            iteration_stats.append(
                {
                    "iteration": iteration,
                    "action": self.ACTION_NAMES[chosen_action],
                    "current_cost": current_cost,
                    "best_cost": best_cost,
                    "reward": reward,
                    "violations": current_info["violations"],
                }
            )

            if len(experience_states) >= self.update_every:
                states_np = np.vstack(experience_states)
                actions_np = np.asarray(experience_actions, dtype=int)
                rewards_np = np.asarray(experience_rewards, dtype=float)
                probs_np = np.vstack(experience_probs)
                if self.policy_mode == "trpo":
                    update_stats.append(self.policy.update(states_np, actions_np, rewards_np, probs_np))
                experience_states.clear()
                experience_actions.clear()
                experience_rewards.clear()
                experience_probs.clear()

        if experience_states:
            states_np = np.vstack(experience_states)
            actions_np = np.asarray(experience_actions, dtype=int)
            rewards_np = np.asarray(experience_rewards, dtype=float)
            probs_np = np.vstack(experience_probs)
            if self.policy_mode == "trpo":
                update_stats.append(self.policy.update(states_np, actions_np, rewards_np, probs_np))

        return SearchResult(best, best_cost, best_trace, iteration_stats, update_stats)


def pretty_print_result(result: SearchResult) -> None:
    print(f"Best cost: {result.best_cost:.3f}")
    print("Best routes:")
    for vehicle, route in result.best_solution.items():
        explicit = ["Depot"] + route + ["Depot"] if route else ["Depot"]
        print(f"  {vehicle}: {' -> '.join(explicit)}")
    print("\nBest-cost trace:")
    for iteration, cost in result.best_trace:
        print(f"  iter={iteration:3d}, cost={cost:.3f}")


def main():
    data = build_extended_data(
        selected_customers=["C1", "C2", "C5", "C6"],
        selected_vehicles=["V1", "V2"],
        battery_capacity_value=24.0,
    )
    search = TabuTRPOSearch(data, seed=11, tabu_tenure=10, neighborhood_samples=12, update_every=12)
    result = search.run(iterations=80)
    pretty_print_result(result)

    print("\nLast 10 iteration stats:")
    for row in result.iteration_stats[-10:]:
        print(
            f"  iter={row['iteration']:3d} action={row['action']:>15s} "
            f"current={row['current_cost']:.3f} best={row['best_cost']:.3f} "
            f"reward={row['reward']:.3f} violations={row['violations']}"
        )


if __name__ == "__main__":
    main()
