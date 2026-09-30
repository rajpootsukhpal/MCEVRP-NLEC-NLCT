from __future__ import annotations

import copy
import math
import random
import time
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
    evaluation_count: int = 0
    policy_update_time_s: float = 0.0
    policy_decision_time_s: float = 0.0
    best_info: dict | None = None
    @property
    def feasible(self) -> bool:
        return self.best_info is not None and self.best_info["violations"] == 0



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

    def state_dict(self) -> dict:
        return {"state_dim": self.state_dim, "action_dim": self.action_dim,
                "W": self.W.tolist(), "b": self.b.tolist(), "value_w": self.value_w.tolist()}

    def load_state_dict(self, state: dict) -> None:
        if state["state_dim"] != self.state_dim or state["action_dim"] != self.action_dim:
            raise ValueError("Policy dimensions do not match")
        arrays = [np.asarray(state[k], dtype=float) for k in ("W", "b", "value_w")]
        for arr, old in zip(arrays, (self.W, self.b, self.value_w)):
            if arr.shape != old.shape or not np.all(np.isfinite(arr)):
                raise ValueError("Invalid policy parameters")
        self.W, self.b, self.value_w = [arr.copy() for arr in arrays]

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


from generalized_evaluator import MCEVRPRouteEvaluator


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
        feasible_only: bool = False,
        policy_state: dict | None = None,
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
        if policy_state is not None:
            self.policy.load_state_dict(policy_state)
        self.policy_mode = policy_mode
        self.trpo_top_actions = max(1, min(trpo_top_actions, len(self.ACTION_NAMES)))
        self.intensify_every = max(1, intensify_every)
        self.stagnation_limit = max(2, stagnation_limit)
        self.feasible_only = feasible_only
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
                    if not route:
                        continue
                    for idx in range(len(route) + 1):
                        cand = self._copy_solution(solution)
                        cand[v].insert(idx, station)
                        candidates.append((cand, {"vehicle": v, "station": station, "index": idx}))
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
        if self.policy_mode in {"trpo", "frozen_trpo"}:
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
            cand_cost,
            cand_info["used_vehicles"],
        )

    def run(
        self,
        iterations: int = 120,
        max_evaluations: int | None = None,
        initial_solution: dict[Vehicle, list[Node]] | None = None,
        deadline: float | None = None,
    ) -> SearchResult:
        current = self._copy_solution(initial_solution) if initial_solution is not None else self.evaluator.initial_solution()
        for vehicle in self.vehicles:
            current.setdefault(vehicle, [])
        current_cost, current_info = self.evaluator.evaluate(current)
        evaluation_count = 1
        best = self._copy_solution(current)
        best_cost = current_cost
        best_info = current_info
        best_trace = [(0, best_cost)]
        stagnation = 0

        tabu = deque(maxlen=self.tabu_tenure)
        experience_states = []
        experience_actions = []
        experience_rewards = []
        experience_probs = []
        iteration_stats = []
        update_stats = []
        policy_update_time_s = 0.0
        policy_decision_time_s = 0.0

        def update_policy(states_np, actions_np, rewards_np, probs_np):
            nonlocal policy_update_time_s
            if self.policy_mode != "trpo":
                return
            update_started = time.perf_counter()
            stats = self.policy.update(states_np, actions_np, rewards_np, probs_np)
            stats["update_time_s"] = time.perf_counter() - update_started
            policy_update_time_s += stats["update_time_s"]
            update_stats.append(stats)

        for iteration in range(1, iterations + 1):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            state = self.state_features(current_cost, current_info, best_cost, iteration, iterations)
            decision_started = time.perf_counter()
            action_set, probs = self.choose_action_set(state, iteration, stagnation)
            policy_decision_time_s += time.perf_counter() - decision_started

            evaluated = []
            seen_candidates = set()
            budget_exhausted = False
            for alt_action in action_set:
                for cand, meta in self.generate_candidates(current, alt_action):
                    if deadline is not None and time.perf_counter() >= deadline:
                        budget_exhausted = True
                        break
                    if max_evaluations is not None and evaluation_count >= max_evaluations:
                        budget_exhausted = True
                        break
                    sig = self.move_signature(alt_action, meta)
                    sol_sig = self._solution_signature(cand)
                    if sol_sig in seen_candidates:
                        continue
                    seen_candidates.add(sol_sig)
                    cand_cost, cand_info = self.evaluator.evaluate(cand)
                    evaluation_count += 1
                    if self.feasible_only and cand_info["violations"] > 0:
                        continue
                    aspiration = (cand_info["violations"] == 0 and best_info["violations"] > 0) or (cand_info["violations"] == best_info["violations"] and cand_cost < best_cost)
                    if sig in tabu and not aspiration:
                        continue
                    evaluated.append((cand_cost, cand_info, cand, sig, alt_action))
                if budget_exhausted:
                    break

            if not evaluated:
                if budget_exhausted or (max_evaluations is not None and evaluation_count >= max_evaluations):
                    break
                evaluated = []
                for alt_action in range(len(self.ACTION_NAMES)):
                    for cand, meta in self.generate_candidates(current, alt_action):
                        if deadline is not None and time.perf_counter() >= deadline:
                            budget_exhausted = True
                            break
                        if max_evaluations is not None and evaluation_count >= max_evaluations:
                            budget_exhausted = True
                            break
                        sig = self.move_signature(alt_action, meta)
                        sol_sig = self._solution_signature(cand)
                        if sol_sig in seen_candidates:
                            continue
                        seen_candidates.add(sol_sig)
                        cand_cost, cand_info = self.evaluator.evaluate(cand)
                        evaluation_count += 1
                        if self.feasible_only and cand_info["violations"] > 0:
                            continue
                        aspiration = (cand_info["violations"] == 0 and best_info["violations"] > 0) or (cand_info["violations"] == best_info["violations"] and cand_cost < best_cost)
                        if sig in tabu and not aspiration:
                            continue
                        evaluated.append((cand_cost, cand_info, cand, sig, alt_action))
                    if budget_exhausted:
                        break
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

            if (current_info["violations"] == 0 and best_info["violations"] > 0) or (
                (current_info["violations"] == 0) == (best_info["violations"] == 0)
                and self._candidate_rank(current_cost, current_info) < self._candidate_rank(best_cost, best_info)
            ):
                best = self._copy_solution(current)
                best_cost = current_cost
                best_info = current_info
                best_trace.append((iteration, best_cost))
                stagnation = 0
            else:
                stagnation += 1

            if stagnation >= self.stagnation_limit and self.policy_mode in {"trpo", "frozen_trpo"}:
                current = self._copy_solution(best)
                current_cost, current_info = best_cost, best_info
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
                update_policy(states_np, actions_np, rewards_np, probs_np)
                experience_states.clear()
                experience_actions.clear()
                experience_rewards.clear()
                experience_probs.clear()
            if max_evaluations is not None and evaluation_count >= max_evaluations:
                break

        if experience_states and (deadline is None or time.perf_counter() < deadline):
            states_np = np.vstack(experience_states)
            actions_np = np.asarray(experience_actions, dtype=int)
            rewards_np = np.asarray(experience_rewards, dtype=float)
            probs_np = np.vstack(experience_probs)
            update_policy(states_np, actions_np, rewards_np, probs_np)

        return SearchResult(
            best,
            best_cost,
            best_trace,
            iteration_stats,
            update_stats,
            evaluation_count,
            policy_update_time_s,
            policy_decision_time_s,
            best_info,
        )


def pretty_print_result(result: SearchResult) -> None:
    print(f"Feasible: {result.feasible}")
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
