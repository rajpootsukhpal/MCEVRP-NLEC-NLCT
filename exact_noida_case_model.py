"""Exact MILP preparation for the verified Noida MCEVRP case.

The TS-TRPO search keeps routes as heuristic decisions. This module prepares the
corresponding global mixed-integer model for the same case data: customer
assignment, routing, compartment-product flow, affine payload-dependent battery
consumption, adjacent-segment PWL charging time, and soft upper time windows.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
import uuid
from pathlib import Path

import pulp

from noida_case_data import build_real_case_data


ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "case_inputs"
OUT = ROOT / "exact_model"


class LocalCBC(pulp.PULP_CBC_CMD):
    """Run bundled CBC from a local scratch directory with paths safe for spaces."""

    def actualSolve(self, model, **kwargs):
        root = Path(self.tmpDir)
        root.mkdir(parents=True, exist_ok=True)
        stem = uuid.uuid4().hex
        mps = root / f"{stem}.mps"
        solution = root / f"{stem}.sol"
        try:
            variables, names, constraints, _ = model.writeMPS(str(mps), rename=1)
            args = [str(Path(self.path).resolve()), mps.name]
            if model.sense == pulp.LpMaximize:
                args.append("-max")
            if self.timeLimit is not None:
                args += ["-sec", str(self.timeLimit)]
            args += ["-threads", "1", "-branch", "-printingOptions", "all", "-solution", solution.name]
            result = subprocess.run(
                args,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            if result.returncode or not solution.exists():
                raise RuntimeError("CBC failed: " + result.stdout.decode(errors="replace"))
            status, values, reduced, shadow, slacks, solution_status = self.readsol_MPS(
                str(solution), model, variables, names, constraints
            )
            model.assignVarsVals(values)
            model.assignVarsDj(reduced)
            model.assignConsPi(shadow)
            model.assignConsSlack(slacks, activity=True)
            model.assignStatus(status, solution_status)
            return status
        finally:
            for path in (mps, solution):
                try:
                    path.unlink(missing_ok=True)
                except PermissionError:
                    pass


def load_noida_case() -> dict:
    distance, duration = {}, {}
    with (INPUTS / "noida_osrm_road_matrices.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            arc = row["origin"], row["destination"]
            distance[arc] = float(row["distance_km"])
            duration[arc] = float(row["duration_h"])
    data = build_real_case_data(distance_duration=(distance, duration))
    return active_case_data(data)


def active_case_data(data: dict) -> dict:
    """Restrict the exact graph to depot, selected customers and stations."""

    keep = [data["depot"]] + list(data["customers"]) + list(data["stations"])
    active = dict(data)
    active["nodes"] = keep
    active["non_depot_nodes"] = [node for node in keep if node != data["depot"]]
    for name in ("distance", "travel_time", "route_cost", "alpha", "beta"):
        active[name] = {
            (i, j): value
            for (i, j), value in data[name].items()
            if i in keep and j in keep and i != j
        }
    active["demand"] = {node: data["demand"][node] for node in keep}
    active["service_time"] = {node: data["service_time"][node] for node in keep}
    active["time_window"] = {node: data["time_window"][node] for node in keep}
    return active


def _route_cost(route_cost: dict, vehicle: str, i: str, j: str) -> float:
    if (vehicle, i, j) in route_cost:
        return route_cost[vehicle, i, j]
    return route_cost[i, j]


def _compatible(compatibility: dict, vehicle: str, compartment: str, product: str) -> int:
    return int(compatibility.get((vehicle, compartment, product), compatibility.get((compartment, product), 0)))


def validate_case(data: dict) -> None:
    nodes = data["nodes"]
    required_arcs = [(i, j) for i in nodes for j in nodes if i != j]
    for name in ("distance", "travel_time", "route_cost", "alpha", "beta"):
        missing = [arc for arc in required_arcs if arc not in data[name]]
        if missing:
            raise ValueError(f"{name} is missing {len(missing)} active arcs")
    for customer in data["customers"]:
        if customer not in nodes:
            raise ValueError(f"Customer {customer} is not in active nodes")
    if set(data["customers"]) & set(data["stations"]):
        raise ValueError("Customers and stations must be disjoint")
    for vehicle in data["vehicles"]:
        if data["safety_soc"][vehicle] > data["battery_capacity"][vehicle]:
            raise ValueError(f"Invalid reserve for {vehicle}")
    points = sorted(data["rho"][b] for b in data["breakpoints"])
    if points[0] > 0 or points[-1] < max(data["battery_capacity"].values()):
        raise ValueError("Charging breakpoints must cover the full battery capacity")
    for station in data["stations"]:
        values = [data["gamma"][station, b] for b in sorted(data["breakpoints"], key=data["rho"].get)]
        if any(left >= right for left, right in zip(values, values[1:])):
            raise ValueError(f"Charging curve is not strictly increasing at {station}")


def build_exact_model(data: dict):
    validate_case(data)
    depot = data["depot"]
    customers = list(data["customers"])
    stations = list(data["stations"])
    vehicles = list(data["vehicles"])
    products = list(data["products"])
    compartments = list(data["compartments"])
    breakpoints = sorted(data["breakpoints"], key=lambda label: data["rho"][label])
    nodes = list(data["nodes"])
    non_depot_nodes = list(data["non_depot_nodes"])
    arcs = [(i, j) for i in nodes for j in nodes if i != j]
    segments = range(len(breakpoints) - 1)
    big_m = float(data["big_m"])

    model = pulp.LpProblem("Exact_Noida_MCEVRP_NLEC_NLCT", pulp.LpMinimize)

    x = pulp.LpVariable.dicts("x", (vehicles, nodes, nodes), 0, 1, cat="Binary")
    used = pulp.LpVariable.dicts("used", vehicles, 0, 1, cat="Binary")
    visit = pulp.LpVariable.dicts("visit", (vehicles, non_depot_nodes), 0, 1, cat="Binary")
    order = pulp.LpVariable.dicts("ord", (vehicles, non_depot_nodes), 0, len(non_depot_nodes), cat="Continuous")
    assign = pulp.LpVariable.dicts("g", (vehicles, compartments, products), 0, 1, cat="Binary")
    flow = pulp.LpVariable.dicts("f", (vehicles, nodes, nodes, compartments, products), 0, None, cat="Continuous")
    payload = pulp.LpVariable.dicts("load", (vehicles, nodes, nodes), 0, None, cat="Continuous")
    energy = pulp.LpVariable.dicts("energy", (vehicles, nodes, nodes), 0, None, cat="Continuous")
    arrival_soc = pulp.LpVariable.dicts("y", (vehicles, nodes), 0, None, cat="Continuous")
    depart_soc = pulp.LpVariable.dicts("Y", (vehicles, nodes), 0, None, cat="Continuous")
    recharge = pulp.LpVariable.dicts("r", (vehicles, stations), 0, None, cat="Continuous")
    charge_time = pulp.LpVariable.dicts("tau_ch", (vehicles, stations), 0, None, cat="Continuous")
    arrival_time = pulp.LpVariable.dicts("T", (vehicles, nodes), 0, None, cat="Continuous")
    tardiness = pulp.LpVariable.dicts("xi", customers, 0, None, cat="Continuous")
    lam_arr = pulp.LpVariable.dicts("lambda_arr", (vehicles, stations, breakpoints), 0, None, cat="Continuous")
    lam_dep = pulp.LpVariable.dicts("lambda_dep", (vehicles, stations, breakpoints), 0, None, cat="Continuous")
    seg_arr = pulp.LpVariable.dicts("seg_arr", (vehicles, stations, segments), 0, 1, cat="Binary")
    seg_dep = pulp.LpVariable.dicts("seg_dep", (vehicles, stations, segments), 0, 1, cat="Binary")

    model += (
        pulp.lpSum(data["fixed_cost"][v] * used[v] for v in vehicles)
        + pulp.lpSum(_route_cost(data["route_cost"], v, i, j) * x[v][i][j] for v in vehicles for i, j in arcs)
        + pulp.lpSum(data["charge_price"][s] * recharge[v][s] for v in vehicles for s in stations)
        + data["charge_time_weight"] * pulp.lpSum(charge_time[v][s] for v in vehicles for s in stations)
        + data["tardiness_weight"] * pulp.lpSum(tardiness[c] for c in customers)
    )

    for v in vehicles:
        for n in nodes:
            model += x[v][n][n] == 0, f"no_loop_{v}_{n}"
        model += pulp.lpSum(x[v][depot][j] for j in non_depot_nodes) == used[v], f"depart_depot_{v}"
        model += pulp.lpSum(x[v][i][depot] for i in non_depot_nodes) == used[v], f"return_depot_{v}"
        model += arrival_soc[v][depot] == data["battery_capacity"][v], f"depot_arr_soc_{v}"
        model += depart_soc[v][depot] == data["battery_capacity"][v], f"depot_dep_soc_{v}"
        model += arrival_time[v][depot] == 0, f"depot_time_{v}"

        for n in non_depot_nodes:
            model += pulp.lpSum(x[v][i][n] for i in nodes if i != n) == visit[v][n], f"in_{v}_{n}"
            model += pulp.lpSum(x[v][n][j] for j in nodes if j != n) == visit[v][n], f"out_{v}_{n}"
            model += order[v][n] >= visit[v][n], f"order_lb_{v}_{n}"
            model += order[v][n] <= len(non_depot_nodes) * visit[v][n], f"order_ub_{v}_{n}"
            model += arrival_time[v][n] <= big_m * visit[v][n], f"unused_time_{v}_{n}"

        for i in non_depot_nodes:
            for j in non_depot_nodes:
                if i != j:
                    model += (
                        order[v][i] - order[v][j] + len(non_depot_nodes) * x[v][i][j]
                        <= len(non_depot_nodes) - 1
                    ), f"mtz_{v}_{i}_{j}"

    for left, right in zip(vehicles, vehicles[1:]):
        model += used[left] >= used[right], f"vehicle_symmetry_{left}_{right}"

    for c in customers:
        model += pulp.lpSum(visit[v][c] for v in vehicles) == 1, f"serve_once_{c}"

    for v in vehicles:
        for l in compartments:
            model += pulp.lpSum(assign[v][l][p] for p in products) <= 1, f"one_product_{v}_{l}"
            for p in products:
                model += assign[v][l][p] <= _compatible(data["compatibility"], v, l, p), f"compat_{v}_{l}_{p}"

        for i, j in arcs:
            model += payload[v][i][j] == pulp.lpSum(flow[v][i][j][l][p] for l in compartments for p in products)
            model += payload[v][i][j] <= data["vehicle_capacity"][v] * x[v][i][j]
            model += energy[v][i][j] == data["alpha"][i, j] * x[v][i][j] + data["beta"][i, j] * payload[v][i][j]
            for l in compartments:
                for p in products:
                    model += flow[v][i][j][l][p] <= data["compartment_capacity"][v, l] * assign[v][l][p]
                    model += flow[v][i][j][l][p] <= data["compartment_capacity"][v, l] * x[v][i][j]

        for p in products:
            model += (
                pulp.lpSum(flow[v][depot][j][l][p] for j in non_depot_nodes for l in compartments)
                == pulp.lpSum(data["demand"][c][p] * visit[v][c] for c in customers)
            ), f"depot_load_{v}_{p}"
            model += pulp.lpSum(flow[v][i][depot][l][p] for i in non_depot_nodes for l in compartments) == 0
            for s in stations:
                model += (
                    pulp.lpSum(flow[v][i][s][l][p] for i in nodes if i != s for l in compartments)
                    == pulp.lpSum(flow[v][s][j][l][p] for j in nodes if j != s for l in compartments)
                ), f"station_flow_{v}_{s}_{p}"
            for c in customers:
                model += (
                    pulp.lpSum(flow[v][i][c][l][p] for i in nodes if i != c for l in compartments)
                    - pulp.lpSum(flow[v][c][j][l][p] for j in nodes if j != c for l in compartments)
                    == data["demand"][c][p] * visit[v][c]
                ), f"customer_flow_{v}_{c}_{p}"

        for i, j in arcs:
            if j == depot:
                model += (
                    depart_soc[v][i] - energy[v][i][depot]
                    >= data["safety_soc"][v] - big_m * (1 - x[v][i][depot])
                ), f"return_reserve_{v}_{i}"
            else:
                model += (
                    arrival_soc[v][j] <= depart_soc[v][i] - energy[v][i][j] + big_m * (1 - x[v][i][j])
                ), f"soc_ub_{v}_{i}_{j}"
                model += (
                    arrival_soc[v][j] >= depart_soc[v][i] - energy[v][i][j] - big_m * (1 - x[v][i][j])
                ), f"soc_lb_{v}_{i}_{j}"
                model += (
                    arrival_time[v][j]
                    >= arrival_time[v][i]
                    + data["service_time"][i]
                    + data["travel_time"][i, j]
                    + (charge_time[v][i] if i in stations else 0)
                    - big_m * (1 - x[v][i][j])
                ), f"time_{v}_{i}_{j}"

        for c in customers:
            model += data["safety_soc"][v] * visit[v][c] <= arrival_soc[v][c]
            model += arrival_soc[v][c] <= data["battery_capacity"][v] * visit[v][c]
            model += depart_soc[v][c] == arrival_soc[v][c]
            a_c, b_c = data["time_window"][c]
            model += arrival_time[v][c] >= a_c - big_m * (1 - visit[v][c])
            model += arrival_time[v][c] <= b_c + tardiness[c] + big_m * (1 - visit[v][c])

        for s in stations:
            model += data["safety_soc"][v] * visit[v][s] <= arrival_soc[v][s]
            model += arrival_soc[v][s] <= data["battery_capacity"][v] * visit[v][s]
            model += data["safety_soc"][v] * visit[v][s] <= depart_soc[v][s]
            model += depart_soc[v][s] <= data["battery_capacity"][v] * visit[v][s]
            model += depart_soc[v][s] == arrival_soc[v][s] + recharge[v][s]
            model += recharge[v][s] <= data["battery_capacity"][v] * visit[v][s]
            model += recharge[v][s] <= data["battery_capacity"][v] - arrival_soc[v][s] + big_m * (1 - visit[v][s])
            model += pulp.lpSum(lam_arr[v][s][b] for b in breakpoints) == visit[v][s]
            model += pulp.lpSum(lam_dep[v][s][b] for b in breakpoints) == visit[v][s]
            model += pulp.lpSum(seg_arr[v][s][h] for h in segments) == visit[v][s]
            model += pulp.lpSum(seg_dep[v][s][h] for h in segments) == visit[v][s]
            for b_idx, b in enumerate(breakpoints):
                adjacent = [h for h in segments if h == b_idx or h + 1 == b_idx]
                model += lam_arr[v][s][b] <= pulp.lpSum(seg_arr[v][s][h] for h in adjacent)
                model += lam_dep[v][s][b] <= pulp.lpSum(seg_dep[v][s][h] for h in adjacent)
            model += arrival_soc[v][s] == pulp.lpSum(data["rho"][b] * lam_arr[v][s][b] for b in breakpoints)
            model += depart_soc[v][s] == pulp.lpSum(data["rho"][b] * lam_dep[v][s][b] for b in breakpoints)
            model += charge_time[v][s] == (
                pulp.lpSum(data["gamma"][s, b] * lam_dep[v][s][b] for b in breakpoints)
                - pulp.lpSum(data["gamma"][s, b] * lam_arr[v][s][b] for b in breakpoints)
            )

    variables = {
        "x": x,
        "used": used,
        "visit": visit,
        "order": order,
        "assign": assign,
        "flow": flow,
        "payload": payload,
        "energy": energy,
        "arrival_soc": arrival_soc,
        "depart_soc": depart_soc,
        "recharge": recharge,
        "charge_time": charge_time,
        "arrival_time": arrival_time,
        "tardiness": tardiness,
        "arcs": arcs,
    }
    return model, variables


def model_summary(data: dict, model: pulp.LpProblem) -> dict:
    variables = model.variables()
    return {
        "case": data["case_name"],
        "active_nodes": len(data["nodes"]),
        "customers": len(data["customers"]),
        "stations": len(data["stations"]),
        "vehicles": len(data["vehicles"]),
        "products": data["products"],
        "compartments": data["compartments"],
        "arcs": len(data["nodes"]) * (len(data["nodes"]) - 1),
        "variables": len(variables),
        "binary_variables": sum(variable.cat in ("Binary", "Integer") and variable.lowBound == 0 and variable.upBound == 1 for variable in variables),
        "constraints": len(model.constraints),
        "objective": [
            "fixed vehicle cost",
            "distance/routing cost",
            "charged energy cost",
            "piecewise-linear charging-time cost",
            "soft time-window tardiness penalty",
        ],
        "energy_equation": "energy[v,i,j] = alpha[i,j] * x[v,i,j] + beta[i,j] * payload[v,i,j]",
        "charging_model": "adjacent-segment SOS2-style convex-combination interpolation on absolute SOC breakpoints",
        "active_node_rule": "depot + selected case customers + case charging stations only",
    }


def solve(model: pulp.LpProblem, time_limit: int, msg: bool) -> str:
    scratch = OUT / ".solver_tmp"
    solver = LocalCBC(msg=msg, timeLimit=time_limit, threads=1)
    solver.tmpDir = str(scratch)
    status = model.solve(solver)
    return pulp.LpStatus[status]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-lp", action="store_true", help="export the exact Noida model as LP")
    parser.add_argument("--write-mps", action="store_true", help="export the exact Noida model as MPS")
    parser.add_argument("--solve", action="store_true", help="run CBC on the full exact case model")
    parser.add_argument("--time-limit", type=int, default=600, help="CBC time limit in seconds when --solve is used")
    parser.add_argument("--msg", action="store_true", help="show CBC output")
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    start = time.perf_counter()
    data = load_noida_case()
    model, _variables = build_exact_model(data)
    summary = model_summary(data, model)
    summary["build_seconds"] = time.perf_counter() - start

    if args.write_lp:
        lp_path = OUT / "noida_exact_model.lp"
        model.writeLP(str(lp_path))
        summary["lp_file"] = str(lp_path.relative_to(ROOT))
    if args.write_mps:
        mps_path = OUT / "noida_exact_model.mps"
        model.writeMPS(str(mps_path), rename=1)
        summary["mps_file"] = str(mps_path.relative_to(ROOT))
    if args.solve:
        solve_start = time.perf_counter()
        summary["solver_status"] = solve(model, args.time_limit, args.msg)
        summary["solve_seconds"] = time.perf_counter() - solve_start
        summary["objective_value"] = None if model.objective is None else pulp.value(model.objective)

    path = OUT / "exact_model_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
