"""Run services-optimizer on 5 residential blocks and report a comparison table.

This is the post-audit verification harness. Each block runs through the same
pipeline: pick the block, build a residential Facade, run TPE for a fixed
budget, dump the solution, area summary, and trial stats. The output CSV is
intended to be reviewed against the new contract (current-x demand,
catchment-only bounds, all-residential demand, parking inlined, schema
valid).
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import geopandas as gpd
import numpy as np
import optuna
import pandas as pd

# Silence Optuna before the optimizer instantiates.
optuna.logging.set_verbosity(optuna.logging.WARNING)

from blocksnet.enums import LandUse
from blocksnet.optimization.services import (
    AreaSolution,
    Facade,
    GradientChooser,
    RandomOrder,
    TPEOptimizer,
    WeightedConstraints,
    WeightedObjective,
)


WEIGHTS = {
    # Basic services only — keeps the variable space small enough for a
    # verification run. The full 18-service weight set is in
    # services-optimization.md (this script is for sanity checks, not a
    # quality benchmark).
    "kindergarten": 0.114,
    "school": 0.114,
    "pharmacy": 0.114,
    "polyclinics": 0.171,
    "convenience": 0.114,
    "playground": 0.114,
    "post_office": 0.057,
    "hairdresser": 0.057,
    "fuel": 0.071,
}


def load_data():
    blocks = gpd.read_file("data/blocks_with_buildings_and_services.gpkg")
    acc_mx = pd.read_pickle("data/acc_mx.pickle")
    # blocks must expose ``site_area`` for AreaChecker; footprint_area is the
    # closest analogue. geometry is dropped only after computing a fallback
    # site_area, just in case footprint_area is missing.
    if "site_area" not in blocks.columns:
        if "footprint_area" in blocks.columns:
            blocks = blocks.copy()
            blocks["site_area"] = blocks["footprint_area"]
        else:
            blocks = blocks.copy()
            blocks["site_area"] = blocks.geometry.area
    blocks = blocks.drop(columns=["geometry"]).copy()
    # Align acc_mx to the block index. acc_mx occasionally references blocks
    # that did not survive to blocks.gpkg (e.g. merged or filtered). Trimming
    # is safe because the optimizer only queries distances between target and
    # the blocks that actually carry data.
    keep = blocks.index
    acc_mx = acc_mx.loc[acc_mx.index.intersection(keep), acc_mx.columns.intersection(keep)]
    return blocks, acc_mx


def available_service_types(blocks, target_lu):
    capacity_cols = [c for c in blocks.columns if c.endswith("__capacity")]
    st_present = []
    for col in capacity_cols:
        st = col[: -len("__capacity")]
        if st in WEIGHTS and f"{st}__capacity" in blocks.columns:
            st_present.append(st)
    return st_present


def run_one(target_block_id, blocks, acc_mx, *, max_runs, timeout, max_evals, seed):
    target_block_id = int(target_block_id)
    blocks_lu = {target_block_id: LandUse.RESIDENTIAL}
    target_idx = blocks.index.get_loc(target_block_id)
    target_lu = LandUse.RESIDENTIAL
    service_types = available_service_types(blocks, target_lu)
    service_weights = {st: WEIGHTS[st] for st in service_types}
    if not service_weights:
        return {"block_id": target_block_id, "error": "no service types available"}

    var_adapter = AreaSolution(blocks_lu)
    facade = Facade(
        var_adapter=var_adapter,
        accessibility_matrix=acc_mx,
        blocks_df=blocks,
        blocks_lu=blocks_lu,
    )
    for st, w in service_weights.items():
        col = f"{st}__capacity"
        services_df = blocks[[col]].rename(columns={col: "capacity"})
        facade.add_service_type(st, w, services_df)

    objective = WeightedObjective(
        num_params=facade.num_params,
        facade=facade,
        weights=service_weights,
        max_evals=max_evals,
    )
    constraints = WeightedConstraints(
        num_params=facade.num_params,
        facade=facade,
        priority=service_weights,
    )
    tpe = TPEOptimizer(
        objective=objective,
        constraints=constraints,
        vars_order=RandomOrder(),
        vars_chooser=GradientChooser(facade, num_params=facade.num_params, num_top=5),
        n_ei_candidates=12,
    )

    t0 = time.time()
    best_x, best_val, success_rate, func_evals = tpe.run(
        max_runs=max_runs, timeout=timeout, initial_runs_num=1
    )
    runtime_s = time.time() - t0

    services_df = facade.solution_to_services_df(best_x)
    area_df = facade.get_solution_area_df(best_x)
    if target_block_id in area_df.index:
        totals = area_df.loc[target_block_id]
        used_bfa = float(totals["build_floor_area"])
        population = float(totals["population"])
    else:
        used_bfa = 0.0
        population = 0.0

    bfa_budget = float(facade._area_checker.build_floor_areas[target_block_id])

    return {
        "block_id": target_block_id,
        "bfa_budget": bfa_budget,
        "sa_budget": float(facade._area_checker.site_areas[target_block_id]),
        "best_val": float(best_val) if best_val is not None else None,
        "success_rate": float(success_rate) if success_rate is not None else None,
        "func_evals": int(func_evals),
        "runtime_s": float(runtime_s),
        "num_service_types": len(service_weights),
        "num_solution_units": int(services_df["count"].sum()) if not services_df.empty else 0,
        "service_types_used": int(services_df["service_type"].nunique()) if not services_df.empty else 0,
        "services_csv_rows": int(len(services_df)),
        "reported_bfa": used_bfa,
        "residual_bfa": bfa_budget - used_bfa,
        "population": population,
        "best_x_keys": int(len(best_x)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-runs", type=int, default=120)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--max-evals", type=int, default=200)
    p.add_argument("--output", type=Path, default=Path("examples/optimization/services/5blocks_audit_2026_07_17/data/_audit_5blocks_results.csv"))
    p.add_argument("--summary", type=Path, default=Path("examples/optimization/services/5blocks_audit_2026_07_17/data/_audit_5blocks_summary.json"))
    args = p.parse_args()

    blocks, acc_mx = load_data()
    picks = pd.read_csv("examples/optimization/services/5blocks_audit_2026_07_17/data/_selected_blocks.csv")
    results = []
    print(
        f"Running services-optimizer on {len(picks)} blocks | "
        f"max_runs={args.max_runs} timeout={args.timeout}s max_evals={args.max_evals}",
        flush=True,
    )
    for _, row in picks.iterrows():
        bid = int(row["index"])
        print(f"--- block_id={bid} ---", flush=True)
        try:
            res = run_one(bid, blocks, acc_mx, max_runs=args.max_runs, timeout=args.timeout, max_evals=args.max_evals, seed=None)
        except Exception as e:
            res = {"block_id": bid, "error": f"{type(e).__name__}: {e}"}
        print(json.dumps(res, default=str, indent=2), flush=True)
        results.append(res)

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    args.summary.write_text(json.dumps(results, default=str, indent=2))
    print(f"\nWrote {args.output} and {args.summary}")

    if "error" not in df.columns:
        df["error"] = None
    if (df["error"].fillna("") != "").any():
        print("Some runs failed; see summary.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
