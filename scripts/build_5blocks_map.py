"""Build a single interactive Folium map covering all 5 target blocks.

The map overlays every block boundary (light grey) on a CARTO Positron
basemap, then highlights the 5 target blocks with their optimization
result. Each target carries a popup with:

- site area, build floor area budget, residual BFA;
- source population, new population, total;
- selected service units (per service type);
- run metadata (TPE best value, success rate, func evals, runtime).

Blocks are clickable: clicking toggles a per-block service overlay that
highlights where units were placed and the residual BFA that feeds the
new-population computation.
"""

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import folium
import geopandas as gpd
import numpy as np
import optuna
import pandas as pd

# Silence Optuna before importing optimizer pieces.
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
    if "site_area" not in blocks.columns:
        if "footprint_area" in blocks.columns:
            blocks = blocks.copy()
            blocks["site_area"] = blocks["footprint_area"]
        else:
            blocks = blocks.copy()
            blocks["site_area"] = blocks.geometry.area
    keep = blocks.index
    acc_mx = acc_mx.loc[acc_mx.index.intersection(keep), acc_mx.columns.intersection(keep)]
    return blocks, acc_mx


def available_service_types(blocks):
    return [st for st in WEIGHTS if f"{st}__capacity" in blocks.columns]


def run_one(target_block_id, blocks, acc_mx, *, max_runs, timeout, max_evals, seed):
    blocks_lu = {int(target_block_id): LandUse.RESIDENTIAL}
    service_types = available_service_types(blocks)
    service_weights = {st: WEIGHTS[st] for st in service_types}

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
    best_x, best_val, success_rate, func_evals = tpe.run(
        max_runs=max_runs, timeout=timeout, initial_runs_num=1
    )
    services_df = facade.solution_to_services_df(best_x)
    area_df = facade.get_solution_area_df(best_x)
    bfa_budget = float(facade._area_checker.build_floor_areas[int(target_block_id)])

    if int(target_block_id) in area_df.index:
        totals = area_df.loc[int(target_block_id)]
        reported_bfa = float(totals["build_floor_area"])
        population = float(totals["population"])
    else:
        reported_bfa = 0.0
        population = 0.0

    # best_x is the {x_i: count} dict from Optuna; convert to a numpy array
    # so the variable adapter can re-inject the solution.
    x_array = np.zeros(facade.num_params)
    for var_name, var_val in best_x.items():
        x_array[int(var_name[2:])] = int(var_val)
    new_population = float(facade.get_delta_population(x_array).get(int(target_block_id), 0))
    source_population = float(blocks.loc[int(target_block_id), "population"]) if "population" in blocks.columns else 0.0
    return {
        "block_id": int(target_block_id),
        "best_val": float(best_val) if best_val is not None else None,
        "success_rate": float(success_rate) if success_rate is not None else None,
        "func_evals": int(func_evals),
        "bfa_budget": bfa_budget,
        "sa_budget": float(facade._area_checker.site_areas[int(target_block_id)]),
        "reported_bfa": reported_bfa,
        "residual_bfa": bfa_budget - reported_bfa,
        "source_population": source_population,
        "new_population": new_population,
        "total_population": source_population + new_population,
        "services_df": services_df,
    }


def build_popup_html(target, target_geom_area_m2, acc_mx_row):
    services = target["services_df"]
    if services.empty:
        svc_rows_html = "<tr><td colspan='4'><i>no service units placed</i></td></tr>"
    else:
        rows = []
        for _, s in services.iterrows():
            rows.append(
                "<tr>"
                f"<td>{s['service_type']}</td>"
                f"<td>{int(s['count'])}</td>"
                f"<td>{int(s['capacity'] * s['count'])}</td>"
                f"<td>{float(s['build_floor_area'] * s['count']):.0f}</td>"
                "</tr>"
            )
        svc_rows_html = "".join(rows)
    delta = acc_mx_row.sort_values().head(6)
    neighbours = "<br>".join(
        f"{idx} → {float(v):.1f}" for idx, v in delta.items() if idx != target["block_id"]
    )
    return f"""
    <div style='font-family:Inter,system-ui;min-width:340px'>
      <h4 style='margin:0 0 6px 0'>Block {target['block_id']}</h4>
      <div style='color:#666;font-size:12px;margin-bottom:8px'>
        optimisation result · audit 2026-07-17
      </div>
      <table style='border-collapse:collapse;width:100%;font-size:12px'>
        <tr><td><b>site area</b></td><td>{target_geom_area_m2:,.0f} m²</td></tr>
        <tr><td><b>build floor area budget</b></td><td>{target['bfa_budget']:,.0f} m²</td></tr>
        <tr><td><b>site budget (SA coef)</b></td><td>{target['sa_budget']:,.0f} m²</td></tr>
        <tr><td><b>used BFA (services)</b></td><td>{target['reported_bfa']:,.0f} m²</td></tr>
        <tr><td><b>residual BFA (→ жильё)</b></td><td>{target['residual_bfa']:,.0f} m²</td></tr>
        <tr><td><b>source population</b></td><td>{int(target['source_population']):,}</td></tr>
        <tr><td><b>new population</b></td><td>{int(target['new_population']):,}</td></tr>
        <tr><td><b>total population</b></td><td>{int(target['total_population']):,}</td></tr>
        <tr><td><b>TPE best value</b></td><td>{target['best_val']:.4f}</td></tr>
        <tr><td><b>success rate</b></td><td>{target['success_rate']:.0%}</td></tr>
        <tr><td><b>func evals</b></td><td>{target['func_evals']}</td></tr>
      </table>
      <div style='font-weight:600;margin-top:10px'>Selected service units</div>
      <table style='border-collapse:collapse;width:100%;font-size:12px'>
        <tr style='background:#f4f4f4'>
          <th>service</th><th>count</th><th>capacity</th><th>BFA (m²)</th>
        </tr>
        {svc_rows_html}
      </table>
      <div style='font-weight:600;margin-top:10px'>Nearest neighbours (accessibility, min)</div>
      <div style='font-size:12px'>{neighbours or '—'}</div>
    </div>
    """


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-runs", type=int, default=4)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--max-evals", type=int, default=15)
    p.add_argument("--output", type=Path, default=Path("examples/optimization/services/5blocks_audit_2026_07_17/_audit_5blocks_map.html"))
    p.add_argument("--results", type=Path, default=Path("examples/optimization/services/5blocks_audit_2026_07_17/data/_audit_5blocks_map_results.json"))
    args = p.parse_args()

    blocks, acc_mx = load_data()
    picks = pd.read_csv("examples/optimization/services/5blocks_audit_2026_07_17/data/_selected_blocks.csv")

    results = []
    for _, row in picks.iterrows():
        bid = int(row["index"])
        print(f"running block_id={bid} for map", flush=True)
        res = run_one(bid, blocks, acc_mx, max_runs=args.max_runs, timeout=args.timeout, max_evals=args.max_evals, seed=42)
        results.append(res)

    target_ids = [r["block_id"] for r in results]
    target_set = set(target_ids)

    # Source geodataframe has the geometry; keep CRS for area + reproject later.
    full_geom = gpd.read_file("data/blocks_with_buildings_and_services.gpkg")

    # Compute area in m² for each target (using original CRS).
    target_geom_area = {bid: float(full_geom.loc[bid, "geometry"].area) for bid in target_ids}

    # Re-project to WGS84 for folium.
    full_wgs = full_geom.to_crs(epsg=4326)

    # Build the map.
    bounds = full_wgs.loc[target_ids].total_bounds  # minx, miny, maxx, maxy
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    fmap = folium.Map(
        location=center,
        zoom_start=12,
        tiles="CartoDB Positron",
        control_scale=True,
    )

    # Layer 1: all blocks as light grey outlines.
    full_layer = folium.FeatureGroup(name="All blocks (boundary)", show=True)
    folium.GeoJson(
        full_wgs[["geometry"]],
        name="All blocks",
        style_function=lambda _f: {
            "color": "#7a7a7a",
            "weight": 0.4,
            "fillOpacity": 0.0,
        },
        control=False,
    ).add_to(full_layer)
    full_layer.add_to(fmap)

    # Layer 2: target block halos (clickable).
    target_layer = folium.FeatureGroup(name="Target blocks (click for details)", show=True)
    for r in results:
        bid = r["block_id"]
        geom = full_wgs.loc[bid, "geometry"]
        # halo / outline
        popup_html = build_popup_html(
            r,
            target_geom_area[bid],
            acc_mx.loc[bid],
        )
        folium.GeoJson(
            {
                "type": "Feature",
                "properties": {},
                "geometry": geom.__geo_interface__,
            },
            style_function=lambda _f, _bid=bid: {
                "color": "#d62728",
                "weight": 2.0,
                "fillColor": "#ff7f7f",
                "fillOpacity": 0.30,
            },
            highlight_function=lambda _f: {
                "color": "#d62728",
                "weight": 3.5,
                "fillOpacity": 0.45,
            },
            tooltip=folium.Tooltip(f"Block {bid} — click for details"),
            popup=folium.Popup(popup_html, max_width=520),
        ).add_to(target_layer)
    target_layer.add_to(fmap)

    # Layer 3: a small marker per target with quick metrics for visual scan.
    for r in results:
        bid = r["block_id"]
        geom = full_wgs.loc[bid, "geometry"]
        c = geom.representative_point()
        html_lines = [
            f"<b>Block {bid}</b>",
            f"best_val = {r['best_val']:.3f}",
            f"new_pop = {int(r['new_population'])}",
            f"BFA used = {int(r['reported_bfa']):,} m²",
        ]
        html = "<br>".join(html_lines)
        popup_html = build_popup_html(r, target_geom_area[bid], acc_mx.loc[bid])
        folium.Marker(
            location=[c.y, c.x],
            icon=folium.DivIcon(
                icon_size=(140, 60),
                icon_anchor=(70, 30),
                html=(
                    "<div style='font-family:Inter,system-ui;font-size:11px;"
                    "background:#fff;border:1px solid #d62728;border-radius:4px;"
                    "padding:3px 5px;text-align:center;line-height:1.15'>"
                    f"{html}"
                    "</div>"
                ),
            ),
            tooltip=folium.Tooltip(f"Block {bid} — click for details"),
            popup=folium.Popup(popup_html, max_width=520),
        ).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    # Persist results JSON for inspection.
    serializable = []
    for r in results:
        rr = {k: v for k, v in r.items() if k != "services_df"}
        rr["services"] = r["services_df"].to_dict(orient="records")
        serializable.append(rr)
    args.results.write_text(json.dumps(serializable, indent=2, default=str))

    fmap.save(str(args.output))
    print(f"Wrote {args.output} and {args.results}")


if __name__ == "__main__":
    main()
