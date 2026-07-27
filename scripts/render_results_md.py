"""Render a markdown summary of the 5-block verification run."""

import json
import sys
from pathlib import Path


def fmt(v, fmt_str="{:>10}"):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return fmt_str.format(round(v, 3))
    return fmt_str.format(v)


def main():
    if len(sys.argv) < 2:
        print("usage: render_results_md.py examples/optimization/services/5blocks_audit_2026_07_17/data/_audit_5blocks_summary.json")
        sys.exit(1)
    path = Path(sys.argv[1])
    rows = json.loads(path.read_text())

    md = ["# 5 территорий — services-optimizer после аудита 2026-07-17",
          "",
          f"Файл результатов: `{path}`",
          "",
          "| block_id | bfa_budget | sa_budget | best_val | success_rate | func_evals | runtime_s | service_types | solution_units | services_rows |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

    for r in rows:
        md.append(
            "| {bid} | {bfa} | {sa} | {val} | {sr} | {fe} | {rt} | {st} | {su} | {sr_rows} |".format(
                bid=fmt(r.get("block_id"), "{:>8}"),
                bfa=fmt(r.get("bfa_budget"), "{:>12.1f}"),
                sa=fmt(r.get("sa_budget"), "{:>10.1f}"),
                val=fmt(r.get("best_val"), "{:>9.4f}"),
                sr=fmt(r.get("success_rate"), "{:>9.2f}"),
                fe=fmt(r.get("func_evals"), "{:>7}"),
                rt=fmt(r.get("runtime_s"), "{:>9.2f}"),
                st=fmt(r.get("num_service_types"), "{:>5}"),
                su=fmt(r.get("num_solution_units"), "{:>5}"),
                sr_rows=fmt(r.get("services_csv_rows"), "{:>5}"),
            )
        )
    md.append("")

    if any("error" in r and r["error"] for r in rows):
        md.append("## Ошибки")
        for r in rows:
            if r.get("error"):
                md.append(f"- block_id={r['block_id']}: {r['error']}")
        md.append("")

    print("\n".join(md))


if __name__ == "__main__":
    main()
