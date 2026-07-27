"""Regression tests for the services-optimizer audit (2026-07-17).

Each test pins one defect from the audit report to a deterministic behaviour.
The tests exercise the public surface of the audited modules; some
intentionally bypass full ``Facade`` initialisation (which loads the real
service-types config) to keep the scope local to the audited code.
"""

import numpy as np
import pandas as pd
import pytest

from blocksnet.config.service_types.schemas import UnitsSchema
from blocksnet.enums import LandUse
from blocksnet.optimization.services.acl.checkers.area_checker import AreaChecker
from blocksnet.optimization.services.acl.checkers.capacity_checker import CapacityChecker
from blocksnet.optimization.services.acl.facade import Facade
from blocksnet.optimization.services.acl.provision_adapter import ProvisionAdapter
from blocksnet.optimization.services.common.variable import Variable


# --------------------------------------------------------------------------- helpers


class _StubConverter:
    """Minimal VariableAdapter stub for Facade unit tests."""

    def __init__(self, variables, used_bfa_per_block=None):
        self._variables = list(variables)
        # Optionally accept a precomputed ``used_bfa_per_block`` so the stub
        # mimics a solution with consumed BFA. Otherwise no BFA is consumed.
        self._used_bfa = dict(used_bfa_per_block or {})

    @property
    def X(self):
        return self._variables

    def __call__(self, _x):
        return self._variables

    def __len__(self):
        return len(self._variables)

    def _variables_to_df(self, variables):
        rows = [
            {
                "block_id": v.block_id,
                "service_type": v.service_type,
                "site_area": v.site_area,
                "build_floor_area": v.build_floor_area,
                "capacity": v.capacity,
                "count": v.count,
                "total_build_floor_area": v.build_floor_area * v.count,
                "total_site_area": v.site_area * v.count,
                "total_capacity": v.capacity * v.count,
            }
            for v in variables
        ]
        # If the stub was seeded with consumed BFA (independent of variables),
        # surface it as synthetic rows so agg_total_build_area matches reality.
        for block_id, bfa in self._used_bfa.items():
            if bfa <= 0:
                continue
            rows.append(
                {
                    "block_id": int(block_id),
                    "service_type": "_stub_consumed_bfa",
                    "site_area": 0,
                    "build_floor_area": float(bfa),
                    "capacity": 0,
                    "count": 1,
                    "total_build_floor_area": float(bfa),
                    "total_site_area": 0,
                    "total_capacity": 0,
                }
            )
        df = pd.DataFrame(rows)
        if df.empty:
            # Provide a typed empty DataFrame so downstream groupby does not
            # blow up on a missing ``block_id`` column.
            df = pd.DataFrame(
                {
                    "block_id": pd.Series(dtype="int64"),
                    "service_type": pd.Series(dtype="object"),
                    "site_area": pd.Series(dtype="float64"),
                    "build_floor_area": pd.Series(dtype="float64"),
                    "capacity": pd.Series(dtype="int64"),
                    "count": pd.Series(dtype="int64"),
                    "total_build_floor_area": pd.Series(dtype="float64"),
                    "total_site_area": pd.Series(dtype="float64"),
                    "total_capacity": pd.Series(dtype="int64"),
                }
            )
        return df


def _build_facade(blocks_bfa, capacity_demand):
    """Build a Facade with controlled BFA and demand state."""
    blocks_df = pd.DataFrame(
        {
            "site_area": [v * 2.0 for v in blocks_bfa.values()],  # doubled so area/ub scales
            "population": [0] * len(blocks_bfa),
            "capacity": [0] * len(blocks_bfa),
            "demand": [0] * len(blocks_bfa),
        },
        index=list(blocks_bfa.keys()),
    )
    accessibility = pd.DataFrame(
        0, index=blocks_df.index, columns=blocks_df.index
    )
    blocks_lu = {bid: LandUse.RESIDENTIAL for bid in blocks_df.index}
    area = AreaChecker(blocks_lu, blocks_df)
    # Override BFA so that the test fully controls residual.
    area.build_floor_areas = dict(blocks_bfa)
    area.site_areas = {bid: 1e6 for bid in blocks_df.index}
    cap = CapacityChecker(list(blocks_df.index), accessibility)
    cap._demands = {bid: dict(capacity_demand.get(bid, {})) for bid in blocks_df.index}

    facade = Facade.__new__(Facade)
    facade._blocks_lu = blocks_lu
    facade._area_checker = area
    facade._capacity_checker = cap
    facade._chosen_service_types = set()
    for bid in blocks_df.index:
        for st in capacity_demand.get(bid, {}).keys():
            facade._chosen_service_types.add(st)
    facade.num_params = 0
    facade._last_provisions = {}
    return facade, blocks_df, accessibility


# --------------------------------------------------------------------------- P0-1


def test_check_constraints_uses_demand_of_current_x_not_cached():
    """P0-1: feasibility must be derived from the current x, not a previous
    trial's cached demand."""
    facade, _, _ = _build_facade({1: 10_000.0, 2: 10_000.0}, {1: {"school": 5}, 2: {"school": 5}})
    facade._converter = _StubConverter([])
    # Inject a stale cache to make sure check_constraints does not touch it.
    class _Boom:
        def __getitem__(self, _):
            raise AssertionError("check_constraints must not read _last_blocks_services_demand")

    facade._last_blocks_services_demand = _Boom()
    # With the stub in place, residual BFA for both blocks is 10_000, so
    # population is positive ⇒ demand for school is positive.
    # Static capacity_demand is 5; new demand 1.9 ⇒ capacity 5 already meets
    # the new demand; this x is feasible.
    assert facade.check_constraints(np.array([])) is True


def test_check_constraints_recomputes_demand_each_call(monkeypatch):
    """P0-1: a single x must produce identical feasibility regardless of any
    earlier state on the facade."""
    facade, _, _ = _build_facade({1: 10_000.0}, {1: {"school": 100}})
    facade._converter = _StubConverter([])
    calls = {"n": 0}

    def _spy(*_a, **_kw):
        calls["n"] += 1
        return {1: {"school": 0}}

    monkeypatch.setattr(facade, "get_delta_demand", _spy)
    facade.check_constraints(np.array([]))
    facade.check_constraints(np.array([]))
    assert calls["n"] == 2


# --------------------------------------------------------------------------- P1-1


def test_get_upper_bound_var_does_not_use_cached_delta_demand():
    """P1-1: upper bounds must come from the static catchment demand only."""
    facade, _, _ = _build_facade({1: 10_000.0}, {1: {"school": 5}})
    var = Variable(
        block_id=1, service_type="school", site_area=0, build_floor_area=100, capacity=10, count=0
    )
    facade._converter = _StubConverter([var])
    facade.num_params = 1

    class _Boom:
        def __getitem__(self, _):
            raise AssertionError("get_upper_bound_var must not read _last_blocks_services_demand")

    facade._last_blocks_services_demand = _Boom()
    ub = facade.get_upper_bound_var(0)
    # Static catchment demand 5 / capacity 10 ⇒ ub = 0 (5 already met by one unit).
    assert ub == 0


# --------------------------------------------------------------------------- P0-2


def test_provision_includes_demand_from_blocks_without_local_unit(monkeypatch):
    """P0-2: every residential block with new residents must contribute to
    the demand total, even if it has no local unit of the current service
    type."""
    facade, blocks_df, accessibility = _build_facade(
        {1: 5_000.0, 2: 5_000.0}, {1: {"school": 0}, 2: {"school": 0}}
    )
    adapter = ProvisionAdapter(facade._blocks_lu, accessibility, blocks_df)
    # Pre-seed the start/last provision dataframes.
    empty = pd.DataFrame(
        {
            "demand": [0, 0],
            "demand_within": [0, 0],
            "demand_without": [0, 0],
            "capacity": [0, 0],
            "capacity_within": [0, 0],
            "capacity_without": [0, 0],
            "demand_left": [0, 0],
            "capacity_left": [0, 0],
        },
        index=blocks_df.index,
    )
    adapter.start_provisions_dfs["school"] = empty.copy()
    adapter.last_provisions_dfs["school"] = empty.copy()

    seen = {}

    def _fake_competitive(prov_df, *_a, **_kw):
        seen["demand_total"] = prov_df["demand"].sum()
        seen["blocks_with_demand"] = int((prov_df["demand"] > 0).sum())
        return prov_df.copy(), pd.DataFrame()

    import blocksnet.optimization.services.acl.provision_adapter as pa_mod

    monkeypatch.setattr(pa_mod, "competitive_provision", _fake_competitive)
    monkeypatch.setattr(pa_mod, "provision_strong_total", lambda _df: 0.5)

    # Place a unit only in block 2.
    variables_df = pd.DataFrame(
        [
            {
                "block_id": 2,
                "service_type": "school",
                "site_area": 0,
                "build_floor_area": 100,
                "capacity": 10,
                "count": 1,
                "total_build_floor_area": 100,
                "total_site_area": 0,
                "total_capacity": 10,
            }
        ]
    )
    adapter.calculate_provision("school", facade._area_checker.build_floor_areas, variables_df)
    assert seen["blocks_with_demand"] == 2, seen


# --------------------------------------------------------------------------- P1-2


def test_units_schema_rejects_degenerate_units():
    bad = pd.DataFrame(
        [
            {"capacity": 1, "site_area": 0, "build_floor_area": 0},
        ]
    )
    with pytest.raises(ValueError, match="site_area=0 and build_floor_area=0"):
        UnitsSchema(bad)


# --------------------------------------------------------------------------- P0-3


def test_units_schema_inlines_parking_into_area_resources():
    """P0-3: parking_area must end up in site_area / build_floor_area so that
    the optimizer constraints see it instead of silently dropping it."""
    df = pd.DataFrame(
        [
            # embedded unit: parking folds into BFA
            {
                "service_type": "school",
                "capacity": 1,
                "site_area": 0,
                "build_floor_area": 200,
                "parking_area": 50,
            },
            # standalone unit: parking extends site area
            {
                "service_type": "school",
                "capacity": 1,
                "site_area": 1000,
                "build_floor_area": 800,
                "parking_area": 50,
            },
        ]
    )
    result = UnitsSchema(df)
    rows = result.reset_index(drop=True)
    assert rows.loc[0, "build_floor_area"] == 250
    assert rows.loc[0, "site_area"] == 0
    assert rows.loc[1, "site_area"] == 1050
    assert rows.loc[1, "build_floor_area"] == 800


# --------------------------------------------------------------------------- P1-3


def test_get_delta_demand_balances_population_then_per_service(monkeypatch):
    """P1-3: demand must equal ``round(population / 1000 * service_demand)``
    with population rounded once."""
    facade, _, _ = _build_facade({1: 30_000.0}, {1: {"school": 0}})
    facade._converter = _StubConverter([])
    facade._chosen_service_types = {"school"}
    import blocksnet.optimization.services.acl.facade as f_mod

    class _StubConfig:
        def __getitem__(self, _key):
            return {"name_ru": "school", "demand": 100, "accessibility": 15}

        def __getattr__(self, name):
            # Real callers (get_block_services, etc.) walk into ``land_use`` /
            # ``units`` attributes; raise so the test stays scoped.
            raise AttributeError(name)

    monkeypatch.setattr(f_mod, "service_types_config", _StubConfig())
    # 30_000 residual BFA (no service BFA consumed) ⇒
    #   population = 30_000 * (1/0.3 - 1) / 25 = 2800
    #   demand = round(2800 / 1000 * 100) = 280
    demand = facade.get_delta_demand(np.array([]))
    assert demand[1]["school"] == 280
