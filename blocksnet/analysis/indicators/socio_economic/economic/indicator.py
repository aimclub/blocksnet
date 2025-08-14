from enum import Enum, unique


@unique
class EconomicIndicator(Enum):
    FIXED_CAPITAL_INVESTMENT_PER_CAPITA = "fixed_capital_investment_per_capita"
    GRP_PER_CAPITA = "grp_per_capita"
    BUDGET_REVENUE = "budget_revenue"
    AVERAGE_WAGE = "average_wage"
    FIXED_ASSETS_DEPRECATION = "fixed_assets_deprecation"
    ORGANIZATIONS = "organizations_count"
    SOLE_PROPRIETORS = "sole_proprietors_count"
