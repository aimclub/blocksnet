from enum import unique
from ..indicator_enum import IndicatorEnum
from ..indicator_meta import IndicatorMeta


@unique
class EconomicIndicator(IndicatorEnum):
    """Economic indicators describing employment and business activity."""
    FIXED_CAPITAL_INVESTMENT_PER_CAPITA = IndicatorMeta("fixed_capital_investment_per_capita", "capita")
    GRP_PER_CAPITA = IndicatorMeta("grp_per_capita", "capita")
    BUDGET_REVENUE = IndicatorMeta("budget_revenue")
    AVERAGE_WAGE = IndicatorMeta("average_wage", "capita")
    FIXED_ASSETS_DEPRECATION = IndicatorMeta("fixed_assets_deprecation")
    ORGANIZATIONS = IndicatorMeta("organizations_count")
    SOLE_PROPRIETORS = IndicatorMeta("sole_proprietors_count")
