from blocksnet.models.land_use import LandUse


class Indicator:
    def __init__(self, FSI_min, FSI_max, site_coverage_min, site_coverage_max):
        self.FSI_min = FSI_min  # минимальный коэффициент плотности застройки
        self.FSI_max = FSI_max  # максимальный коэффициент плотности застройки
        self.site_coverage_min = site_coverage_min  # минимальный процент застроенности участка
        self.site_coverage_max = site_coverage_max  # максимальный процент застроенности участка


LAND_USE_INDICATORS = {
    LandUse.RESIDENTIAL: Indicator(FSI_min=0.5, FSI_max=3.0, site_coverage_min=0.2, site_coverage_max=0.8),
    LandUse.BUSINESS: Indicator(FSI_min=1.0, FSI_max=3.0, site_coverage_min=0.0, site_coverage_max=0.8),
    LandUse.RECREATION: Indicator(FSI_min=0.05, FSI_max=0.2, site_coverage_min=0.0, site_coverage_max=0.3),
    LandUse.SPECIAL: Indicator(FSI_min=0.05, FSI_max=0.2, site_coverage_min=0.05, site_coverage_max=0.15),
    LandUse.INDUSTRIAL: Indicator(FSI_min=0.3, FSI_max=1.5, site_coverage_min=0.2, site_coverage_max=0.8),
    LandUse.AGRICULTURE: Indicator(FSI_min=0.1, FSI_max=0.2, site_coverage_min=0.0, site_coverage_max=0.6),
    LandUse.TRANSPORT: Indicator(FSI_min=0.2, FSI_max=1.0, site_coverage_min=0.0, site_coverage_max=0.8),
}
