def wm2_to_annual_kwh(avg_wm2):
    return avg_wm2 * 8760 / 1000


def estimate_series7_energy(
    annual_irradiation_kwh_m2,
    area=2.80,
    efficiency=0.197,
    performance_ratio=0.85
):
    return annual_irradiation_kwh_m2 * area * efficiency * performance_ratio

