import math

import numpy as np

def weibull_pdf(v, k, c):
    return (k / c) * (v / c)**(k - 1) * np.exp(-(v / c)**k)

def adjust_to_hub_height(v_ref, z_ref=10, z_hub=100, alpha=0.14):
    return v_ref * (z_hub / z_ref) ** alpha

def simple_power_curve(v, cut_in=3.0, rated=12.0, cut_out=25.0, rated_power=3000.0):
    if v < cut_in or v >= cut_out:
        return 0.0
    elif v < rated:
        return rated_power * ((v**3 - cut_in**3) / (rated**3 - cut_in**3))
    else:
        return rated_power

def estimate_aep_from_mean_speed(mean_speed_10m,
                                 k=2.0,
                                 z_ref=10,
                                 z_hub=100,
                                 alpha=0.14,
                                 rated_power_kw=3000.0):

    mean_speed_hub = adjust_to_hub_height(mean_speed_10m, z_ref, z_hub, alpha)

    c = mean_speed_hub / math.gamma(1 + 1/k)

    v = np.linspace(0, 40, 4001)
    pdf = weibull_pdf(v, k, c)
    power = np.array([simple_power_curve(x, rated_power=rated_power_kw) for x in v])

    avg_power_kw = np.trapezoid(power * pdf, v)
    aep_kwh = avg_power_kw  * 8760

    return {
        "mean_speed_hub_mps": mean_speed_hub,
        "average_power_kw": avg_power_kw,
        "annual_energy_kwh": aep_kwh,
        "capacity_factor": avg_power_kw / rated_power_kw
    }
