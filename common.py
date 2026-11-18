"""Bendros konstantos ir fizikinis saugaus greičio modelis (naudojamas etiketėms ir baziniam modeliui B3)."""
import numpy as np

REACTION_S = 2.5          # AASHTO reakcijos laikas, s
FRICTION_SAFETY = 0.8     # projektinė trintis = 0.8 * mu (padangų / ABS variacija)

STATES = ["sausa", "drėgna", "šlapia", "šlapdriba", "sniegas", "plikledis"]
# Tipinės trinties reikšmės (Gao ir kt., core.ac.uk/267923515; SIRWEC Quebec D11; AASHTO)
MU_BY_STATE = np.array([0.80, 0.70, 0.50, 0.35, 0.28, 0.14])
# Kažys & Jančiauskaitė (2022), 1 lentelė, E85: avarijų dalis % ir laiko dalis % pagal dangą
KAZYS_ACC_SHARE = np.array([65.6, 13.3, 15.9, 1.5, 2.0, 1.7])
KAZYS_TIME_SHARE = np.array([67.7, 21.0, 10.2, 0.4, 0.4, 0.3])

def kazys_relative_risk():
    rr = KAZYS_ACC_SHARE / KAZYS_TIME_SHARE
    return rr / rr[0]

def ssd_speed(mu, sight_m):
    """V (km/h), kai reakcijos kelias + V^2/(254 f) = matomas atstumas."""
    f = np.maximum(mu * FRICTION_SAFETY, 0.03)
    a = REACTION_S / 3.6
    return 254.0 * f * (-a + np.sqrt(a * a + 4.0 * sight_m / (254.0 * f))) / 2.0

def curve_speed(radius_m, mu, superelev=0.05):
    f_side = 0.6 * mu * FRICTION_SAFETY
    return np.sqrt(127.0 * radius_m * np.maximum(f_side + superelev, 0.01))

def wind_factor(gust_ms):
    return np.clip(1.0 - 0.03 * np.maximum(gust_ms - 14.0, 0.0), 0.55, 1.0)

def flow_factor(veh_per_h, lanes):
    util = np.clip(veh_per_h / (1800.0 * lanes), 0, 1.2)
    return np.clip(1.0 - 0.18 * np.maximum(util - 0.5, 0.0), 0.88, 1.0)

def safe_speed(mu, sight_m, radius_m, gust_ms, veh_per_h, lanes, legal_limit):
    v = np.minimum(ssd_speed(mu, sight_m), curve_speed(radius_m, mu))
    v = v * wind_factor(gust_ms) * flow_factor(veh_per_h, lanes)
    return np.clip(np.minimum(v, legal_limit), 20.0, legal_limit)
