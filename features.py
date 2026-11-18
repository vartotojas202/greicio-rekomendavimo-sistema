"""Požymių inžinerija (ta pati logika kaip model.py) – naudojama demo.py ir realių duomenų režime."""
import numpy as np, pandas as pd

# ------------------------------------------------------------------ požymiai
def build_features(d):
    g = d.groupby("segment", sort=False)
    d["dewp"] = d.rws_air - ((100 - d.rws_rh) / 5.0)                      # apytikslis rasos taškas (Lawrence)
    d["surf_dew"] = d.rws_surf - d.dewp
    d["air_surf"] = d.rws_air - d.rws_surf
    d["pws_air_diff"] = d.pws_t - d.rws_air
    p_any = d.met_p.fillna(0)
    d["p_obs"] = np.where(d.rws_pint.notna(), d.rws_pint, p_any)
    for k in (3, 12, 24):
        d[f"p_sum_{k}h"] = g["p_obs"].transform(lambda s: s.rolling(k, min_periods=1).sum())
    d["snow_flag"] = (d.rws_ptype == 3).astype(float)
    d["snow_12h"] = d.groupby("segment", sort=False)["snow_flag"].transform(lambda s: s.rolling(12, min_periods=1).sum())
    d["surf_min_12h"] = g["rws_surf"].transform(lambda s: s.rolling(12, min_periods=1).min())
    d["surf_max_12h"] = g["rws_surf"].transform(lambda s: s.rolling(12, min_periods=1).max())
    d["frz_h_24"] = g["rws_surf"].transform(lambda s: (s < 0).astype(float).rolling(24, min_periods=1).sum())
    d["dT_3h"] = g["rws_air"].transform(lambda s: s - s.shift(3)).fillna(0)
    d["surf_dT_3h"] = g["rws_surf"].transform(lambda s: s - s.shift(3)).fillna(0)
    d["state_prev3"] = g["rws_state"].transform(lambda s: s.shift(3))
    d["state_max_6h"] = g["rws_state"].transform(lambda s: s.rolling(6, min_periods=1).max())
    d["hr_s"] = np.sin(2 * np.pi * d.time.dt.hour / 24); d["hr_c"] = np.cos(2 * np.pi * d.time.dt.hour / 24)
    d["doy_s"] = np.sin(2 * np.pi * d.time.dt.dayofyear / 365); d["doy_c"] = np.cos(2 * np.pi * d.time.dt.dayofyear / 365)
    d["weekend"] = (d.time.dt.dayofweek >= 5).astype(np.int8)
    d["traffic_util"] = d.traffic / (1800.0 * d.lanes)
    return d


GROUPS = {
    "seg":  ["legal_limit", "road_class", "lanes", "sight_geo", "curve_radius", "pave_age", "lit", "bridge", "gravel", "night", "sin_elev", "hr_s", "hr_c", "doy_s", "doy_c", "weekend"],
    "met":  ["met_t", "met_p", "met_wind", "met_gust", "met_rh", "met_pres", "d_met_km", "p_sum_3h", "p_sum_12h", "p_sum_24h"],
    "rws":  ["rws_air", "rws_surf", "rws_rh", "rws_wind", "rws_gust", "rws_vis", "rws_pint", "rws_ptype", "rws_state", "dewp", "surf_dew", "air_surf",
             "snow_12h", "surf_min_12h", "surf_max_12h", "frz_h_24", "dT_3h", "surf_dT_3h", "state_prev3", "state_max_6h", "d_rws_km"],
    "pws":  ["pws_t", "pws_rh", "pws_p", "pws_air_diff"],
    "fc":   ["fc_t3", "fc_t6", "fc_p3", "fc_p6", "fc_w3"],
    "traf": ["traffic", "traffic_util", "aadt", "hv_share"],
}
ALL = [c for g in GROUPS.values() for c in g]
