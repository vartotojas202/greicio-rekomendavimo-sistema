"""Požymių inžinerija, bazinių ir siūlomų modelių mokymas, eksperimentai (palyginimas, ablacija, atsparumas, erdvinis testas)."""
import numpy as np, pandas as pd, json, os, sys, time as _t
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, accuracy_score
from sklearn.inspection import permutation_importance

HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "..", "data"); OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)
SEED = 42

df = pd.read_pickle(os.path.join(DATA, "dataset.pkl"))
df = df.sort_values(["segment", "time"]).reset_index(drop=True)

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

t0 = _t.time(); df = build_features(df); print("požymiai", round(_t.time() - t0), "s")

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
Y = "v_safe"

train = df[df.year_idx <= 1]; val = df[df.year_idx == 1]; tr1 = df[df.year_idx == 0]; test = df[df.year_idx == 2]
STRIDE = 3
def sub(d, k=STRIDE): return d.iloc[::k]

# ------------------------------------------------------------------ metrikos
def eval_pred(y, p, legal, name):
    e = p - y
    cls_t = np.floor(y / 10).astype(int); cls_p = np.floor(p / 10).astype(int)
    return dict(model=name, MAE=mean_absolute_error(y, p), RMSE=mean_squared_error(y, p) ** 0.5, R2=r2_score(y, p),
                overshoot_rate=float((e > 5).mean() * 100), critical_rate=float((e > 15).mean() * 100),
                mean_overshoot=float(np.maximum(e, 0).mean()), mobility_loss=float(np.maximum(-e, 0).mean()),
                within10=float((np.abs(e) <= 10).mean() * 100), acc10=float(accuracy_score(cls_t, cls_p) * 100),
                F1_macro=float(f1_score(cls_t, cls_p, average="macro")))

# ------------------------------------------------------------------ baziniai modeliai
def b0_legal(d): return d.legal_limit.values

def b1_rules(d):
    """Taisyklėmis grįstas (Georgia DDS / Via Lietuva stiliaus): sumažinti ribą pagal stebimą dangos kategoriją ir matomumą."""
    f = np.array([1.0, 1.0, 0.80, 0.65, 0.55, 0.40])
    st = d.rws_state.fillna(2).astype(int).values
    v = d.legal_limit.values * f[st]
    vis = d.rws_vis.fillna(10000); gu = d.rws_gust.fillna(d.met_gust)
    v = np.where(vis < 200, np.minimum(v, 50), np.where(vis < 1000, np.minimum(v, 0.8 * d.legal_limit.values), v))
    v = np.where(gu > 20, v * 0.85, v)
    return np.maximum(v, 30)

def b3_physics(d):
    """Fizikinis modelis, taikomas STEBIMAI kategorijai (lentelinė trintis) – stipriausias nepriklausomas pagrindas."""
    st = d.rws_state.fillna(2).astype(int).values
    mu = MU_BY_STATE[st]
    vis = d.rws_vis.fillna(10000).values; gu = d.rws_gust.fillna(d.met_gust).values
    sight = np.minimum(np.minimum(d.sight_geo.values, vis), np.where((d.night.values == 1) & (d.lit.values == 0), 140.0, 1e9))
    return safe_speed(mu, sight, d.curve_radius.values, gu, d.traffic.values, d.lanes.values, d.legal_limit.values)

def lr_fit(Xtr, ytr, cols):
    med = Xtr[cols].median()
    m = LinearRegression().fit(Xtr[cols].fillna(med), ytr)
    return lambda X: np.clip(m.predict(X[cols].fillna(med)), 20, X.legal_limit.values)

def gb_fit(Xtr, ytr, cols, loss="squared_error", q=None, it=250, lr=0.1, leaf=63, seed=SEED):
    kw = dict(loss=loss, max_iter=it, learning_rate=lr, max_leaf_nodes=leaf, min_samples_leaf=50, l2_regularization=1.0, random_state=seed, early_stopping=False)
    if q is not None: kw["quantile"] = q
    m = HistGradientBoostingRegressor(**kw).fit(Xtr[cols], ytr)
    return m, (lambda X: np.clip(m.predict(X[cols]), 20, X.legal_limit.values))

results = {}
rows = []
def run(name, f, d):
    p = f(d); r = eval_pred(d[Y].values, p, d.legal_limit.values, name); rows.append(r); return p, r

# ---- Eksperimentas 1: pagrindinis palyginimas (mokymas: 1–2 m., testas: 3 m. – atšiaurioji žiema)
print("== Exp1"); t0 = _t.time()
Xtr, ytr = sub(train), sub(train)[Y]
preds = {}
preds["B0 Teisinė riba"], _ = run("B0 Teisinė riba (status quo)", b0_legal, test)
preds["B1 Taisyklės"], _ = run("B1 Taisyklėmis grįstas", b1_rules, test)
preds["B3 Fizika"], _ = run("B3 Fizika pagal stebimą kategoriją", b3_physics, test)
lr = lr_fit(Xtr, ytr, ALL); preds["B2 LR"], _ = run("B2 Tiesinė regresija", lr, test)
gbm, gb = gb_fit(Xtr, ytr, ALL); preds["P1 GBM"], _ = run("P1 Gradientinis stiprinimas (vidurkis)", gb, test)
gbq, gbq_f = gb_fit(Xtr, ytr, ALL, loss="quantile", q=0.10); preds["P2 GBM q10"], _ = run("P2 GBM kvantilis 0,10 (konservatyvus)", gbq_f, test)
# galutinė rekomendacija: P2, apvalinama žemyn iki 10 km/h
p2 = preds["P2 GBM q10"]; reco = np.minimum(np.floor(p2 / 10) * 10, test.legal_limit.values)
r = eval_pred(test[Y].values, reco, test.legal_limit.values, "P2* P2 + apvalinimas žemyn iki 10 km/h"); rows.append(r)
preds["P2*"] = reco
exp1 = pd.DataFrame(rows); exp1.to_csv(os.path.join(OUT, "exp1_main.csv"), index=False); print(exp1.round(2).to_string()); print(round(_t.time() - t0), "s")

# ---- Eksperimentas 2: šaltinių ablacija (GBM vidurkis, tas pats testas)
print("== Exp2 ablation"); rows_a = []
combos = {"Tik ruožo atributai (seg)": ["seg"], "seg + meteo.lt (met)": ["seg", "met"], "seg + met + PWS (Wunderground/Netatmo)": ["seg", "met", "pws"],
          "seg + met + eismoinfo RWS": ["seg", "met", "rws"], "seg + met + RWS + PWS": ["seg", "met", "rws", "pws"],
          "seg + met + RWS + PWS + prognozė (Windy/Harmonie)": ["seg", "met", "rws", "pws", "fc"], "Visi šaltiniai (+ eismo intensyvumas)": list(GROUPS)}
for nm, gs in combos.items():
    cols = [c for g in gs for c in GROUPS[g]]
    _, f = gb_fit(Xtr, ytr, cols, it=150)
    r = eval_pred(test[Y].values, f(test), test.legal_limit.values, nm); r["n_features"] = len(cols); rows_a.append(r); print(nm, round(r["MAE"], 2), round(r["R2"], 3))
pd.DataFrame(rows_a).to_csv(os.path.join(OUT, "exp2_ablation.csv"), index=False)

# ---- Eksperimentas 3: atsparumas jutiklių gedimams (testo duomenyse sugadiname RWS)
print("== Exp3 robustness"); rows_r = []
def corrupt(d, kind):
    d = d.copy()
    rws_cols = [c for c in GROUPS["rws"] if c != "d_rws_km"]
    if kind == "RWS dangos jutiklis neveikia":
        for c in ["rws_state", "state_prev3", "state_max_6h"]: d[c] = np.nan
    elif kind == "Visa RWS stotis neveikia":
        for c in rws_cols: d[c] = np.nan
    elif kind == "RWS + PWS neveikia (tik meteo.lt + prognozė)":
        for c in rws_cols + GROUPS["pws"]: d[c] = np.nan
    return d
for kind in ["Be gedimų", "RWS dangos jutiklis neveikia", "Visa RWS stotis neveikia", "RWS + PWS neveikia (tik meteo.lt + prognozė)"]:
    dd = test if kind == "Be gedimų" else corrupt(test, kind)
    for nm, f in [("B3 Fizika", b3_physics), ("B1 Taisyklės", b1_rules), ("P1 GBM", gb), ("P2 GBM q10", gbq_f)]:
        if nm.startswith("B") and kind != "Be gedimų": 
            dd2 = dd.copy(); 
            try: p = f(dd2)
            except Exception: continue
        else: p = f(dd)
        r = eval_pred(test[Y].values, p, test.legal_limit.values, nm); r["scenarijus"] = kind; rows_r.append(r)
pd.DataFrame(rows_r).to_csv(os.path.join(OUT, "exp3_robustness.csv"), index=False)

# ---- Eksperimentas 4: erdvinis apibendrinimas (nematyti ruožai)
print("== Exp4 spatial")
rng = np.random.default_rng(SEED); segs = np.arange(30); rng.shuffle(segs); hold = segs[:6]
trs = train[~train.segment.isin(hold)]; tes = test[test.segment.isin(hold)]
_, f_sp = gb_fit(sub(trs), sub(trs)[Y], ALL, it=200)
_, f_spq = gb_fit(sub(trs), sub(trs)[Y], ALL, loss="quantile", q=0.10, it=200)
rows_s = [eval_pred(tes[Y].values, b3_physics(tes), tes.legal_limit.values, "B3 Fizika"),
          eval_pred(tes[Y].values, f_sp(tes), tes.legal_limit.values, "P1 GBM (nematyti ruožai)"),
          eval_pred(tes[Y].values, f_spq(tes), tes.legal_limit.values, "P2 GBM q10 (nematyti ruožai)")]
pd.DataFrame(rows_s).to_csv(os.path.join(OUT, "exp4_spatial.csv"), index=False)

# ---- Eksperimentas 5: pasiskirstymo poslinkis – mokymas tik švelniais metais (1 m.), testas atšiauriu (3 m.)
print("== Exp5 shift")
_, f_sh = gb_fit(sub(tr1), sub(tr1)[Y], ALL, it=200)
_, f_sh2 = gb_fit(sub(train), sub(train)[Y], ALL, it=200)
w = test.time.dt.month.isin([12, 1, 2]); tw = test[w]
rows_h = [eval_pred(tw[Y].values, f(tw), tw.legal_limit.values, nm) for nm, f in
          [("B3 Fizika (žiema 3 m.)", b3_physics), ("P1 mokytas tik 1 m. (švelnus)", f_sh), ("P1 mokytas 1–2 m.", f_sh2)]]
pd.DataFrame(rows_h).to_csv(os.path.join(OUT, "exp5_shift.csv"), index=False)

# ---- Pagal dangos būseną (žiemos poaibis, P1 ir P2 ir B3)
by_state = []
for k, nm in enumerate(STATES):
    m = (test.state_true.values == k)
    for mn in ["B0 Teisinė riba", "B3 Fizika", "P1 GBM", "P2 GBM q10"]:
        e = preds[mn][m] - test[Y].values[m]
        by_state.append(dict(state=nm, model=mn, n=int(m.sum()), MAE=float(np.abs(e).mean()), overshoot_rate=float((e > 5).mean() * 100), mean_true=float(test[Y].values[m].mean())))
pd.DataFrame(by_state).to_csv(os.path.join(OUT, "by_state.csv"), index=False)

# ---- XAI: permutacinė svarba (P1) ir dalinė priklausomybė
print("== XAI")
ts = test.sample(20000, random_state=SEED)
pi = permutation_importance(gbm, ts[ALL], ts[Y], n_repeats=3, random_state=SEED, scoring="neg_mean_absolute_error")
imp = pd.DataFrame(dict(feature=ALL, importance=pi.importances_mean, std=pi.importances_std)).sort_values("importance", ascending=False)
imp["group"] = imp.feature.map({c: g for g, cs in GROUPS.items() for c in cs}); imp.to_csv(os.path.join(OUT, "importance.csv"), index=False)
print(imp.head(12).to_string())

# išsaugome prognozes figūroms
sv = test[["time", "segment", "legal_limit", Y, "state_true", "road_class", "rws_state", "rws_surf", "rws_air"]].copy()
for k, v in preds.items(): sv[k] = v
sv.to_pickle(os.path.join(OUT, "test_preds.pkl"))
# kalibracija: kvantilio q10 padengimas
cov = float((gbq_f(test) <= test[Y].values).mean() * 100)
json.dump(dict(q10_coverage=cov, n_train=len(Xtr), n_test=len(test), n_features=len(ALL), hold_segments=[int(x) for x in hold]), open(os.path.join(OUT, "meta.json"), "w"))
print("q10 padengimas %:", round(cov, 1))
import joblib; joblib.dump(dict(model=gbq, cols=ALL), os.path.join(OUT, "model_p2_q10.joblib"))
print("baigta")
