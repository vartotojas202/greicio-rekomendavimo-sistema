"""Eksperimentas 8: apsauginė (fail-safe) taisyklė, kai RWS dangos jutiklis neveikia ir dangos temperatūra artima nuliui."""
import os
HERE_ = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE_, "model.py")).read().split("results = {}")[0]
__file__ = os.path.join(HERE_, "model.py"); exec(src)
import joblib
art = joblib.load(os.path.join(OUT, "model_p2_q10.joblib"))
def apply_guard(d, p, T_thr=1.5, factor=0.6):
    miss = d.rws_state.isna().values
    cold = (d.rws_surf.fillna(d.rws_air).fillna(d.met_t).values < T_thr)
    return np.where(miss & cold, np.minimum(p, factor * d.legal_limit.values), p), (miss & cold)
tc = test.copy()
for c in ["rws_state", "state_prev3", "state_max_6h"]: tc[c] = np.nan
rows = []
for nm, d in [("Be gedimų (3 % natūralių RWS išjungimų)", test), ("Dangos jutiklis neveikia (100 %)", tc)]:
    p = np.clip(art["model"].predict(d[art["cols"]]), 20, d.legal_limit.values)
    pg, flag = apply_guard(d, p)
    for lab, pp in [("P2", p), ("P2 + apsauga", pg)]:
        r = eval_pred(d[Y].values, pp, d.legal_limit.values, lab); r["scenarijus"] = nm; r["apsauga_aktyvi_%"] = float(flag.mean() * 100) if "apsauga" in lab else 0.0; rows.append(r)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "exp8_safeguard.csv"), index=False); print(pd.DataFrame(rows)[["scenarijus","model","MAE","overshoot_rate","critical_rate","mean_overshoot","mobility_loss","apsauga_aktyvi_%"]].round(2).to_string())
