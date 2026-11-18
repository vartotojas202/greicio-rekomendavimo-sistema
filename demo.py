"""Demonstracija: konservatyvaus modelio (P2, GBM q10) rekomendacijos pagal 3 dienų pavyzdį data/sample_3days.csv.
Paleidimas:  python code/demo.py [--segment 12] [--time "2026-01-02 07:00"]
Išvestis: rekomenduojamas greitis (km/h, apvalinamas žemyn iki 5), teisinė riba ir paaiškinimas žmogui.  PATARIAMASIS įrankis – nevaldo ženklų."""
import argparse, os, sys, joblib, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
from features import build_features
from common import STATES
ap = argparse.ArgumentParser(); ap.add_argument("--segment", type=int, default=12); ap.add_argument("--time", default="2026-01-02 07:00"); a = ap.parse_args()
d = pd.read_csv(os.path.join(H, "..", "data", "sample_3days.csv"), parse_dates=["time"]).sort_values(["segment", "time"]).reset_index(drop=True)
d = build_features(d)
art = joblib.load(os.path.join(H, "..", "results", "model_p2_q10.joblib"))
r = d[(d.segment == a.segment) & (d.time == pd.Timestamp(a.time))]
if r.empty: sys.exit("Nėra tokio laiko/ruožo pavyzdyje (galimas laikotarpis 2026-01-01..2026-01-03).")
v = float(np.clip(art["model"].predict(r[art["cols"]])[0], 20, r.legal_limit.iloc[0])); rec = min(np.floor(v / 5) * 5, r.legal_limit.iloc[0])
st = r.rws_state.iloc[0]
guard = False
if st != st and float(r.rws_surf.fillna(r.rws_air).iloc[0]) < 1.5:      # fail-safe: dangos jutiklis neveikia ir šalta
    cap = 0.6 * r.legal_limit.iloc[0]
    if rec > cap: rec, guard = np.floor(cap / 5) * 5, True
print(f"Ruožas {a.segment}, {a.time}\n  Teisinė riba: {r.legal_limit.iloc[0]:.0f} km/h\n  RWS dangos būsena: {STATES[int(st)] if st == st else 'nežinoma (jutiklis neveikia)'}; "
      f"oro T {r.rws_air.iloc[0]:.1f} °C, dangos T {r.rws_surf.iloc[0]:.1f} °C, matomumas {r.rws_vis.iloc[0]:.0f} m, gūsiai {r.rws_gust.iloc[0]:.1f} m/s")
print(f"  >>> REKOMENDUOJAMAS GREITIS: {rec:.0f} km/h   (modelio q10 įvertis {v:.1f} km/h)")
if guard: print("  ! Aktyvuota apsauginė taisyklė: dangos jutiklis neveikia, o dangos T < 1,5 °C -> riba ribojama iki 60 % teisinės.")
print("  Tikrasis saugus greitis sintetiniuose duomenyse (tik kontrolei):", f"{r.v_safe.iloc[0]:.1f} km/h")
