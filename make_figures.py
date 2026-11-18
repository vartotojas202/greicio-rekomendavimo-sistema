"""Sukuria visas ataskaitos iliustracijas iš results/*.csv ir test_preds.pkl."""
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, os.path.dirname(__file__)); from common import *
H = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(H, "..", "results"); FIG = os.path.join(H, "..", "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
C = {"B0": "#9aa0a6", "B1": "#c0a060", "B2": "#7fa6c9", "B3": "#5b8a72", "P1": "#d1603d", "P2": "#8e4585", "P3": "#c0392b", "P4": "#6c3483"}
def save(n): plt.tight_layout(); plt.savefig(os.path.join(FIG, n), bbox_inches="tight"); plt.close()

# 1. dangos būsenų pasiskirstymas: simuliuota (automagistralės) vs tikra (Kažys, A1/E85)
sv = pd.read_pickle(os.path.join(RES, "test_preds.pkl"))
dfall = pd.read_pickle(os.path.join(H, "..", "data", "dataset.pkl"), compression=None)[["state_true", "road_class"]]
sim = np.bincount(dfall[dfall.road_class == 0].state_true, minlength=6) / (dfall.road_class == 0).sum() * 100
fig, ax = plt.subplots(figsize=(7, 3.6)); x = np.arange(6); w = 0.38
ax.bar(x - w / 2, KAZYS_TIME_SHARE, w, label="Tikra: A1/E85 RWS 2017–2019 (Kažys ir Jančiauskaitė, 2022)", color="#4c78a8")
ax.bar(x + w / 2, sim, w, label="Sintetinė: automagistralių klasė", color="#f58518")
ax.set_xticks(x); ax.set_xticklabels(STATES); ax.set_yscale("log"); ax.set_ylabel("Laiko dalis, % (log skalė)"); ax.legend(fontsize=8, frameon=False)
save("fig1_surface_distribution.png")

# 2. pavyzdinė savaitė
d = sv[(sv.road_class == 0)].copy(); d["wk"] = d.time.dt.to_period("W")
bad = d.assign(b=(d.state_true >= 4)).groupby(["segment", "wk"]).b.sum().sort_values(ascending=False)
seg, wk = bad.index[0]; e = d[(d.segment == seg) & (d.wk == wk)].sort_values("time")
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
a1.plot(e.time, e.legal_limit, color=C["B0"], lw=1.2, ls="--", label="Teisinė riba (B0)")
a1.plot(e.time, e.v_safe, color="black", lw=1.6, label="Tikrasis saugus greitis (latentinis)")
a1.plot(e.time, e["B3 Fizika"], color=C["B3"], lw=1, label="B3 fizika pagal stebimą kategoriją")
a1.plot(e.time, e["P3 Hibridas"], color=C["P3"], lw=1.2, label="P3 hibridas")
a1.plot(e.time, e["P4 Hibridas q10"], color=C["P4"], lw=1.2, label="P4 konservatyvus (q10)")
a1.set_ylabel("Greitis, km/h"); a1.legend(fontsize=7.5, ncol=2, frameon=False, loc="lower left")
cols = ["#e8f1e8", "#d7ecf6", "#a8cfe8", "#cdb6e6", "#f0d9a7", "#f4a9a0"]
a2.scatter(e.time, e.state_true, c=[cols[int(s)] for s in e.state_true], s=14, edgecolor="grey", lw=0.3)
a2.set_yticks(range(6)); a2.set_yticklabels(STATES, fontsize=7); a2.set_ylabel("Tikra danga")
plt.setp(a2.get_xticklabels(), rotation=20)
a1.set_title(f"Ruožas {seg} (automagistralė), savaitė {wk}", fontsize=9)
save("fig2_example_week.png")

# 3. pagrindinis palyginimas
e1 = pd.read_csv(os.path.join(RES, "exp1_main.csv")); e6 = pd.read_csv(os.path.join(RES, "exp6_hybrid.csv"))
e6a = e6[e6.eksperimentas.str.startswith("Visi")].drop(columns="eksperimentas")
m = pd.concat([e1[~e1.model.str.startswith("P2*")], e6a]).reset_index(drop=True)
short = ["B0 teis. riba", "B1 taisyklės", "B3 fizika", "B2 tiesinė reg.", "P1 GBM", "P2 GBM q10", "P3 hibridas", "P4 hibr. q10"]
colors = [C["B0"], C["B1"], C["B3"], C["B2"], C["P1"], C["P2"], C["P3"], C["P4"]]
fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
for a, col, t in zip(ax, ["MAE", "overshoot_rate", "critical_rate"], ["MAE, km/h", "Viršijimo >5 km/h dalis, %", "Kritinis viršijimas >15 km/h, %"]):
    a.barh(short, m[col], color=colors); a.invert_yaxis(); a.set_title(t, fontsize=9)
    for i, v in enumerate(m[col]): a.text(v, i, f" {v:.2f}", va="center", fontsize=7.5)
for a in ax[1:]: a.set_yticklabels([])
save("fig3_main_comparison.png")

# 4. ablacija
ab = pd.read_csv(os.path.join(RES, "exp2_ablation.csv"))
lab = ["Ruožo atributai", "+ meteo.lt", "+ PWS (Wunderground/\nNetatmo)", "+ eismoinfo RWS\n(be PWS)", "+ RWS + PWS", "+ prognozė\n(Windy/Harmonie)", "+ eismo\nintensyvumas"]
fig, ax = plt.subplots(figsize=(8, 3.6)); ax.bar(range(7), ab.MAE, color="#4c78a8")
for i, (v, o) in enumerate(zip(ab.MAE, ab.overshoot_rate)): ax.text(i, v + 0.05, f"{v:.2f}\n({o:.1f}%)", ha="center", fontsize=7.5)
ax.set_xticks(range(7)); ax.set_xticklabels(lab, fontsize=7.5); ax.set_ylabel("MAE, km/h"); ax.set_ylim(0, ab.MAE.max() * 1.25)
ax.set_title("Šaltinių ablacija (P1, 3 m. testas); skliaustuose – viršijimo >5 km/h dalis", fontsize=9)
save("fig4_ablation.png")

# 5. atsparumas
rb = pd.read_csv(os.path.join(RES, "exp3_robustness.csv")); rb = rb[rb.model != "B1 Taisyklės"]
scen = ["Be gedimų", "RWS dangos jutiklis neveikia", "Visa RWS stotis neveikia", "RWS + PWS neveikia (tik meteo.lt + prognozė)"]
sl = ["Be gedimų", "Dangos jutiklis\nneveikia", "Visa RWS\nneveikia", "RWS + PWS\nneveikia"]
fig, ax = plt.subplots(figsize=(8, 3.6)); w = 0.2
for j, (mn, c) in enumerate([("B3 Fizika", C["B3"]), ("P1 GBM", C["P1"]), ("P2 GBM q10", C["P2"])]):
    v = [rb[(rb.model == mn) & (rb.scenarijus == s)].MAE.values[0] for s in scen]; ax.bar(np.arange(4) + (j - 1) * w, v, w, label=mn, color=c)
ax.set_xticks(range(4)); ax.set_xticklabels(sl); ax.set_ylabel("MAE, km/h"); ax.legend(frameon=False, fontsize=8)
save("fig5_robustness.png")

# 6. svarba
im = pd.read_csv(os.path.join(RES, "importance.csv")).head(15)
gc = {"seg": "#7f7f7f", "met": "#4c78a8", "rws": "#e45756", "pws": "#72b7b2", "fc": "#b279a2", "traf": "#f58518"}
gn = {"seg": "ruožo atributai/laikas", "met": "meteo.lt", "rws": "eismoinfo RWS", "pws": "PWS", "fc": "prognozė", "traf": "eismas"}
fig, ax = plt.subplots(figsize=(7, 4.4)); ax.barh(im.feature[::-1], im.importance[::-1], color=[gc[g] for g in im.group[::-1]])
ax.set_xlabel("Permutacinė svarba (MAE padidėjimas, km/h)")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=gc[g]) for g in gn], labels=list(gn.values()), fontsize=7.5, frameon=False, loc="lower right")
save("fig6_importance.png")

# 7. pagal būseną
bs = pd.read_csv(os.path.join(RES, "by_state.csv")); fig, ax = plt.subplots(figsize=(8, 3.6)); w = 0.2
for j, (mn, c) in enumerate([("B0 Teisinė riba", C["B0"]), ("B3 Fizika", C["B3"]), ("P1 GBM", C["P1"]), ("P2 GBM q10", C["P2"])]):
    v = [bs[(bs.model == mn) & (bs.state == s)].overshoot_rate.values[0] for s in STATES]; ax.bar(np.arange(6) + (j - 1.5) * w, v, w, label=mn, color=c)
ax.set_xticks(range(6)); ax.set_xticklabels(STATES); ax.set_ylabel("Viršijimo >5 km/h dalis, %"); ax.legend(frameon=False, fontsize=8)
save("fig7_overshoot_by_state.png")

# 8. painiavos matrica
cm = np.load(os.path.join(RES, "cm_P2.npy")); cmn = cm / cm.sum(1, keepdims=True) * 100
cl = ["<50 %", "50–70 %", "70–90 %", "≥90 %"]
fig, ax = plt.subplots(figsize=(4.6, 4)); ax.imshow(cmn, cmap="Blues", vmin=0, vmax=100)
for i in range(4):
    for j in range(4): ax.text(j, i, f"{cmn[i, j]:.0f}%\n({cm[i, j]})", ha="center", va="center", fontsize=7.5, color="white" if cmn[i, j] > 50 else "black")
ax.set_xticks(range(4)); ax.set_xticklabels(cl); ax.set_yticks(range(4)); ax.set_yticklabels(cl); ax.set_xlabel("Rekomenduota (P2), % teisinės ribos"); ax.set_ylabel("Tikra klasė")
save("fig8_confusion.png")

# 9. architektūra
fig, ax = plt.subplots(figsize=(10, 4.2)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4.2)
def box(x, y, w, h, t, c="#eef3f8", fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=c, ec="#555", lw=0.8)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)
def arr(x1, y1, x2, y2): ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=10, color="#444"))
src = ["meteo.lt API\n(stebėjimai, prognozė)", "eismoinfo.lt\n(RWS, intensyvumas)", "Wunderground /\nNetatmo (PWS)", "Windy / Open-Meteo\n(prognozė, ERA5)", "Ruožo atributai\n(OSM, KET, LAKD)"]
for i, s in enumerate(src): box(0.1, 3.4 - i * 0.82, 2.1, 0.62, s, "#fdf1e0"); arr(2.25, 3.7 - i * 0.82, 2.95, 2.1)
box(3.0, 1.5, 1.7, 1.2, "Surinkimas,\nvalymas, laiko\nsuderinimas\n(valandinis tinklelis,\nartimiausia stotis)")
arr(4.75, 2.1, 5.25, 2.1)
box(5.3, 1.5, 1.7, 1.2, "Požymių inžinerija\n(rasos taškas, 3/12/24 h\nkritulių sumos, šalčio\nvalandos, saulės aukštis)")
arr(7.05, 2.1, 7.55, 2.1)
box(7.6, 1.5, 2.2, 1.2, "Modelis:\nB3 fizika (stabdymo kelias)\n+ GBM likutis (P3)\n+ q10 konservatyvus (P4)", "#e3f1e3")
arr(8.7, 1.45, 8.7, 0.95)
box(7.2, 0.15, 2.8, 0.8, "Rekomenduojamas greitis ruožui\n(patariamasis, ≤ teisinė riba; KIŽ – tik su žmogaus patvirtinimu)", "#fbe3e0", 7.5)
ax.text(3.0, 3.1, "Duomenų srautas (batch / kas 5–10 min)", fontsize=8, style="italic")
save("fig9_architecture.png")
print("ok")
