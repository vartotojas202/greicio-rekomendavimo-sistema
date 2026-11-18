"""Papildomi eksperimentai: (6) hibridinis modelis „fizika + GBM likutis“ ir (7) 4 klasių rekomendacijų klasifikavimo metrikos."""
import os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.py")).read().split("results = {}")[0]
__file__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.py")
exec(src)

def hybrid_fit(trn, quant=None, it=200):
    X = trn.copy(); X["b3"] = b3_physics(X); y = X[Y] - X["b3"]
    cols = ALL + ["b3"]
    kw = dict(loss="squared_error" if quant is None else "quantile", max_iter=it, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=50,
              l2_regularization=1.0, random_state=SEED, early_stopping=False)
    if quant: kw["quantile"] = quant
    m = HistGradientBoostingRegressor(**kw).fit(X[cols], y)
    def f(d):
        d = d.copy(); d["b3"] = b3_physics(d)
        return np.clip(d["b3"].values + m.predict(d[cols]), 20, d.legal_limit.values)
    return f

rows = []
print("== Exp6 hybrid")
trn = sub(train)
h1 = hybrid_fit(trn); h2 = hybrid_fit(trn, 0.10)
rows += [dict(eksperimentas="Visi ruožai, 3 m. testas", **eval_pred(test[Y].values, h1(test), test.legal_limit.values, "P3 Hibridas (fizika + GBM likutis)")),
         dict(eksperimentas="Visi ruožai, 3 m. testas", **eval_pred(test[Y].values, h2(test), test.legal_limit.values, "P4 Hibridas q10 (konservatyvus)"))]
rng = np.random.default_rng(SEED); segs = np.arange(30); rng.shuffle(segs); hold = segs[:6]
trs = train[~train.segment.isin(hold)]; tes = test[test.segment.isin(hold)]
h1s = hybrid_fit(sub(trs)); h2s = hybrid_fit(sub(trs), 0.10)
rows += [dict(eksperimentas="Nematyti ruožai", **eval_pred(tes[Y].values, h1s(tes), tes.legal_limit.values, "P3 Hibridas (fizika + GBM likutis)")),
         dict(eksperimentas="Nematyti ruožai", **eval_pred(tes[Y].values, h2s(tes), tes.legal_limit.values, "P4 Hibridas q10 (konservatyvus)"))]
tc = test.copy()
for c in ["rws_state", "state_prev3", "state_max_6h"]: tc[c] = np.nan
rows += [dict(eksperimentas="RWS dangos jutiklis neveikia", **eval_pred(test[Y].values, h1(tc), test.legal_limit.values, "P3 Hibridas (fizika + GBM likutis)")),
         dict(eksperimentas="RWS dangos jutiklis neveikia", **eval_pred(test[Y].values, h2(tc), test.legal_limit.values, "P4 Hibridas q10 (konservatyvus)"))]
pd.DataFrame(rows).to_csv(os.path.join(OUT, "exp6_hybrid.csv"), index=False)
print(pd.DataFrame(rows).round(2).to_string())
sv = pd.read_pickle(os.path.join(OUT, "test_preds.pkl")); sv["P3 Hibridas"] = h1(test); sv["P4 Hibridas q10"] = h2(test)
sv.to_pickle(os.path.join(OUT, "test_preds.pkl"))

print("== Exp7 4 klasės")
def cls(v, lim): 
    r = v / lim; return np.digitize(r, [0.5, 0.7, 0.9])
names = {"B0 Teisinė riba": "B0", "B1 Taisyklės": "B1", "B2 LR": "B2", "B3 Fizika": "B3", "P1 GBM": "P1", "P2 GBM q10": "P2", "P2*": "P2* (apvalinta)", "P3 Hibridas": "P3", "P4 Hibridas q10": "P4"}
yt = cls(sv[Y].values, sv.legal_limit.values); rc = []
for k, nm in names.items():
    yp = cls(sv[k].values, sv.legal_limit.values)
    rc.append(dict(model=nm, accuracy=accuracy_score(yt, yp) * 100, F1_macro=f1_score(yt, yp, average="macro"), F1_weighted=f1_score(yt, yp, average="weighted"),
                   unsafe_class_rate=float((yp > yt).mean() * 100)))   # rekomenduotas aukštesnės (mažiau ribojančios) klasės nei tikrasis
    # klasių indeksai: 0 – <50% ribos (labai ribojama) ... 3 – ≥90% ribos; 'unsafe' = rekomenduota daugiau nei reikia
pd.DataFrame(rc).to_csv(os.path.join(OUT, "exp7_classes.csv"), index=False); print(pd.DataFrame(rc).round(2).to_string())
print("klasių pasiskirstymas testo:", np.bincount(yt) / len(yt) * 100)
from sklearn.metrics import confusion_matrix
for k in ["P1 GBM", "P2 GBM q10"]:
    cm = confusion_matrix(yt, cls(sv[k].values, sv.legal_limit.values)); np.save(os.path.join(OUT, f"cm_{k.split()[0]}.npy"), cm)
