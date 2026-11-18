"""
Sintetinio (bet fiziškai pagrįsto) duomenų rinkinio generatorius.
Imituoja 5 duomenų šaltinių srautus (meteo.lt, eismoinfo.lt RWS, Wunderground/Netatmo PWS, Windy/Harmonie prognozė)
30 kelio ruožų Lietuvoje per 3 hidrologinius metus (2023-11-01 .. 2026-10-31), valandinė skiriamoji geba.
Slapta (latentinė) kelio dangos būsena ir trintis nustato "tikrąjį" saugų greitį; modelis mato tik triukšmingus matavimus.
"""
import numpy as np, pandas as pd, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *

SEED = 42
rng = np.random.default_rng(SEED)
START = "2023-11-01"
T = 26304                      # 1096 d * 24 h
S = 30
R = 4                          # klimato regionai: 0 vakarai, 1 šiaurė, 2 centras (Kaunas), 3 rytai (Vilnius)

time = pd.date_range(START, periods=T, freq="h")
doy = time.dayofyear.values.astype(float)
hour = time.hour.values.astype(float)
year_idx = np.repeat([0, 1, 2], [365 * 24, 366 * 24, 365 * 24])[:T]
winter_w = 0.5 + 0.5 * np.cos(2 * np.pi * (doy - 17) / 365)       # 1 sausio viduryje, 0 liepos viduryje
YEAR_WINTER_OFFSET = np.array([+1.5, 0.0, -2.5])                   # 1 m. švelnus, 2 m. vidutinis, 3 m. atšiaurus (testas)

def ar1(n, phi, size=None, rng=rng):
    size = () if size is None else (size if isinstance(size, tuple) else (size,))
    e = rng.normal(0, np.sqrt(1 - phi ** 2), (n,) + size)
    x = np.zeros((n,) + size)
    x[0] = rng.normal(0, 1, size)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x

# ------------------------------------------------------------------ regioninė orų simuliacija
season_mean = 7.0 - 11.0 * np.cos(2 * np.pi * (doy - 17) / 365)
diurnal_amp = 2.0 + 1.5 * (1 - winter_w)
diurnal = diurnal_amp * np.cos(2 * np.pi * (hour - 14) / 24)
nat_anom = 4.0 * ar1(T, 0.985)
_wm = np.isin(pd.DatetimeIndex(time).month, [12, 1, 2])
for _y in range(3):      # užtikriname, kad 3-ieji metai tikrai atšiauriausi (testo pasiskirstymo poslinkis)
    _m = nat_anom[(year_idx == _y) & _wm].mean()
    nat_anom[year_idx == _y] -= _m * winter_w[year_idx == _y]
reg_off = np.array([1.0, -0.5, 0.0, -0.3])
reg_anom = 0.8 * ar1(T, 0.95, R)
Ta_reg = (season_mean + diurnal + nat_anom + YEAR_WINTER_OFFSET[year_idx] * winter_w)[:, None] + reg_off + reg_anom

W_nat = ar1(T, 0.97)
W_reg = W_nat[:, None] * 0.9 + 0.45 * ar1(T, 0.95, R)
thr = 1.25 - 0.15 * winter_w[:, None]
wet_occ = W_reg > thr
intensity = rng.gamma(0.7, 0.68, (T, R)) * (1 + 0.6 * np.clip(W_reg - thr, 0, 3))
P_reg = np.where(wet_occ, intensity, 0.0)
ptype_reg = np.where(P_reg <= 0, 0, np.where(Ta_reg < 0.0, 3, np.where(Ta_reg < 1.8, 2, 1)))   # 0 nėra,1 lietus,2 šlapdriba,3 sniegas

cloud_reg = np.clip(0.55 + 0.30 * ar1(T, 0.96, R) + 0.3 * (P_reg > 0), 0, 1)
RH_reg = np.clip(80 - 9 * np.cos(2 * np.pi * (hour - 15) / 24)[:, None] + 11 * (P_reg > 0) + 4 * ar1(T, 0.95, R), 35, 100)
z_w = ar1(T, 0.97, R)
wind_reg = 4.2 * np.exp(0.45 * z_w - 0.1) * (1 + 0.12 * winter_w[:, None])
gust_reg = wind_reg * (1.45 + 0.25 * np.abs(rng.normal(0, 1, (T, R))))
pres_reg = 1013 + 8 * ar1(T, 0.995, R) - 3 * (P_reg > 0)

def dewpoint(Ta, RH):
    a, b = 17.62, 243.12
    g = np.log(np.clip(RH, 1, 100) / 100) + a * Ta / (b + Ta)
    return b * g / (a - g)

sin_lat, cos_lat = np.sin(np.radians(55.0)), np.cos(np.radians(55.0))
decl = np.radians(23.44) * np.sin(2 * np.pi * (doy - 81) / 365)
ha = np.radians((hour - 11.0) * 15)                        # ~vietinis saulės laikas (UTC+2/+3 apytiksliai)
sin_elev = sin_lat * np.sin(decl) + cos_lat * np.cos(decl) * np.cos(ha)
is_night = sin_elev < -0.05

# rūkas ir matomumas (regioninis)
fog_susc = (RH_reg > 95) & (wind_reg < 3.0) & ((hour[:, None] >= 22) | (hour[:, None] <= 9))
fog_state = (ar1(T, 0.9, R) > 0.9) & fog_susc | (ar1(T, 0.97, R) > 2.1)
fog_vis = np.exp(rng.normal(np.log(450), 0.7, (T, R)))
vis_precip = np.where(ptype_reg == 1, 14000 / (1 + P_reg ** 1.2),
              np.where(ptype_reg == 2, 3500 / (0.5 + P_reg),
              np.where(ptype_reg == 3, 5000 / (0.35 + P_reg * 1.3), 25000.0)))
vis_reg = np.clip(np.where(fog_state, fog_vis, np.minimum(25000, vis_precip * np.exp(rng.normal(0, 0.15, (T, R))))), 40, 25000)

# ------------------------------------------------------------------ kelio ruožai
class_counts = {0: 6, 1: 5, 2: 9, 3: 7, 4: 3}
road_class = np.concatenate([[c] * n for c, n in class_counts.items()])
rng.shuffle(road_class)
region = rng.integers(0, R, S)
class_names = {0: "automagistralė", 1: "greitkelis", 2: "magistralinis", 3: "krašto/rajoninis", 4: "žvyrkelis"}
lim_summer = np.array([130, 120, 90, 90, 70])[road_class].astype(float)
lim_winter = np.array([110, 110, 90, 90, 70])[road_class].astype(float)
lanes = np.where(road_class <= 1, 2, 1)
sight_geo = np.array([rng.uniform(350, 600), rng.uniform(300, 500), rng.uniform(220, 400), rng.uniform(150, 300), rng.uniform(100, 200)], dtype=object)
sight_geo = np.array([[rng.uniform(350, 600), rng.uniform(300, 500), rng.uniform(220, 400), rng.uniform(150, 300), rng.uniform(100, 200)][c] for c in road_class])
radius = np.array([[rng.uniform(800, 2000), rng.uniform(600, 1500), rng.uniform(300, 1000), rng.uniform(150, 600), rng.uniform(80, 300)][c] for c in road_class])
aadt = np.array([15000, 9000, 5000, 2000, 300])[road_class] * rng.uniform(0.6, 1.5, S)
hv_share = rng.uniform(0.08, 0.30, S)
pave_age = rng.integers(0, 4, S)
lit = rng.random(S) < np.where(road_class <= 1, 0.15, 0.05)
bridge = rng.random(S) < 0.2
gravel = road_class == 4
mu_pave_lat = -0.03 * pave_age + rng.normal(0, 0.025, S)          # dalis latentinė (mikrotekstūra)
micro_T = rng.normal(0, 0.7, S) - 0.4 * bridge                      # lokalus mikroklimatas
d_rws = rng.uniform(0.5, 14, S)
d_met = rng.uniform(5, 40, S)
lam_pws = np.where(np.isin(road_class, [0, 1]), 3.0, 1.2) * rng.uniform(0.3, 1.6, S)
class_prio = np.array([0.35, 0.30, 0.22, 0.12, 0.05])[road_class]         # valymo intensyvumas / val.
class_salt = np.array([0.95, 0.90, 0.70, 0.35, 0.05])[road_class]

segs = pd.DataFrame(dict(segment=np.arange(S), road_class=road_class, class_name=[class_names[c] for c in road_class],
                         region=region, lim_summer=lim_summer, lim_winter=lim_winter, lanes=lanes, sight_geo=sight_geo,
                         curve_radius=radius, aadt=aadt, hv_share=hv_share, pave_age=pave_age, lit=lit.astype(int),
                         bridge=bridge.astype(int), gravel=gravel.astype(int), d_rws_km=d_rws, d_met_km=d_met))

# ------------------------------------------------------------------ vietiniai (ruožo) orai
reg_idx = region
Ta = Ta_reg[:, reg_idx] + micro_T + 0.5 * ar1(T, 0.9, S)
P_loc = P_reg[:, reg_idx] * np.exp(rng.normal(0, 0.3, (T, S))) * (P_reg[:, reg_idx] > 0)
ptype = ptype_reg[:, reg_idx]
# vietinė rūšies korekcija pagal vietinę temperatūrą
ptype = np.where(P_loc <= 0, 0, np.where(Ta < 0.0, 3, np.where(Ta < 1.8, 2, 1)))
RH = np.clip(RH_reg[:, reg_idx] + rng.normal(0, 2, (T, S)), 30, 100)
Td = dewpoint(Ta, RH)
wind = wind_reg[:, reg_idx] * np.exp(rng.normal(0, 0.1, (T, S)))
gust = gust_reg[:, reg_idx] * np.exp(rng.normal(0, 0.1, (T, S)))
cloud = cloud_reg[:, reg_idx]
fog_loc = np.exp(rng.normal(0, 0.35, S))
vis = np.clip(vis_reg[:, reg_idx] * np.where(vis_reg[:, reg_idx] < 3000, fog_loc, 1.0), 40, 25000)

sun = np.clip(sin_elev, 0, None)[:, None] * (1 - cloud)
night_rad = -2.0 * (1 - cloud) * (sin_elev < 0.05)[:, None] * np.clip(1 - wind / 8.0, 0.1, 1.0)
Ts_raw = Ta + night_rad + 3.5 * sun - 0.6 * bridge
Ts = np.zeros_like(Ts_raw); Ts[0] = Ts_raw[0]
for t in range(1, T):
    Ts[t] = 0.55 * Ts[t - 1] + 0.45 * Ts_raw[t]

# eismas (srautas vienai krypčiai, aut/h)
dow = time.dayofweek.values
prof_wd = 0.01 + 0.05 * np.exp(-0.5 * ((hour - 7.5) / 1.6) ** 2) + 0.06 * np.exp(-0.5 * ((hour - 16.5) / 2.0) ** 2) + 0.03 * np.exp(-0.5 * ((hour - 12) / 3) ** 2)
prof_we = 0.01 + 0.055 * np.exp(-0.5 * ((hour - 13) / 3.5) ** 2) + 0.02 * np.exp(-0.5 * ((hour - 18) / 2.5) ** 2)
prof = np.where(dow >= 5, prof_we, prof_wd)
prof = prof / prof.reshape(-1, 24).mean() / 24 * 1.0
traffic = (aadt[None, :] / 2.0) * prof[:, None] * 24 / 1.0 * np.exp(rng.normal(0, 0.12, (T, S)))
traffic *= np.where(ptype == 3, 0.88, 1.0)
traffic = np.clip(traffic, 0, None)
tr_f = np.clip(traffic / (aadt[None, :] / 2.0 / 24 * 3.0 + 1), 0, 3)  # santykinė apkrova valymo/važinėjimo efektui

# ------------------------------------------------------------------ kelio dangos latentinė dinamika
w_ = np.zeros(S); s_ = np.zeros(S); i_ = np.zeros(S)
state = np.zeros((T, S), dtype=np.int8)
mu = np.zeros((T, S))
W_, S_, I_ = np.zeros((T, S)), np.zeros((T, S)), np.zeros((T, S))
for t in range(T):
    p = P_loc[t]; tp = ptype[t]; ts = Ts[t]; ta = Ta[t]
    pos = ts > 0
    # krituliai
    rain = (tp == 1); sleet = (tp == 2); snow = (tp == 3)
    w_ += np.where(rain & pos, p, 0) + np.where(rain & ~pos, 0.4 * p, 0) + np.where(sleet & pos, 0.6 * p, 0) + np.where(sleet & ~pos, 0.3 * p, 0)
    i_ += np.where(rain & ~pos, 0.6 * p, 0) + np.where(sleet & ~pos, 0.4 * p, 0)
    s_ += np.where(snow, p, 0) + np.where(sleet, 0.4 * p, 0)
    # tirpimas
    melt_s = np.where(pos, np.minimum(s_, 0.6 * ts + 0.1), 0)
    s_ -= melt_s; w_ += melt_s
    melt_i = np.where(pos, np.minimum(i_, 0.4 * ts + 0.05), 0)
    i_ -= melt_i; w_ += melt_i
    # druska
    salt_eff = class_salt * np.clip((ts + 8.0) / 6.0, 0, 1) * (ts < 1.5)
    # užšalimas (juodasis ledas) ir šerkšnas
    frz = np.where(ts < 0, np.minimum(w_, (0.30 + 0.05 * np.minimum(-ts, 6)) * (1 - 0.85 * salt_eff)), 0)
    w_ -= frz; i_ += frz
    frost = (ts < 0) & (ts < Td[t]) & (RH[t] > 85)
    i_ += np.where(frost, 0.04 * (1 - 0.8 * salt_eff), 0)
    dew = (ts > 0) & (ts < Td[t])
    w_ += np.where(dew, 0.03, 0) + np.where(pos & (RH[t] > 88), 0.025, 0)
    # džiūvimas
    tau = 5.0 / (1 + 0.12 * wind[t] + 0.6 * sun[t])
    w_ *= np.where(pos, np.exp(-1.0 / tau), np.exp(-1.0 / (tau * 3)))
    i_ = np.maximum(i_ - 0.01, 0)
    # priežiūra (valymas/barstymas) ir eismo poveikis
    rem = class_prio * (1 + 0.5 * np.minimum(tr_f[t], 2.0)) * (1 + 0.6 * salt_eff)
    # priežiūra reaguoja su vėlavimu: dažnai dirba tik kai sniego sluoksnis jau susidaręs
    s_ = np.maximum(s_ - (rem * s_ + 0.01), 0)
    i_ = np.maximum(i_ - (0.25 * salt_eff * i_ + 0.4 * class_prio * np.minimum(i_, 0.3)), 0)
    w_ = np.maximum(w_ - 0.0, 0)
    w_ = np.minimum(w_, 2.0); s_ = np.minimum(s_, 6.0); i_ = np.minimum(i_, 1.5)
    st = np.zeros(S, dtype=np.int8)
    st = np.where(w_ > 0.04, 1, st)
    st = np.where(w_ > 0.35, 2, st)
    st = np.where((s_ > 0.05) & (w_ > 0.15), 3, st)
    st = np.where((s_ > 0.20) & (w_ <= 0.15), 4, st)
    st = np.where(i_ > 0.08, 5, st)
    state[t] = st; W_[t] = w_; S_[t] = s_; I_[t] = i_

# trintis
m_dry = 0.80; 
mu = np.select(
    [state == 0, state == 1, state == 2, state == 3, state == 4, state == 5],
    [np.full((T, S), 0.80), np.full((T, S), 0.72), 0.60 - 0.22 * np.clip(W_ / 1.5, 0, 1),
     0.40 - 0.08 * np.clip(S_ / 1.0, 0, 1), 0.30 - 0.06 * np.clip(S_ / 2.0, 0, 1),
     np.where(Ts > -1.0, 0.09, 0.16) + 0.02 * np.clip(I_ / 1.0, 0, 1)])
mu = mu + mu_pave_lat[None, :]
mu = np.where(gravel[None, :] & (state <= 2), mu * 0.72, mu)
mu = np.clip(mu, 0.05, 0.85)

# ------------------------------------------------------------------ etiketė: tikrasis saugus greitis
legal = np.where(((time.month >= 11) | (time.month <= 3))[:, None], lim_winter[None, :], lim_summer[None, :])
night_cap = np.where(lit[None, :], 1e9, 140.0)
sight = np.minimum(np.minimum(sight_geo[None, :], vis), np.where(is_night[:, None], night_cap, 1e9))
v_safe = safe_speed(mu, sight, radius[None, :], gust, traffic, lanes[None, :], legal)
v_safe = np.clip(v_safe + rng.normal(0, 1.0, v_safe.shape), 20, legal)           # 1 km/h etiketės triukšmas

# ------------------------------------------------------------------ stebimi požymiai (matavimai su triukšmu)
rws_noise_T = 0.3 + 0.04 * d_rws
def bern(p, shape): return rng.random(shape) < p
miss = lambda arr, p: np.where(bern(p, arr.shape), np.nan, arr)

rws_air = Ta + rng.normal(0, 1, (T, S)) * rws_noise_T
rws_surf = Ts + rng.normal(0, 1, (T, S)) * (0.4 + 0.05 * d_rws)
rws_rh = np.clip(RH + rng.normal(0, 2.5, (T, S)), 20, 100)
rws_wind = np.clip(wind * np.exp(rng.normal(0, 0.12, (T, S))), 0, None)
rws_gust = np.clip(gust * np.exp(rng.normal(0, 0.12, (T, S))), 0, None)
rws_vis = np.clip(vis * np.exp(rng.normal(0, 0.2, (T, S))), 30, 25000)
rws_pint = np.clip(P_loc + rng.normal(0, 0.1, (T, S)) * (P_loc > 0), 0, None) * bern(0.97, (T, S))
ptype_obs = np.where(bern(0.08, (T, S)) & (ptype > 0), rng.integers(1, 4, (T, S)), ptype)
p_err = 0.07 + 0.012 * d_rws
flip = bern(p_err[None, :], (T, S))
state_obs = np.clip(state + np.where(flip, rng.choice([-1, 1], (T, S)), 0), 0, 5).astype(float)
state_obs = np.where(bern(0.03, (T, S)), np.nan, state_obs)
traffic_obs = np.clip(traffic * np.exp(rng.normal(0, 0.05, (T, S))), 0, None)

met_t = Ta_reg[:, reg_idx] + rng.normal(0, 0.4, (T, S))
met_p = np.clip(P_reg[:, reg_idx] + rng.normal(0, 0.08, (T, S)) * (P_reg[:, reg_idx] > 0), 0, None)
met_wind = wind_reg[:, reg_idx] * np.exp(rng.normal(0, 0.08, (T, S)))
met_gust = gust_reg[:, reg_idx] * np.exp(rng.normal(0, 0.08, (T, S)))
met_rh = np.clip(RH_reg[:, reg_idx] + rng.normal(0, 2, (T, S)), 20, 100)
met_pres = pres_reg[:, reg_idx] + rng.normal(0, 0.3, (T, S))

n_pws = rng.poisson(lam_pws, S)
n_pws_t = np.repeat(n_pws[None, :], T, 0)
pws_ok = (n_pws_t > 0) & bern(0.88, (T, S))
pws_t = np.where(pws_ok, Ta + 0.6 * sun + rng.normal(0, 1, (T, S)) * 0.9 / np.sqrt(np.maximum(n_pws_t, 1)), np.nan)
pws_rh = np.where(pws_ok, np.clip(RH + rng.normal(0, 4, (T, S)), 20, 100), np.nan)
pws_p = np.where(pws_ok, np.clip(P_loc * np.where(ptype == 3, 0.35, 1.0) + rng.normal(0, 0.05, (T, S)) * (P_loc > 0), 0, None), np.nan)

def shift(a, k):  # reikšmė t+k (prognozė žino ateitį su paklaida)
    out = np.empty_like(a); out[:-k] = a[k:]; out[-k:] = a[-k:]; return out
fc_t3 = shift(Ta_reg, 3)[:, reg_idx] + rng.normal(0, 1.1, (T, S))
fc_t6 = shift(Ta_reg, 6)[:, reg_idx] + rng.normal(0, 1.6, (T, S))
P_s = pd.DataFrame(P_reg).rolling(3, center=True, min_periods=1).mean().values
fc_p3 = np.clip(shift(P_s, 3)[:, reg_idx] * np.exp(rng.normal(0, 0.5, (T, S))) + bern(0.04, (T, S)) * rng.gamma(1, 0.5, (T, S)), 0, None)
fc_p6 = np.clip(shift(P_s, 6)[:, reg_idx] * np.exp(rng.normal(0, 0.7, (T, S))) + bern(0.06, (T, S)) * rng.gamma(1, 0.5, (T, S)), 0, None)
fc_w3 = shift(wind_reg, 3)[:, reg_idx] * np.exp(rng.normal(0, 0.25, (T, S)))

# ------------------------------------------------------------------ surinkimas į lentelę
def flat(a): return np.asarray(a, dtype=np.float32).reshape(-1)           # (T,S) -> eilutės: laikas didėja lėčiau, ruožas greičiau
tt = np.repeat(np.arange(T), S); ss = np.tile(np.arange(S), T)
df = pd.DataFrame(dict(
    time=time[tt], segment=ss.astype(np.int16), year_idx=year_idx[tt].astype(np.int8),
    legal_limit=flat(legal), v_safe=flat(v_safe), mu_true=flat(mu), state_true=state.reshape(-1),
    met_t=flat(met_t), met_p=flat(met_p), met_wind=flat(met_wind), met_gust=flat(met_gust), met_rh=flat(met_rh), met_pres=flat(met_pres),
    rws_air=flat(rws_air), rws_surf=flat(rws_surf), rws_rh=flat(rws_rh), rws_wind=flat(rws_wind), rws_gust=flat(rws_gust),
    rws_vis=flat(rws_vis), rws_pint=flat(rws_pint), rws_ptype=ptype_obs.reshape(-1).astype(np.float32), rws_state=flat(state_obs),
    traffic=flat(traffic_obs),
    pws_t=flat(pws_t), pws_rh=flat(pws_rh), pws_p=flat(pws_p),
    fc_t3=flat(fc_t3), fc_t6=flat(fc_t6), fc_p3=flat(fc_p3), fc_p6=flat(fc_p6), fc_w3=flat(fc_w3),
    night=np.repeat(is_night, S).astype(np.int8), sin_elev=np.repeat(sin_elev, S).astype(np.float32),
))
df = df.merge(segs[["segment", "road_class", "region", "lanes", "sight_geo", "curve_radius", "aadt", "hv_share", "pave_age", "lit", "bridge", "gravel", "d_rws_km", "d_met_km"]], on="segment", how="left")
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data"), exist_ok=True)
out = os.path.join(os.path.dirname(__file__), "..", "data")
segs.to_csv(os.path.join(out, "segments.csv"), index=False)
df.to_pickle(os.path.join(out, "dataset.pkl"))
df[(df.time >= "2026-01-01") & (df.time < "2026-01-04")].to_csv(os.path.join(out, "sample_3days.csv"), index=False)
print(df.shape, df.memory_usage().sum() / 1e6, "MB")

# ------------------------------------------------------------------ sanity: simuliuoto vs. tikro (Kažys 2022) dangos pasiskirstymas
share = np.bincount(state.reshape(-1), minlength=6) / state.size * 100
print("Simuliuota dangos būsenų dalis % (visi ruožai):", dict(zip(STATES, share.round(2))))
for c in range(5):
    m = road_class == c
    sh = np.bincount(state[:, m].reshape(-1), minlength=6) / state[:, m].size * 100
    print(class_names[c], sh.round(2))
print("Tikra (Kažys, E85) %:", dict(zip(STATES, KAZYS_TIME_SHARE)))
print("v_safe vidurkis pagal būseną:", [round(float(v_safe[state == k].mean()), 1) for k in range(6)])
print("Kritulių valandų dalis %:", ((P_reg > 0).mean() * 100).round(1), " metinė suma mm:", (P_reg.sum(0) / 3).round(0))
print("Vid. metų temp reg:", Ta_reg.mean(0).round(2), " žiemos (XII-II) vid. pagal m.:",
      [round(float(Ta_reg[(year_idx == y) & np.isin(time.month, [12, 1, 2])].mean()), 2) for y in range(3)])
