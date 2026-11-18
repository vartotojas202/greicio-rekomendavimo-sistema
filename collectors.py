"""
REALIŲ duomenų kolektoriai (NEPATIKRINTI šioje aplinkoje – kūrimo sandėlyje nebuvo interneto).
Naudojami tik dokumentuoti viešieji galiniai taškai; raktus/slaptažodžius imkite iš aplinkos kintamųjų.
Išvestis visur – pandas.DataFrame su stulpeliais, kuriuos naudoja features.py (žr. README, lentelę „Schema“).
"""
import os, requests, pandas as pd

def meteo_lt_latest(station_code: str) -> pd.DataFrame:
    """meteo.lt API (CC BY-SA 4.0; limitas 180 užklausų/min. vienam IP): paskutinės 24 h stoties stebėjimai.
    Dok.: https://api.meteo.lt/ ; kelias: /v1/stations/{station-code}/observations/latest"""
    r = requests.get(f"https://api.meteo.lt/v1/stations/{station_code}/observations/latest", timeout=30); r.raise_for_status()
    j = r.json(); return pd.DataFrame(j.get("observations", []))

def meteo_lt_forecast(place: str) -> pd.DataFrame:
    """7 dienų prognozė (Harmonie/ECMWF): /v1/places/{place}/forecasts/long-term"""
    r = requests.get(f"https://api.meteo.lt/v1/places/{place}/forecasts/long-term", timeout=30); r.raise_for_status()
    return pd.DataFrame(r.json().get("forecastTimestamps", []))

def eismoinfo(url_from_open_data_page: str) -> pd.DataFrame:
    """eismoinfo.lt > Atviri duomenys (kelių oro stotelės, eismo intensyvumas, apribojimai). Tikslūs URL ir laukai
    pateikiami puslapyje https://eismoinfo.lt (skiltis „Atviri duomenys“) – juos reikia nukopijuoti į EISMOINFO_URL.
    Atkarpų koordinatės intensyvumo rinkinyje yra LKS94 (EPSG:3346) – reikia konvertuoti į WGS84."""
    r = requests.get(url_from_open_data_page, timeout=30); r.raise_for_status()
    try: return pd.json_normalize(r.json())
    except ValueError: return pd.read_xml(r.text)

def wunderground_pws_current(station_id: str, units="m") -> dict:
    """Weather Underground / The Weather Company PWS (reikia API rakto: WU_API_KEY).
    https://api.weather.com/v2/pws/observations/current?stationId=...&format=json&units=m&apiKey=..."""
    r = requests.get("https://api.weather.com/v2/pws/observations/current",
                     params=dict(stationId=station_id, format="json", units=units, apiKey=os.environ["WU_API_KEY"]), timeout=30); r.raise_for_status()
    return r.json()

def windy_point_forecast(lat, lon, model="iconEu", params=("temp", "precip", "wind", "windGust", "rh")) -> dict:
    """Windy Point Forecast v2 (POST, raktas WINDY_API_KEY). Grąžina TIK naujausią prognozę – istorinių prognozių nėra,
    todėl mokymui reikia pačiam kaupti archyvą. Nemokamas „Testing“ raktas grąžina sumaišytus duomenis (tik kūrimui)."""
    r = requests.post("https://api.windy.com/api/point-forecast/v2",
                      json=dict(lat=lat, lon=lon, model=model, parameters=list(params), levels=["surface"], key=os.environ["WINDY_API_KEY"]), timeout=30)
    r.raise_for_status(); return r.json()

def netatmo_public(lat_ne, lon_ne, lat_sw, lon_sw, access_token: str) -> dict:
    """Netatmo Weather API /getpublicdata (OAuth2; tik stotys, kurių savininkai sutiko rodyti žemėlapyje).
    Siųskite mažus plotus (didelis plotas gali grąžinti 503). Lietaus matuokliai nepakankamai fiksuoja sniegą."""
    r = requests.get("https://api.netatmo.com/api/getpublicdata",
                     params=dict(lat_ne=lat_ne, lon_ne=lon_ne, lat_sw=lat_sw, lon_sw=lon_sw, required_data="temperature,humidity,rain", filter="true"),
                     headers={"Authorization": f"Bearer {access_token}"}, timeout=30); r.raise_for_status(); return r.json()

def open_meteo_era5(lat, lon, start, end) -> pd.DataFrame:
    """Open-Meteo istorinis ERA5 (be rakto; CC BY 4.0; nekomercinis naudojimas). Papildomas šaltinis ilgoms istorinėms eilutėms."""
    r = requests.get("https://archive-api.open-meteo.com/v1/archive",
                     params=dict(latitude=lat, longitude=lon, start_date=start, end_date=end,
                                 hourly="temperature_2m,relative_humidity_2m,precipitation,snowfall,wind_speed_10m,wind_gusts_10m,visibility,surface_pressure",
                                 timezone="Europe/Vilnius"), timeout=60); r.raise_for_status()
    return pd.DataFrame(r.json()["hourly"])
