# Išmanioji greičio rekomendavimo sistema pagal kelių sąlygas

Patariamoji sistema: pagal orų (meteo.lt, Windy), kelių oro stotelių ir eismo (eismoinfo.lt) bei privačių stočių (Wunderground, Netatmo) duomenis kas valandą apskaičiuoja saugų greitį kiekvienam kelio ruožui.

**SVARBU:** kūrimo aplinkoje nebuvo interneto, todėl duomenys yra SINTETINIAI (fiziškai pagrįsti, kalibruoti pagal Kažys ir Jančiauskaitė 2022). Rezultatai rodo metodų palyginimą, bet neįrodo tikslumo realiuose keliuose. `code/collectors.py` yra neišbandytas.

## Paleidimas
```
pip install -r requirements.txt
python code/generate_data.py   # data/dataset.pkl (~170 MB), segments.csv, sample_3days.csv
python code/model.py           # eksperimentai 1-5, results/*.csv
python code/extra.py           # hibridas + 4 klasių metrikos (6-7)
python code/safeguard.py       # apsauginė taisyklė (8)
python code/make_figures.py    # figures/*.png
python code/demo.py --segment 12 --time "2026-01-02 07:00"
```
Visa grandinė ~6-7 min. 1 CPU branduolyje, ~4 GB RAM. Dockerfile: `docker build -t speedrec . && docker run speedrec`.


