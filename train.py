"""LSTM para predecir la temperatura 1 hora adelante — Solemne 3, Microprocesadores.

Datos (carpeta datos/):
  - quinta_normal_oct2026_horaria.csv : temperatura MEDIDA cada 1 h, estación Quinta Normal
    (DMC 330020), octubre 2026. Es el conjunto de entrenamiento, validación y prueba.
  - arduino_20261125.csv              : nuestra estación (Nano 33 BLE Sense, HTS221), 25-11-2026,
    cada 5 s. Se promedia por hora y se usa como prueba extra con datos del proyecto.
  - open_meteo_santiago_oct2026_15min.csv : modelo Open-Meteo (no medido). Solo como referencia.

Idea (igual que la ventana de 3 valores del train.py original):
  entrada = las últimas 3 horas de temperatura + la hora del día
  salida  = cuánto cambiará la temperatura en la próxima hora (delta)
Se predice el CAMBIO y no el valor, así el modelo sirve aunque el Arduino mida otro nivel
de temperatura (sala a ~31 °C contra exterior a ~15 °C).

Compuerta: el modelo solo se aprueba si su MAE en validación es MENOR que el de la
persistencia (suponer que la temperatura no cambia en la próxima hora).
"""
import json
import random as rd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Sequential

VENTANA = 3          # horas de historia
HORIZONTE = 1        # horas hacia adelante
SEED = 12122008
rd.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ------------------------------------------------------------------ datos
def serie_quinta_normal():
    d = pd.read_csv("datos/quinta_normal_oct2026_horaria.csv", parse_dates=["fecha_hora"])
    s = d.set_index("fecha_hora")["temp_c"].asfreq("h")
    print(f"Quinta Normal: {len(s)} horas, {s.isna().sum()} vacía(s) rellenada(s) por interpolación")
    return s.interpolate(limit=2)


def serie_arduino():
    d = pd.read_csv("datos/arduino_20261125.csv", parse_dates=["fecha_hora"])
    s = d.set_index("fecha_hora")["temperatura_C"]
    s = s[(s > -40) & (s < 120)]
    h = s.resample("h").mean()
    h = h[s.resample("h").count() >= 360]       # solo horas con al menos 30 min de datos
    print(f"Arduino: {len(s)} lecturas -> {len(h)} horas completas")
    return h.asfreq("h")


def armar(s):
    """X: (n, VENTANA, 3) = [temperatura relativa a la actual, sin(hora), cos(hora)]
       y: delta de temperatura a HORIZONTE horas.  t: hora de cada muestra.  actual: T(t)."""
    v, idx = s.to_numpy(), s.index
    X, y, t, actual = [], [], [], []
    for i in range(VENTANA - 1, len(v) - HORIZONTE):
        tramo = v[i - VENTANA + 1:i + 1]
        fut = v[i + HORIZONTE]
        if np.isnan(tramo).any() or np.isnan(fut):
            continue
        hrs = np.array([idx[j].hour + idx[j].minute / 60 for j in range(i - VENTANA + 1, i + 1)])
        X.append(np.stack([tramo - tramo[-1],
                           np.sin(2 * np.pi * hrs / 24),
                           np.cos(2 * np.pi * hrs / 24)], axis=1))
        y.append(fut - tramo[-1])
        t.append(idx[i])
        actual.append(tramo[-1])
    return np.array(X), np.array(y), pd.DatetimeIndex(t), np.array(actual)


def particion(n, embargo=VENTANA + HORIZONTE):
    """70/15/15 en orden de tiempo, con un embargo entre conjuntos (no al azar:
    en series de tiempo, mezclar deja que el modelo 'vea el futuro')."""
    a, b = int(n * 0.70), int(n * 0.85)
    return slice(0, a - embargo), slice(a, b - embargo), slice(b, n)


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


# ------------------------------------------------------------------ entrenamiento
X, y, t, actual = armar(serie_quinta_normal())
tr, va, te = particion(len(y))
print(f"Muestras: entrenamiento {len(y[tr])}, validación {len(y[va])}, prueba {len(y[te])}")

model = Sequential([
    Input(name="serie", shape=(VENTANA, 3)),
    LSTM(32),
    Dense(1, activation="linear"),
])
model.compile(optimizer=tf.keras.optimizers.Adam(0.003), loss="mse", metrics=["mae"])
model.summary()
hist = model.fit(X[tr], y[tr], validation_data=(X[va], y[va]), epochs=150, batch_size=32, verbose=2,
                 callbacks=[EarlyStopping(monitor="val_mae", patience=15, restore_best_weights=True)])


def evaluar(Xs, ys):
    pred = model.predict(Xs, verbose=0).ravel()
    return mae(pred, ys), mae(0, ys), pred          # (LSTM, persistencia, predicciones)


res = {}
for nombre, sl in (("validacion", va), ("prueba", te)):
    m, p, _ = evaluar(X[sl], y[sl])
    res[nombre] = {"mae_lstm": round(m, 3), "mae_persistencia": round(p, 3), "n": int(len(y[sl]))}

Xa, ya, ta, act_a = armar(serie_arduino())
if len(ya):
    m, p, pred_a = evaluar(Xa, ya)
    res["arduino"] = {"mae_lstm": round(m, 3), "mae_persistencia": round(p, 3), "n": int(len(ya))}

aprobado = res["validacion"]["mae_lstm"] < res["validacion"]["mae_persistencia"]
res["compuerta"] = "APROBADO" if aprobado else "RECHAZADO"
res["epocas"] = len(hist.history["loss"])

print("\n=========== RESULTADOS (MAE en °C, predicción a 1 hora) ===========")
for k in ("validacion", "prueba", "arduino"):
    if k in res:
        r = res[k]
        print(f"{k:11s} LSTM {r['mae_lstm']:.3f} | persistencia {r['mae_persistencia']:.3f} | n={r['n']}")
print(f"COMPUERTA: {res['compuerta']} "
      f"({'el modelo le gana a la persistencia' if aprobado else 'el modelo NO le gana a la persistencia'})")

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
if aprobado:
    model.save("modelo_lstm.keras")

# ------------------------------------------------------------------ gráficos
_, _, pred_te = evaluar(X[te], y[te])
fig, ax = plt.subplots(2, 2, figsize=(13, 8))

ax[0, 0].plot(t[te], actual[te] + y[te], "k", label="Real (Quinta Normal)")
ax[0, 0].plot(t[te], actual[te] + pred_te, "r", alpha=.8, label="LSTM")
ax[0, 0].plot(t[te], actual[te], "b--", alpha=.5, label="Persistencia")
ax[0, 0].set_title("Prueba: temperatura dentro de 1 hora")
ax[0, 0].set_ylabel("°C")
ax[0, 0].legend()
ax[0, 0].tick_params(axis="x", rotation=30)

ax[0, 1].plot(hist.history["mae"], label="entrenamiento")
ax[0, 1].plot(hist.history["val_mae"], label="validación")
ax[0, 1].set_title("Aprendizaje (MAE por época)")
ax[0, 1].set_xlabel("época")
ax[0, 1].legend()

conj = [k for k in ("validacion", "prueba", "arduino") if k in res]
xx = np.arange(len(conj))
ax[1, 0].bar(xx - .2, [res[k]["mae_persistencia"] for k in conj], .4, label="Persistencia")
ax[1, 0].bar(xx + .2, [res[k]["mae_lstm"] for k in conj], .4, label="LSTM")
ax[1, 0].set_xticks(xx, conj)
ax[1, 0].set_ylabel("MAE (°C)")
ax[1, 0].set_title(f"Compuerta: {res['compuerta']}")
ax[1, 0].legend()

if len(ya):
    ax[1, 1].plot(ta, act_a + ya, "ko-", label="Real (Arduino, promedio por hora)")
    ax[1, 1].plot(ta, act_a + pred_a, "rs--", label="LSTM")
    ax[1, 1].set_title("Arduino 25-11-2026 (sala)")
    ax[1, 1].legend()
    ax[1, 1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("predicciones.png", dpi=120)
print("Gráfico guardado en predicciones.png")
