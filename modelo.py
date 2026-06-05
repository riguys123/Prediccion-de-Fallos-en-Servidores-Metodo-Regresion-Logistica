"""
modelo.py
=========
Entrena y evalúa un modelo de Regresión Logística para predecir
fallos inminentes en servidores a partir de métricas continuas.

Uso:
    python modelo.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # backend sin interfaz gráfica

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, roc_auc_score)

# ─────────────────────────────────────────────
# 1. CARGAR DATOS
# ─────────────────────────────────────────────
print("=" * 55)
print("  PREDICCIÓN DE FALLOS EN SERVIDORES")
print("  Método: Regresión Logística")
print("=" * 55)

df = pd.read_csv("dataset_servidores.csv")
print(f"\n[1] Dataset cargado: {len(df)} registros")
print(f"    Normales: {(df['fallo']==0).sum()} | Fallos: {(df['fallo']==1).sum()}")

X = df[["temperatura_cpu", "uso_cpu", "uso_ram", "paquetes_perdidos"]]
y = df["fallo"]

# ─────────────────────────────────────────────
# 2. DIVISIÓN ENTRENAMIENTO / PRUEBA
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[2] División: {len(X_train)} entrenamiento | {len(X_test)} prueba")

# ─────────────────────────────────────────────
# 3. ESCALADO (estandarización)
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print("\n[3] Variables estandarizadas (media=0, desv=1)")

# ─────────────────────────────────────────────
# 4. ENTRENAMIENTO
# ─────────────────────────────────────────────
modelo = LogisticRegression(max_iter=1000, random_state=42)
modelo.fit(X_train_sc, y_train)
print("\n[4] Modelo entrenado exitosamente")
print("\n    Coeficientes (β):")
for nombre, coef in zip(X.columns, modelo.coef_[0]):
    print(f"      {nombre:<22}: {coef:+.4f}")
print(f"      {'intercepto (β₀)':<22}: {modelo.intercept_[0]:+.4f}")

# ─────────────────────────────────────────────
# 5. EVALUACIÓN
# ─────────────────────────────────────────────
y_pred  = modelo.predict(X_test_sc)
y_proba = modelo.predict_proba(X_test_sc)[:, 1]
auc     = roc_auc_score(y_test, y_proba)

print("\n[5] Resultados en conjunto de prueba:")
print(classification_report(y_test, y_pred,
      target_names=["Normal (0)", "Fallo (1)"]))
print(f"    AUC-ROC: {auc:.4f}")

# ─────────────────────────────────────────────
# 6. GRÁFICAS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Regresión Logística — Predicción de Fallos en Servidores",
             fontsize=13, fontweight="bold")

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Normal", "Fallo"])
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Matriz de Confusión")

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[1].plot(fpr, tpr, color="steelblue", lw=2,
             label=f"AUC = {auc:.4f}")
axes[1].plot([0,1],[0,1], "k--", lw=1)
axes[1].set_xlabel("Tasa de Falsos Positivos")
axes[1].set_ylabel("Tasa de Verdaderos Positivos")
axes[1].set_title("Curva ROC")
axes[1].legend()

plt.tight_layout()
plt.savefig("resultados.png", dpi=150, bbox_inches="tight")
print("\n[6] Gráficas guardadas en 'resultados.png'")

# ─────────────────────────────────────────────
# 7. INFERENCIA CON NUEVOS DATOS
# ─────────────────────────────────────────────
print("\n[7] Inferencia con nuevos servidores:")
print("-" * 55)

nuevos = pd.DataFrame({
    "temperatura_cpu":   [58.0, 91.0, 70.0],
    "uso_cpu":           [40.0, 95.0, 65.0],
    "uso_ram":           [45.0, 92.0, 60.0],
    "paquetes_perdidos": [ 1.5, 25.0,  8.0],
})

nuevos_sc = scaler.transform(nuevos)
predicciones = modelo.predict(nuevos_sc)
probabilidades = modelo.predict_proba(nuevos_sc)[:, 1]

etiquetas = {0: " Normal", 1: "  FALLO INMINENTE"}
for i, (pred, prob) in enumerate(zip(predicciones, probabilidades)):
    print(f"\n  Servidor {i+1}:")
    print(f"    Temp={nuevos.iloc[i]['temperatura_cpu']}°C | "
          f"CPU={nuevos.iloc[i]['uso_cpu']}% | "
          f"RAM={nuevos.iloc[i]['uso_ram']}% | "
          f"Paquetes perdidos={nuevos.iloc[i]['paquetes_perdidos']}%")
    print(f"    Probabilidad de fallo : {prob:.2%}")
    print(f"    Predicción            : {etiquetas[pred]}")

print("\n" + "=" * 55)
print("  Ejecución completada.")
print("=" * 55)
