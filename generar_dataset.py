"""
generar_dataset.py
==================
Genera un dataset simulado de métricas de servidores.

Variables de entrada (continuas):
  - temperatura_cpu : temperatura del procesador en °C
  - uso_cpu         : porcentaje de uso del CPU (0-100%)
  - uso_ram         : porcentaje de uso de RAM (0-100%)
  - paquetes_perdidos: porcentaje de paquetes de red perdidos (0-100%)

Variable de salida (categórica binaria):
  - fallo           : 0 = Normal, 1 = Fallo inminente
"""

import numpy as np
import pandas as pd

# Semilla para reproducibilidad
np.random.seed(42)
N = 500  # número de registros

# --- Servidores NORMALES (350 registros) ---
n_normal = 350
temperatura_normal   = np.random.normal(loc=55,  scale=8,  size=n_normal).clip(30, 75)
uso_cpu_normal       = np.random.normal(loc=45,  scale=15, size=n_normal).clip(5,  70)
uso_ram_normal       = np.random.normal(loc=50,  scale=15, size=n_normal).clip(10, 75)
paquetes_normal      = np.random.normal(loc=2,   scale=2,  size=n_normal).clip(0,  10)
fallos_normal        = np.zeros(n_normal, dtype=int)

# --- Servidores con FALLO INMINENTE (150 registros) ---
n_fallo = 150
temperatura_fallo    = np.random.normal(loc=82,  scale=7,  size=n_fallo).clip(70, 100)
uso_cpu_fallo        = np.random.normal(loc=88,  scale=8,  size=n_fallo).clip(70, 100)
uso_ram_fallo        = np.random.normal(loc=85,  scale=8,  size=n_fallo).clip(70, 100)
paquetes_fallo       = np.random.normal(loc=18,  scale=7,  size=n_fallo).clip(8,  50)
fallos_fallo         = np.ones(n_fallo, dtype=int)

# --- Combinar y mezclar ---
temperatura    = np.concatenate([temperatura_normal,  temperatura_fallo])
uso_cpu        = np.concatenate([uso_cpu_normal,       uso_cpu_fallo])
uso_ram        = np.concatenate([uso_ram_normal,       uso_ram_fallo])
paquetes       = np.concatenate([paquetes_normal,      paquetes_fallo])
fallo          = np.concatenate([fallos_normal,        fallos_fallo])

df = pd.DataFrame({
    "temperatura_cpu":    np.round(temperatura, 1),
    "uso_cpu":            np.round(uso_cpu, 1),
    "uso_ram":            np.round(uso_ram, 1),
    "paquetes_perdidos":  np.round(paquetes, 1),
    "fallo":              fallo
})

# Mezclar filas aleatoriamente
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Guardar
df.to_csv("dataset_servidores.csv", index=False)
print(f"Dataset generado: {len(df)} registros")
print(f"  Normales : {(df['fallo']==0).sum()}")
print(f"  Fallos   : {(df['fallo']==1).sum()}")
print("\nPrimeras 5 filas:")
print(df.head())
