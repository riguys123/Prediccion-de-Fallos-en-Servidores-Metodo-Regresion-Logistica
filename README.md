# 🖥️ Predicción de Fallos en Servidores — Regresión Logística

Proyecto desarrollado para la asignatura **Métodos Numéricos (MAT205)** —
Universidad San Francisco Xavier de Chuquisaca.

---

## 📋 Descripción

Sistema de predicción que utiliza **Regresión Logística** para anticipar
fallos inminentes en servidores a partir de métricas de monitoreo en
tiempo real (variables continuas).

| Variable de entrada | Tipo | Descripción |
|---|---|---|
| `temperatura_cpu` | Continua | Temperatura del procesador en °C |
| `uso_cpu` | Continua | Porcentaje de uso del CPU |
| `uso_ram` | Continua | Porcentaje de uso de RAM |
| `paquetes_perdidos` | Continua | % de paquetes de red perdidos |
| **`fallo`** (salida) | **Categórica** | 0 = Normal, 1 = Fallo inminente |

---

## 📁 Estructura del proyecto

```
servidor_prediccion/
├── README.md               ← Este archivo
├── requirements.txt        ← Dependencias Python
├── generar_dataset.py      ← Genera el dataset simulado (.csv)
└── modelo.py               ← Entrena, evalúa y realiza inferencias
```

---

## ⚙️ Requisitos

- Python 3.8 o superior
- pip

---

## 🚀 Instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/servidor-prediccion.git
cd servidor-prediccion
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Generar el dataset

```bash
python generar_dataset.py
```

Esto crea el archivo `dataset_servidores.csv` con 500 registros simulados.

### 4. Entrenar el modelo y ver resultados

```bash
python modelo.py
```

Esto imprime en consola:
- Coeficientes β del modelo
- Reporte de clasificación (precisión, recall, F1)
- AUC-ROC
- Inferencia con 3 servidores de ejemplo

Y genera el archivo `resultados.png` con:
- Matriz de confusión
- Curva ROC

---

## 📊 Ejemplo de salida esperada

```
[5] Resultados en conjunto de prueba:
              precision    recall  f1-score

  Normal (0)       0.97      0.99      0.98
   Fallo (1)       0.97      0.93      0.95

    AUC-ROC: 0.9950

[7] Inferencia con nuevos servidores:
  Servidor 1: Temp=58.0°C | CPU=40.0% | RAM=45.0% | Paquetes=1.5%
    Probabilidad de fallo : 0.32%
    Predicción            : ✅ Normal

  Servidor 2: Temp=91.0°C | CPU=95.0% | RAM=92.0% | Paquetes=25.0%
    Probabilidad de fallo : 99.98%
    Predicción            : ⚠️  FALLO INMINENTE
```

---

## 📐 Base matemática

El modelo aplica la función sigmoide sobre una combinación lineal de las variables:

```
z = β₀ + β₁·temperatura + β₂·uso_cpu + β₃·uso_ram + β₄·paquetes_perdidos

P(fallo) = 1 / (1 + e^(-z))

Si P(fallo) ≥ 0.5  →  Fallo inminente
Si P(fallo) < 0.5  →  Normal
```

Los parámetros β se optimizan minimizando la función de pérdida Log Loss
mediante gradiente descendente.

---

## 📚 Dependencias

Ver `requirements.txt`:

```
numpy
pandas
scikit-learn
matplotlib
```
