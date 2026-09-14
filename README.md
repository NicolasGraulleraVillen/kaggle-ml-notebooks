# Notebooks de Machine Learning (Kaggle)

Repositorio con **tres notebooks** de competiciones Kaggle, para revisión de reclutadores o hiring managers. Cada archivo incluye el flujo completo: exploración, preparación de datos, modelado y métricas.

Autor: [Nicolás Graullera Villén](https://github.com/NicolasGraulleraVillen)

---

## Cómo abrirlos

Los `.ipynb` se pueden ver en GitHub (clic en el archivo) o ejecutar en [Kaggle Notebooks](https://www.kaggle.com/) / Jupyter. Las rutas de datos (`/kaggle/input/...`) corresponden al entorno de Kaggle; en local hay que descargar el dataset de cada competición.

Dependencias habituales: `pandas`, `numpy`, `scikit-learn`, `catboost`, `xgboost`, `optuna`, `matplotlib`, `seaborn`.

---

## 1. Clasificación binaria — Spaceship Titanic

**Archivo:** [`ST_Nicolas_Victor_Graullera_Villen.ipynb`](ST_Nicolas_Victor_Graullera_Villen.ipynb)

**Problema:** predecir si un pasajero fue *transported* tras la colisión de la nave (clasificación binaria).

**Enfoque:**
- Modelo **CatBoostClassifier** (gradient boosting), elegido por su manejo nativo de categóricas y nulos.
- Ingeniería de variables (incluido apellido como señal de grupo).
- Diagnóstico y corrección de tipos para que CatBoost acepte nulos en features categóricas.

**Métrica:** accuracy de validación aproximadamente **0,80–0,81**.

---

## 2. Clasificación — diagnóstico de diabetes

**Archivo:** [`Nicolas_graullera_diabetes.ipynb`](Nicolas_graullera_diabetes.ipynb)

**Problema:** [Playground Series S5E12](https://www.kaggle.com/competitions/playground-series-s5e12) — predecir `diagnosed_diabetes`.

**Datos:** ~**700.000** observaciones y **24** features (enteras, categóricas y continuas), sin nulos.

**Enfoque:**
- Análisis de asimetría y curtosis; sin transformaciones gaussianas (modelos de árboles).
- **XGBoost** y **CatBoost**, validación **estratificada** (K-fold) y ajuste de hiperparámetros con **Optuna**.
- Ingeniería de variables de riesgo metabólico (p. ej. combinación de obesidad central y BMI).

**Métricas:** OOF AUC **0,726**; mejor *private score* **0,693**.

---

## 3. Regresión — peso al nacer (DBWT)

**Archivo:** [`BW_Nicolas_Victor_Graullera_Villen.ipynb`](BW_Nicolas_Victor_Graullera_Villen.ipynb)

**Problema:** estimar el peso al nacer (`DBWT`) a partir de datos del Sistema Nacional de Estadísticas Vitales.

**Enfoque:**
- **CatBoostRegressor**, con tratamiento nativo de nulos y categóricas (mejor que la imputación manual en RMSE).
- Filtro de outliers extremos (`DBWT` &lt; 1000 g).
- *Early stopping* sobre conjunto de validación.

**Métrica:** RMSE aproximadamente **451–480 g**.

---

## Qué se puede evaluar aquí

No es un portfolio de producto; es evidencia de trabajo analítico:

- plantear un problema (clasificación vs regresión),
- limpiar y validar datos,
- elegir un modelo con criterio (nulos, categóricas, escala),
- reportar métricas honestas (validación / OOF / private).
