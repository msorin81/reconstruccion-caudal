# Reconstrucción de caudal diario con aprendizaje profundo

**Autor:** Oscar Mauricio Montaño Bolaños

Repositorio de soporte del trabajo de maestría sobre reconstrucción de series de caudal diario en tres estaciones del suroccidente colombiano: **Buga** (Valle del Cauca), **Santander de Quilichao** y **Totoró** (Cauca).

Contiene los datos de entrada, los cuadernos de Python con todo el proceso (comparación de modelos de árboles, modelos de aprendizaje profundo, validación y reconstrucción), las series finales reconstruidas, los modelos entrenados y una aplicación web para aplicar los modelos a nuevos datos.

## Series reconstruidas por estación

Los archivos de [`datos/reconstruidos/`](datos/reconstruidos/) contienen la serie diaria final de cada estación. La columna `Fuente` indica el origen de cada valor.

| Estación | Periodo | Días | Observados | Reconstruidos | Sin reconstruir (vacío > 100 días) | Archivo |
|---|---|---:|---:|---:|---:|---|
| Buga | 2010-01-01 a 2025-12-31 | 5 833 | 3 100 | 115 | 2 618 | [Buga_caudal_reconstruido.csv](datos/reconstruidos/Buga_caudal_reconstruido.csv) |
| Santander de Quilichao | 2010-01-01 a 2021-12-31 | 4 364 | 2 724 | 230 | 1 410 | [Santander_caudal_reconstruido.csv](datos/reconstruidos/Santander_caudal_reconstruido.csv) |
| Totoró | 2005-11-13 a 2015-12-31 | 3 510 | 3 077 | 68 | 365 | [Totoro_caudal_reconstruido.csv](datos/reconstruidos/Totoro_caudal_reconstruido.csv) |

Solo se reconstruyen los vacíos de 100 días consecutivos o menos. Los vacíos más largos se dejan sin valor porque el modelo no tiene suficiente información observada alrededor para estimarlos con confiabilidad.

La descripción de cada columna está en el [diccionario de datos](datos/diccionario_datos.md).

## Estructura

```
├── app_reconstruccion.py     Aplicación Streamlit para reconstruir caudal con los modelos entrenados
├── modelos/                  Modelos CNN-BiLSTM finales, uno por estación (.keras)
├── datos/
│   ├── entrada/              Variables climáticas (Cenicaña) y caudal observado (IDEAM) por estación
│   ├── reconstruidos/        Series finales de caudal por estación
│   └── diccionario_datos.md
├── notebooks/
│   ├── 01_ml_comparacion_*.ipynb       Comparación de modelos de árboles (XGBoost, LightGBM, CatBoost, etc.)
│   └── 02_dl_reconstruccion_*.ipynb    Modelos LSTM, BiLSTM y CNN-BiLSTM: entrenamiento, validación y reconstrucción
└── resultados/
    ├── hiperparametros/      Hiperparámetros óptimos encontrados con Hyperband
    ├── ablacion/             Estudio de ablación de variables (XGBoost y CNN-BiLSTM)
    └── tablas/               Tablas de experimentos, resumen año-mes y vacíos por estación
```

Los cuadernos se publican con sus salidas guardadas, de modo que las tablas y figuras se pueden consultar directamente en GitHub sin ejecutar nada.

## Modelos de aprendizaje profundo

Las tres arquitecturas reciben una ventana de 23 días (15 previos, el día a estimar y 7 posteriores) con 10 variables: caudal pasado en escala logarítmica, precipitación, temperatura media, humedad relativa media, evaporación, radiación solar y cuatro variables cíclicas de estacionalidad.

| Modelo | Arquitectura | Parámetros |
|---|---|---:|
| LSTM_v2 | LSTM(64) → Dropout → LSTM(32) → Dropout → Dense(8) → Dense(1) | 31 889 |
| BiLSTM_v2 | BiLSTM(64) → Dropout → BiLSTM(32) → Dropout → Dense(32) → Dense(1) | 81 729 |
| CNN_BiLSTM_v2 | Conv1D(32, k=7) → Dropout → BiLSTM(64) → Dropout → BiLSTM(32) → Dropout → Dense(16) → Dense(1) | 94 209 |

Hiperparámetros elegidos con Hyperband ([`mejores_hp_dl.json`](resultados/hiperparametros/mejores_hp_dl.json)): dropout 0,1, Adam con tasa de aprendizaje 0,001, pérdida MSE. El modelo usado para la reconstrucción final es **CNN_BiLSTM_v2**.

## Cómo ejecutar

Requiere Python 3.11 o superior.

**Aplicación:**

```bash
pip install -r requirements.txt
streamlit run app_reconstruccion.py
```

**Cuadernos:**

```bash
pip install -r requirements-notebooks.txt
cd notebooks
jupyter notebook
```

Los cuadernos leen los datos desde `../datos/entrada/` y los modelos desde `../modelos/`, por lo que deben abrirse desde la carpeta `notebooks/`. Los cuadernos `02_dl_*` cargan el modelo final guardado; para reentrenarlo, cambie `FORCE_RETRAIN = True`.

## Fuentes de datos

- **Caudal diario:** Instituto de Hidrología, Meteorología y Estudios Ambientales (IDEAM).
- **Variables climáticas:** Centro de Investigación de la Caña de Azúcar de Colombia (Cenicaña), red de estaciones meteorológicas automáticas.

## Licencia

- Código: [MIT](LICENSE).
- Datos y series reconstruidas: [CC BY 4.0](LICENSE-DATA.md), citando además a las fuentes originales (IDEAM y Cenicaña).
