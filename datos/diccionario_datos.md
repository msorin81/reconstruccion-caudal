# Diccionario de datos

## Datos de entrada (`datos/entrada/*.xlsx`, hoja `Data`)

Archivos Excel originales con variables climáticas diarias y caudal observado por estación.

| Estación | Archivo | Período | Filas |
|----------|---------|---------|------:|
| Buga | `Buga_variables_Cenicana_IDEAM.xlsx` | 2007-01-01 a 2022-12-31 | 5 833 |
| Santander de Quilichao | `Santander_Quilichao_variables_Cenicana_IDEAM.xlsx` | 2011-01-02 a 2022-12-31 | 4 364 |
| Totoró | `Totoro_variables.xlsx` | 2013-05-01 a 2023-01-10 | 3 510 |

| Columna | Descripción |
|---|---|
| `nm_ema` | Nombre de la estación meteorológica automática |
| `fc_obsrvcion`, `Fecha_` | Fecha de la observación |
| `Año`, `Mes`, `Día` | Componentes de la fecha |
| `Faltantes` | Indicador de registro completo (`Ok`) o incompleto |
| `tmp_minnma` / `tmp_minma` | Temperatura mínima diaria (°C) |
| `tmp_mxma` / `tmp_maxma` | Temperatura máxima diaria (°C) |
| `tmp_mdia` | Temperatura media diaria (°C) |
| `hmd_rel_minma` | Humedad relativa mínima diaria (%) |
| `hmd_rel_max` / `hmd_rel_maxma` | Humedad relativa máxima diaria (%) |
| `hmd_rel_mdia` | Humedad relativa media diaria (%) |
| `prcptcion` | Precipitación diaria (mm) |
| `evp_clclada` | Evaporación calculada diaria |
| `rdcion_slar` | Radiación solar diaria: cal/cm²/día en Buga y Santander; MJ/m²/día en Totoró |
| `Caudal (m^3/s)` / `m^3/s` | Caudal medio diario observado (m³/s), IDEAM |

Los nombres de algunas columnas varían entre estaciones (indicados con `/`). Los cuadernos y datasets completos los normalizan.

La radiación solar viene en unidades distintas según la estación. Esto no afecta los modelos porque cada estación tiene su propio modelo y las variables se escalan por separado (MinMax a [0, 1]). Para comparar entre estaciones: 1 cal/cm²/día = 0,04184 MJ/m²/día.

**Fuentes:**
- Variables climáticas: Cenicaña (estaciones automáticas)
- Caudal observado: IDEAM (estaciones limnigráficas)

## Series reconstruidas (`datos/reconstruidos/*_caudal_reconstruido.csv`)

Un archivo por estación, codificado en UTF-8, separado por comas, una fila por día.

| Columna | Tipo | Descripción |
|---|---|---|
| `Fecha` | fecha (YYYY-MM-DD) | Día de la observación |
| `Caudal_m3s` | decimal | Caudal medio diario en m³/s. Vacío cuando el día no se pudo reconstruir |
| `Fuente` | texto | Origen del valor (ver tabla siguiente) |

Valores posibles de `Fuente`:

| Valor | Significado |
|---|---|
| `Observado` | Caudal medido por el IDEAM, sin modificar |
| `Reconstruido (CNN_BiLSTM_v2)` | Caudal estimado por el modelo CNN-BiLSTM en un vacío de 100 días o menos |
| `No reconstruido (gap >100d)` | Día dentro de un vacío de más de 100 días consecutivos; `Caudal_m3s` queda vacío |

## Datasets completos (`datos/reconstruidos/*_dataset_completo.csv`)

Integran las variables climáticas normalizadas con la serie de caudal reconstruida. Diseñados para uso directo en modelado hidrológico (HEC-HMS, SWAT, etc.).

| Columna | Tipo | Unidad | Descripción |
|---------|------|--------|-------------|
| `Fecha` | date | — | Fecha de la observación (YYYY-MM-DD), índice |
| `tmp_min_C` | float | °C | Temperatura mínima diaria |
| `tmp_max_C` | float | °C | Temperatura máxima diaria |
| `tmp_media_C` | float | °C | Temperatura media diaria |
| `hmd_rel_min_pct` | float | % | Humedad relativa mínima |
| `hmd_rel_max_pct` | float | % | Humedad relativa máxima |
| `hmd_rel_media_pct` | float | % | Humedad relativa media |
| `precipitacion_mm` | float | mm | Precipitación diaria acumulada |
| `evaporacion_mm` | float | mm | Evaporación calculada diaria |
| `radiacion_solar` | float | var. | Radiación solar diaria (unidades varían por estación) |
| `Caudal_m3s` | float | m³/s | Caudal diario. `NaN` si no fue posible reconstruir |
| `Fuente` | string | — | Origen del valor de caudal |

### Resumen por estación

| Estación | Filas | Caudal disponible | Clima completo | Observados | Reconstruidos | No reconstruidos |
|----------|------:|------------------:|---------------:|-----------:|--------------:|-----------------:|
| Buga | 5 833 | 3 215 (55,1%) | 5 831 (>99%) | 3 100 | 115 | 2 618 |
| Santander | 4 364 | 2 954 (67,7%) | 4 357 (>99%) | 2 724 | 230 | 1 410 |
| Totoró | 3 510 | 3 145 (89,6%) | 1 213 (34,6%) | 3 077 | 68 | 365 |

## Hiperparámetros (`resultados/hiperparametros/`)

| Archivo | Contenido |
|---------|-----------|
| `mejores_hp_dl.json` | Hiperparámetros óptimos de los modelos Deep Learning (LSTM, BiLSTM, CNN-BiLSTM) |
| `mejores_hp_ml.json` | Hiperparámetros óptimos de los modelos ML tradicional (MissForest/RF, XGBoost, CatBoost, LightGBM) |

## Modelos entrenados (`modelos/`)

| Archivo | Modelo | Estación |
|---------|--------|----------|
| `Buga_CNN_BiLSTM_v2.keras` | CNN-BiLSTM | Buga |
| `Santander_CNN_BiLSTM_v2.keras` | CNN-BiLSTM | Santander de Quilichao |
| `Totoro_CNN_BiLSTM_v2.keras` | CNN-BiLSTM | Totoró |

## Resultados de ablación (`resultados/ablacion/`)

Métricas de cada modelo DL bajo diferentes divisiones temporales y niveles de datos faltantes (10%, 15%, 20%), con 20 máscaras aleatorias por nivel.

## Tablas de resultados (`resultados/tablas/`)

Tablas Excel con métricas detalladas por estación: comparación integrada ML vs DL, resumen anual-mensual, y análisis de gaps faltantes.
