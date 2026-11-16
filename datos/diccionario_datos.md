# Diccionario de datos

## Series reconstruidas (`datos/reconstruidos/*.csv`)

Un archivo por estación, codificado en UTF-8, separado por comas, una fila por día.

| Columna | Tipo | Descripción |
|---|---|---|
| `Fecha` | fecha (AAAA-MM-DD) | Día de la observación |
| `Caudal_m3s` | decimal | Caudal medio diario en m³/s. Vacío cuando el día no se pudo reconstruir |
| `Fuente` | texto | Origen del valor (ver tabla siguiente) |

Valores posibles de `Fuente`:

| Valor | Significado |
|---|---|
| `Observado` | Caudal medido por el IDEAM, sin modificar |
| `Reconstruido (CNN_BiLSTM_v2)` | Caudal estimado por el modelo CNN-BiLSTM en un vacío de 100 días o menos |
| `No reconstruido (gap >100d)` | Día dentro de un vacío de más de 100 días consecutivos. `Caudal_m3s` queda vacío |

Las series incluyen solo los días con todas las variables climáticas disponibles, por eso pueden tener unos pocos días menos que los archivos de entrada.

## Datos de entrada (`datos/entrada/*.xlsx`, hoja `Data`)

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
| `rdcion_slar` | Radiación solar diaria: cal/cm²/día en Buga y Santander de Quilichao; MJ/m²/día en Totoró |
| `Caudal (m^3/s)` / `m^3/s` | Caudal medio diario observado (m³/s), IDEAM |

Los nombres de algunas columnas varían ligeramente entre estaciones (indicados con `/`). Los cuadernos los normalizan al cargar los datos.

La radiación solar viene en unidades distintas según la estación. Esto no afecta los modelos porque cada estación tiene su propio modelo y sus variables se escalan por separado (MinMax a [0, 1]). Para comparar valores entre estaciones: 1 cal/cm²/día = 0,04184 MJ/m²/día.
