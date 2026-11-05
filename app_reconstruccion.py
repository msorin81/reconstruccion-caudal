r"""
Herramienta de reconstrucción de series de caudal diario
usando el modelo CNN-BiLSTM v2.

Ejecutar localmente:
    streamlit run app_reconstruccion.py
"""

import io
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Input,
    LSTM,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ── Constantes del modelo ────────────────────────────────────────────────────

WINDOW = 15
HORIZON = 7
TOTAL_LEN = WINDOW + HORIZON + 1

HP_CNN_BILSTM = {
    "units": 64,
    "dropout": 0.1,
    "filters": 32,
    "kernel_size": 7,
    "dense_units": 16,
    "learning_rate": 0.001,
}

FEATURES_V2 = [
    "Q_log_input",
    "prcptcion",
    "tmp_mdia",
    "hmd_rel_mdia",
    "evp_clclada",
    "rdcion_slar",
    "mes_sin",
    "mes_cos",
    "dia_sin",
    "dia_cos",
]

TARGET_STD = "Caudal (m^3/s)"

COLUMNAS_CLIMA = ["prcptcion", "tmp_mdia", "hmd_rel_mdia", "evp_clclada", "rdcion_slar"]

COLUMNAS_IMPUTAR = ["tmp_minnma", "tmp_minma", "tmp_mdia", "hmd_rel_mdia"]

MODELOS_DIR = os.path.join(os.path.dirname(__file__), "modelos")

ESTACIONES = {
    "Buga": {
        "archivo": "modelo_cnn_bilstm_v2_Buga.keras",
        "descripcion": "Valle del Cauca",
        "nse": 0.9785,
    },
    "Santander de Quilichao": {
        "archivo": "modelo_cnn_bilstm_v2_Santander.keras",
        "descripcion": "Cauca",
        "nse": 0.9633,
    },
    "Totoro": {
        "archivo": "modelo_cnn_bilstm_v2_Totoro.keras",
        "descripcion": "Cauca",
        "nse": 0.9621,
    },
}

# ── Paleta de colores ────────────────────────────────────────────────────────

COLOR_PRIMARY = "#DC143C"
COLOR_OBSERVED = "#1f77b4"
COLOR_RECONSTRUCTED = "#DC143C"
COLOR_BG_CARD = "#f8f9fa"

# ── Configuracion de pagina ──────────────────────────────────────────────────

st.set_page_config(
    page_title="Reconstrucción de Caudal",
    page_icon=":material/waves:",
    layout="wide",
)

# ── CSS personalizado ────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* Header principal */
    .main-header {
        padding: 1.5rem 0 1rem 0;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        font-size: 0.95rem;
        color: #666;
        margin: 0;
    }

    /* Tarjetas de metrica */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 12px 16px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #fafbfc;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 20px;
    }

    /* Ocultar hamburger y footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Funciones auxiliares ─────────────────────────────────────────────────────


def detectar_columna_caudal(columnas: list[str]) -> str | None:
    """Detecta automáticamente la columna de caudal en el dataset."""
    candidatos = [
        "Caudal (m^3/s)",
        "caudal (m^3/s)",
        "Caudal(m^3/s)",
        "m^3/s",
        "Caudal",
        "caudal",
        "Q_m3s",
        "Q",
    ]
    for c in candidatos:
        if c in columnas:
            return c

    for col in columnas:
        col_lower = col.lower().strip()
        if "caudal" in col_lower or "m^3/s" in col_lower or "m3/s" in col_lower:
            return col
    return None


def estandarizar_columnas(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Estandariza nombres de columnas. Retorna (df, nombre_original_caudal)."""
    df = df.copy()
    original = None

    col_caudal = detectar_columna_caudal(df.columns.tolist())
    if col_caudal and col_caudal != TARGET_STD:
        original = col_caudal
        df = df.rename(columns={col_caudal: TARGET_STD})
    elif col_caudal == TARGET_STD:
        original = TARGET_STD

    if "tmp_minma" in df.columns and "tmp_minnma" not in df.columns:
        df = df.rename(columns={"tmp_minma": "tmp_minnma"})

    return df, original


def imputar_promedio_historico(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    if columna not in df.columns:
        return df
    df = df.sort_values("Fecha_").copy()
    col_dia = "Dia"
    if "Dia" not in df.columns and "Día" in df.columns:
        col_dia = "Día"
    if col_dia not in df.columns:
        df["Dia"] = pd.to_datetime(df["Fecha_"]).dt.day
        col_dia = "Dia"
    if "Mes" not in df.columns:
        df["Mes"] = pd.to_datetime(df["Fecha_"]).dt.month
    df["ph"] = df.groupby(["Mes", col_dia])[columna].transform(
        lambda x: x.shift().expanding().mean()
    )
    df[columna] = df[columna].fillna(df["ph"])
    return df.drop(columns=["ph"])


def preparar_features(df_model: pd.DataFrame) -> pd.DataFrame:
    df_model["mes_sin"] = np.sin(2 * np.pi * df_model.index.month / 12)
    df_model["mes_cos"] = np.cos(2 * np.pi * df_model.index.month / 12)
    df_model["dia_sin"] = np.sin(2 * np.pi * df_model.index.dayofyear / 365)
    df_model["dia_cos"] = np.cos(2 * np.pi * df_model.index.dayofyear / 365)

    df_model["Q_log_input"] = np.log1p(df_model[TARGET_STD])
    df_model["Q_log_hist"] = df_model.groupby(df_model.index.dayofyear)[
        "Q_log_input"
    ].transform(lambda x: x.shift().expanding().mean())
    df_model["Q_log_input"] = df_model["Q_log_input"].fillna(df_model["Q_log_hist"])
    df_model = df_model.drop(columns=["Q_log_hist"])
    return df_model


def construir_secuencias(
    df, features, target, window, horizon, scaler_X=None, scaler_y=None, fit=False
):
    arr = df[features].values.astype(np.float32)
    caudal = df[target].values.astype(np.float64)
    fechas = df.index

    if fit:
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        scaler_X.fit(arr)
        mask_obs = ~np.isnan(caudal)
        scaler_y.fit(np.log1p(caudal[mask_obs]).reshape(-1, 1))

    arr_sc = scaler_X.transform(arr)
    total_len = window + horizon + 1
    X_list, y_list, idx_list = [], [], []

    for i in range(window, len(df) - horizon):
        if np.isnan(caudal[i]):
            continue
        seq = arr_sc[i - window : i + horizon + 1]
        if seq.shape[0] != total_len:
            continue
        X_list.append(seq)
        y_list.append(scaler_y.transform([[np.log1p(caudal[i])]])[0, 0])
        idx_list.append(fechas[i])

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        idx_list,
        scaler_X,
        scaler_y,
    )


def build_cnn_bilstm_v2(input_shape, hp):
    inp = Input(shape=input_shape)
    x = Conv1D(hp["filters"], hp["kernel_size"], activation="relu", padding="same")(inp)
    x = Bidirectional(LSTM(hp["units"], return_sequences=False, dropout=hp["dropout"]))(x)
    x = Dropout(hp["dropout"])(x)
    x = Dense(hp["dense_units"], activation="relu")(x)
    out = Dense(1)(x)
    return Model(inp, out, name="CNN_BiLSTM_v2")


def calcular_nse(obs, pred):
    """Nash-Sutcliffe Efficiency en escala original (m³/s)."""
    mean_obs = np.mean(obs)
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - mean_obs) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot


def get_callbacks():
    return [
        EarlyStopping(
            monitor="val_loss", patience=25, restore_best_weights=True, verbose=0
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=12, min_lr=1e-6, verbose=0
        ),
    ]


def cargar_modelo_preentrenado(estacion):
    info = ESTACIONES[estacion]
    path = os.path.join(MODELOS_DIR, info["archivo"])
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path)


def reconstruir_gaps(model, df_full, scaler_X, scaler_y, umbral_gap=100):
    serie_caudal = df_full[TARGET_STD]

    gap_groups = serie_caudal.isna().ne(serie_caudal.isna().shift()).cumsum()
    gap_lengths = serie_caudal.isna().groupby(gap_groups).transform("sum")
    mask_rec = serie_caudal.isna() & (gap_lengths <= umbral_gap)

    arr_sc = scaler_X.transform(df_full[FEATURES_V2].values.astype(np.float32))
    pred_seqs, pred_fechas = [], []

    for i in range(WINDOW, len(df_full) - HORIZON):
        fecha = df_full.index[i]
        if not mask_rec.iloc[i]:
            continue
        seq = arr_sc[i - WINDOW : i + HORIZON + 1]
        if seq.shape[0] == TOTAL_LEN:
            pred_seqs.append(seq)
            pred_fechas.append(fecha)

    Q_rec = serie_caudal.values.copy().astype(float)
    fuente = np.array(["Observado"] * len(df_full), dtype=object)

    if pred_seqs:
        y_sc = model.predict(
            np.array(pred_seqs, dtype=np.float32), verbose=0
        ).flatten()
        y_m3s = np.maximum(
            np.expm1(scaler_y.inverse_transform(y_sc.reshape(-1, 1)).flatten()), 0
        )
        for fecha, val in zip(pred_fechas, y_m3s):
            idx = df_full.index.get_loc(fecha)
            Q_rec[idx] = val
            fuente[idx] = "Reconstruido"

    for i in range(len(df_full)):
        if serie_caudal.isna().iloc[i] and fuente[i] == "Observado":
            fuente[i] = f"No reconstruido (gap >{umbral_gap}d)"

    df_final = pd.DataFrame(
        {"Fecha": df_full.index, "Caudal_m3s": Q_rec, "Fuente": fuente}
    )
    df_final.set_index("Fecha", inplace=True)
    return df_final


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/water.png",
        width=48,
    )
    st.markdown("### Reconstrucción de Caudal")
    st.caption("CNN-BiLSTM v2 | Proyecto de Maestría")
    st.divider()

    st.markdown("**1. Cargar datos**")
    archivo = st.file_uploader(
        "Dataset (CSV o Excel)",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**2. Seleccionar modelo**")

    modelo_opcion = st.radio(
        "Fuente del modelo",
        ["Modelo preentrenado", "Entrenar nuevo modelo"],
        label_visibility="collapsed",
    )

    if modelo_opcion == "Modelo preentrenado":
        estacion = st.selectbox(
            "Estación hidrológica",
            list(ESTACIONES.keys()),
            label_visibility="collapsed",
        )
        info_est = ESTACIONES[estacion]
        st.caption(f":material/location_on: {info_est['descripcion']}")
        if info_est["nse"]:
            st.caption(f":material/monitoring: NSE = {info_est['nse']}")

    st.divider()
    st.markdown("**3. Parámetros**")

    umbral_gap = st.slider(
        "Umbral máximo de gap (días)",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
    )

    if modelo_opcion == "Entrenar nuevo modelo":
        division = st.selectbox(
            "División train/val",
            ["80/20", "85/15 (recomendado)", "90/10"],
            index=1,
        )
        val_split = {"80/20": 0.20, "85/15 (recomendado)": 0.15, "90/10": 0.10}[division]
        epochs = st.slider("Épocas", 50, 300, 200, 10)
        batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1)

    st.divider()
    st.caption(
        "Sube un dataset con columnas climáticas y de caudal. "
        "La columna de caudal se detecta automáticamente."
    )

# ── Header principal ─────────────────────────────────────────────────────────

st.markdown(
    '<div class="main-header">'
    "<h1>Reconstrucción de Series de Caudal Diario</h1>"
    "<p>Modelo CNN-BiLSTM v2 &mdash; Proyecto Aplicado de Maestría</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ── Funcion de portada (reutilizable) ────────────────────────────────────────


def render_portada():
    """Renderiza la pagina de bienvenida / inicio."""
    # ── Ilustracion hero ─────────────────────────────────────────────────
    t = np.linspace(0, 6 * np.pi, 300)
    np.random.seed(42)
    y_base = (
        8 + 3 * np.sin(t) + 1.5 * np.sin(2.3 * t + 0.5)
        + 5 * np.exp(-((t - 4.5) ** 2) / 0.3)
        + 4 * np.exp(-((t - 10.5) ** 2) / 0.4)
        + 6 * np.exp(-((t - 16) ** 2) / 0.5)
        + 0.4 * np.random.randn(300)
    )
    y_base = np.clip(y_base, 1, None)

    gap_start, gap_end = 105, 165
    y_obs = y_base.copy()
    y_obs[gap_start:gap_end] = np.nan
    y_rec = np.full_like(y_base, np.nan)
    y_rec[gap_start - 1 : gap_end + 1] = y_base[gap_start - 1 : gap_end + 1]

    fig_hero = go.Figure()
    fig_hero.add_vrect(
        x0=t[gap_start], x1=t[gap_end - 1],
        fillcolor="rgba(220,20,60,0.08)", line_width=0, layer="below",
    )
    fig_hero.add_trace(go.Scatter(
        x=t, y=y_obs, mode="lines", name="Observado",
        line=dict(color=COLOR_OBSERVED, width=2),
    ))
    fig_hero.add_trace(go.Scatter(
        x=t, y=y_rec, mode="lines", name="Reconstruido (CNN-BiLSTM)",
        line=dict(color=COLOR_RECONSTRUCTED, width=2, dash="dash"),
    ))
    fig_hero.add_annotation(
        x=t[(gap_start + gap_end) // 2], y=max(y_base[gap_start:gap_end]) + 1.5,
        text="Gap reconstruido", showarrow=False,
        font=dict(size=11, color=COLOR_RECONSTRUCTED),
    )
    fig_hero.update_layout(
        template="plotly_white",
        height=220,
        margin=dict(t=10, b=30, l=40, r=20),
        xaxis=dict(showticklabels=False, title=""),
        yaxis=dict(title="Caudal (m³/s)", title_font_size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5),
        showlegend=True,
    )
    st.plotly_chart(fig_hero, use_container_width=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown("#### ¿Cómo funciona?")
        st.markdown(
            """
1. **Suba su dataset** en la barra lateral (Excel o CSV).
2. **Seleccione un modelo** preentrenado o entrene uno nuevo.
3. **Visualice y descargue** la serie reconstruida.

El modelo CNN-BiLSTM v2 combina capas convolucionales (CNN) con redes
LSTM bidireccionales para capturar patrones temporales y reconstruir
datos faltantes en series de caudal diario.
"""
        )

        st.markdown("#### Columnas requeridas")
        cols_df = pd.DataFrame(
            {
                "Columna": [
                    "Fecha_",
                    "prcptcion",
                    "tmp_mdia",
                    "hmd_rel_mdia",
                    "evp_clclada",
                    "rdcion_slar",
                    "Caudal *",
                ],
                "Descripción": [
                    "Fecha (YYYY-MM-DD)",
                    "Precipitación diaria (mm)",
                    "Temperatura media (°C)",
                    "Humedad relativa media (%)",
                    "Evaporación calculada (mm)",
                    "Radiación solar (cal/cm²)",
                    "Caudal medio diario (m³/s)",
                ],
            }
        )
        st.dataframe(cols_df, hide_index=True, use_container_width=True)
        st.caption(
            "* La columna de caudal se detecta automáticamente: "
            "acepta `Caudal (m^3/s)`, `m^3/s`, `Caudal`, u otra variante."
        )

    with col_right:
        st.markdown("#### Modelos disponibles")
        for nombre, info in ESTACIONES.items():
            nse_txt = f"NSE = {info['nse']}" if info["nse"] else "NSE pendiente"
            st.markdown(
                f"**{nombre}**  \n"
                f":material/location_on: {info['descripcion']} | {nse_txt}"
            )
        st.markdown("---")
        st.markdown(
            "#### Entrenar para otra estación\n"
            "Si su dataset corresponde a una estación diferente, "
            "seleccione *Entrenar nuevo modelo* en la barra lateral. "
            "Se utiliza la misma arquitectura y los mismos hiperparámetros "
            "optimizados en el proyecto, aplicados a sus datos."
        )

    st.markdown("---")
    with st.expander("Arquitectura del modelo CNN-BiLSTM v2", expanded=False):
        col_arch, col_hp = st.columns(2)
        with col_arch:
            st.markdown(
                """
**Estructura de la red neuronal:**

| Capa | Configuración |
|------|---------------|
| **Entrada** | Secuencia de 23 pasos × 10 features |
| **Conv1D** | 32 filtros, kernel 7, ReLU, padding same |
| **BiLSTM** | 64 unidades bidireccionales, dropout 0.1 |
| **Dropout** | 10% |
| **Dense** | 16 neuronas, ReLU |
| **Salida** | 1 neurona (caudal predicho) |

La capa **Conv1D** extrae patrones locales en la ventana temporal
(estacionalidad, picos). La capa **BiLSTM** (Bidirectional LSTM)
procesa la secuencia en ambas direcciones (pasado → futuro y
futuro → pasado), capturando dependencias temporales a largo plazo.
"""
            )
        with col_hp:
            st.markdown(
                f"""
**Hiperparámetros y entrenamiento:**

| Parámetro | Valor |
|-----------|-------|
| Ventana pasada | {WINDOW} días |
| Horizonte futuro | {HORIZON} días |
| Secuencia total | {TOTAL_LEN} pasos |
| Learning rate | 0.001 |
| Optimizador | Adam |
| Loss | MSE |
| División train/val | 80/20, 85/15 o 90/10 |
| EarlyStopping | patience 25 |
| ReduceLROnPlateau | patience 12, factor 0.5 |

**10 features de entrada:** caudal histórico (log), precipitación,
temperatura media, humedad relativa, evaporación, radiación solar,
y 4 variables cíclicas (seno/coseno del mes y día del año).

**Target:** log1p(caudal) normalizado con MinMaxScaler.
"""
            )

    with st.expander("¿Cómo funciona el entrenamiento con un dataset nuevo?", expanded=False):
        st.markdown(
            f"""
Cuando selecciona *Entrenar nuevo modelo*, la herramienta ejecuta el siguiente
proceso con sus datos:

**1. Preprocesamiento**

Se aplica el mismo pipeline del proyecto: se imputan faltantes en variables
climáticas usando el promedio histórico por día del año, se crean las 4 variables
cíclicas (seno/coseno del mes y día), y se genera `Q_log_input` (log1p del caudal,
con faltantes rellenados por promedio histórico causal — solo usa datos pasados).

**2. Construcción de secuencias (solo para entrenamiento)**

Solo los días con caudal **observado** generan secuencias de entrenamiento,
porque se necesita el valor real como target para que el modelo aprenda.
Cada secuencia tiene {TOTAL_LEN} pasos: {WINDOW} días pasados + 1 día objetivo
+ {HORIZON} días futuros, con {len(FEATURES_V2)} features por paso. Los scalers
(MinMaxScaler) se ajustan exclusivamente con los datos del dataset cargado.

**3. Entrenamiento desde cero**

Se construye un modelo nuevo con la **misma arquitectura y los mismos
hiperparámetros** optimizados en el proyecto. No se transfieren pesos de los
modelos preentrenados, ya que los rangos de caudal y los patrones hidrológicos
varían entre cuencas. El dataset se divide según la proporción seleccionada
(80/20, 85/15 o 90/10) para entrenar y validar.

**4. Evaluación (NSE)**

Tras el entrenamiento, se predicen los datos de validación, se desnormalizan
a m³/s y se calcula el coeficiente de Nash-Sutcliffe (NSE). Un NSE cercano
a 1 indica alta confiabilidad; valores bajos (< 0.7) sugieren que las
reconstrucciones pueden no ser confiables para esa estación.

**5. Reconstrucción de faltantes**

El modelo recorre los días sin caudal. Para cada día faltante, **no necesita
que el caudal exista** — necesita las **10 features de entrada**, que siempre
están disponibles: las 5 variables climáticas provienen de estaciones
meteorológicas, las 4 cíclicas se calculan de la fecha, y `Q_log_input`
se rellena con el promedio histórico del día del año.

Cada día dentro de un gap se predice **independientemente** usando su contexto
climático real — no es una predicción encadenada donde el error se acumula.
Por eso el modelo puede reconstruir gaps de hasta {umbral_gap} días consecutivos
manteniendo una precisión alta (NSE > 0.96 en las estaciones del proyecto).

Los gaps mayores al umbral configurado quedan sin reconstruir por precaución.
"""
        )


# ── Sin archivo: pagina de bienvenida ────────────────────────────────────────

if archivo is None:
    render_portada()
    st.stop()

# ── Lectura del dataset ──────────────────────────────────────────────────────

try:
    if archivo.name.endswith(".csv"):
        df_raw = pd.read_csv(archivo)
    else:
        xls = pd.ExcelFile(archivo)
        sheet_names = xls.sheet_names
        if "Data" in sheet_names:
            default_sheet = "Data"
        else:
            default_sheet = sheet_names[0]
        if len(sheet_names) > 1:
            with st.sidebar:
                selected_sheet = st.selectbox(
                    "Hoja del Excel",
                    sheet_names,
                    index=sheet_names.index(default_sheet),
                )
        else:
            selected_sheet = default_sheet
        df_raw = pd.read_excel(xls, sheet_name=selected_sheet)
except Exception as e:
    st.error(f"Error al leer el archivo: {e}")
    st.stop()

# ── Estandarizar columnas ───────────────────────────────────────────────────

df_raw, col_caudal_original = estandarizar_columnas(df_raw)

if TARGET_STD not in df_raw.columns:
    st.error(
        "No se encontró una columna de caudal en el dataset.\n\n"
        "Se buscaron variantes como: `Caudal (m^3/s)`, `m^3/s`, `Caudal`, `Q`.\n\n"
        f"**Columnas encontradas:** {', '.join(df_raw.columns.tolist())}"
    )
    st.stop()

if col_caudal_original and col_caudal_original != TARGET_STD:
    st.info(
        f"Columna de caudal detectada: **{col_caudal_original}** "
        f"(estandarizada a `{TARGET_STD}`)"
    )

faltantes_clima = [c for c in COLUMNAS_CLIMA if c not in df_raw.columns]
if "Fecha_" not in df_raw.columns:
    faltantes_clima.insert(0, "Fecha_")

if faltantes_clima:
    st.error(
        f"Columnas faltantes: **{', '.join(faltantes_clima)}**\n\n"
        "El dataset debe contener las columnas climáticas requeridas."
    )
    st.stop()

# ── Preprocesamiento ─────────────────────────────────────────────────────────

with st.spinner("Preprocesando datos..."):
    df_raw["Fecha_"] = pd.to_datetime(df_raw["Fecha_"], errors="coerce")
    df_raw["tmp_mdia"] = pd.to_numeric(df_raw["tmp_mdia"], errors="coerce")
    df_raw["hmd_rel_mdia"] = pd.to_numeric(df_raw["hmd_rel_mdia"], errors="coerce")

    if "Mes" not in df_raw.columns:
        df_raw["Mes"] = df_raw["Fecha_"].dt.month
    if "Dia" not in df_raw.columns and "Día" not in df_raw.columns:
        df_raw["Dia"] = df_raw["Fecha_"].dt.day
    if "Día" in df_raw.columns and "Dia" not in df_raw.columns:
        df_raw["Dia"] = df_raw["Día"]

    df_imputado = df_raw.copy()
    for col in COLUMNAS_IMPUTAR:
        if col in df_imputado.columns:
            df_imputado = imputar_promedio_historico(df_imputado, col)

    df_model = df_imputado.sort_values("Fecha_").set_index("Fecha_").copy()
    df_model = preparar_features(df_model)

    df_full = df_model[FEATURES_V2 + [TARGET_STD]].dropna(subset=FEATURES_V2)

n_total = len(df_full)
n_obs = df_full[TARGET_STD].notna().sum()
n_faltante = df_full[TARGET_STD].isna().sum()
pct_faltante = n_faltante / n_total * 100

# ── Tabs principales ─────────────────────────────────────────────────────────

tab_inicio, tab_resumen, tab_reconstruccion, tab_datos = st.tabs(
    [
        ":material/home: Inicio",
        ":material/bar_chart: Resumen",
        ":material/construction: Reconstrucción",
        ":material/table_chart: Datos",
    ]
)

with tab_inicio:
    render_portada()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: RESUMEN
# ══════════════════════════════════════════════════════════════════════════════

with tab_resumen:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", f"{n_total:,}")
    c2.metric("Observados", f"{n_obs:,}")
    c3.metric("Faltantes", f"{n_faltante:,}")
    c4.metric("% Faltante", f"{pct_faltante:.1f}%")

    st.markdown("")

    fig_raw = go.Figure()
    fig_raw.add_trace(
        go.Scatter(
            x=df_full.index,
            y=df_full[TARGET_STD],
            mode="lines",
            name="Caudal observado",
            line=dict(color=COLOR_OBSERVED, width=1),
        )
    )
    fig_raw.update_layout(
        title="Serie de caudal original",
        xaxis_title="Fecha",
        yaxis_title="Caudal (m³/s)",
        template="plotly_white",
        height=380,
        margin=dict(t=40, b=40, l=60, r=20),
        hovermode="x unified",
    )
    st.plotly_chart(fig_raw, use_container_width=True)

    if n_faltante > 0:
        caudal_orig = df_full[TARGET_STD]
        is_gap = caudal_orig.isna()
        gap_groups = is_gap.ne(is_gap.shift()).cumsum()
        gap_only = gap_groups[is_gap]

        if len(gap_only) > 0:
            gap_sizes = gap_only.value_counts().sort_index()
            gap_lengths = gap_sizes.values

            n_gaps_rec = int(np.sum(gap_lengths <= umbral_gap))
            n_gaps_nrec = int(np.sum(gap_lengths > umbral_gap))
            dias_rec = int(np.sum(gap_lengths[gap_lengths <= umbral_gap]))
            dias_nrec = int(np.sum(gap_lengths[gap_lengths > umbral_gap]))

            st.markdown("##### Análisis de gaps")
            g1, g2, g3, g4, g5 = st.columns(5)
            g1.metric("Total gaps", f"{len(gap_lengths):,}")
            g2.metric(f"Gaps ≤{umbral_gap}d", f"{n_gaps_rec:,}", delta=f"{dias_rec:,} días")
            g3.metric(f"Gaps >{umbral_gap}d", f"{n_gaps_nrec:,}", delta=f"{dias_nrec:,} días", delta_color="inverse")
            g4.metric("Longitud máx.", f"{np.max(gap_lengths):,} d")
            g5.metric("Total faltantes", f"{n_faltante:,} d", delta=f"{pct_faltante:.1f}%", delta_color="inverse")

            fig_hist = go.Figure()
            gaps_rec_arr = gap_lengths[gap_lengths <= umbral_gap]
            gaps_nrec_arr = gap_lengths[gap_lengths > umbral_gap]

            if len(gaps_rec_arr) > 0:
                fig_hist.add_trace(
                    go.Histogram(
                        x=gaps_rec_arr,
                        nbinsx=20,
                        marker_color=COLOR_OBSERVED,
                        opacity=0.8,
                        name=f"Reconstruibles (≤{umbral_gap}d)",
                    )
                )
            if len(gaps_nrec_arr) > 0:
                fig_hist.add_trace(
                    go.Histogram(
                        x=gaps_nrec_arr,
                        nbinsx=10,
                        marker_color="#aaaaaa",
                        opacity=0.6,
                        name=f"No reconstruibles (>{umbral_gap}d)",
                    )
                )

            fig_hist.add_vline(
                x=umbral_gap,
                line_dash="dash",
                line_color=COLOR_RECONSTRUCTED,
                annotation_text=f"Umbral: {umbral_gap}d",
                annotation_position="top right",
            )
            fig_hist.update_layout(
                title="Distribución de longitud de gaps",
                xaxis_title="Longitud (días)",
                yaxis_title="Frecuencia",
                template="plotly_white",
                height=300,
                margin=dict(t=40, b=40, l=60, r=20),
                barmode="overlay",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    with st.expander("Vista previa del dataset"):
        st.dataframe(df_full.head(30), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: RECONSTRUCCION
# ══════════════════════════════════════════════════════════════════════════════

with tab_reconstruccion:
    if n_faltante == 0:
        st.success("El dataset no tiene datos faltantes. No se requiere reconstrucción.")
    else:
        with st.spinner("Construyendo secuencias..."):
            X_all, y_all, fechas_all, scaler_X, scaler_y = construir_secuencias(
                df_full, FEATURES_V2, TARGET_STD, WINDOW, HORIZON, fit=True
            )
            input_shape = (TOTAL_LEN, len(FEATURES_V2))

        st.caption(
            f"Secuencias: {X_all.shape[0]:,} | "
            f"Ventana: {WINDOW}d + 1 + {HORIZON}d | "
            f"Features: {len(FEATURES_V2)}"
        )

        # ── Cargar o entrenar modelo ─────────────────────────────────────────
        model = None

        if modelo_opcion == "Modelo preentrenado":
            with st.spinner(f"Cargando modelo para {estacion}..."):
                model = cargar_modelo_preentrenado(estacion)

            if model is not None:
                st.success(
                    f"Modelo cargado: **CNN_BiLSTM_v2 ({estacion})** — "
                    f"{model.count_params():,} parámetros"
                )
            else:
                st.error(
                    f"No se encontró el modelo para {estacion}. "
                    "Seleccione 'Entrenar nuevo modelo'."
                )
                st.stop()

        elif modelo_opcion == "Entrenar nuevo modelo":
            if st.button(
                "Entrenar modelo",
                type="primary",
                use_container_width=True,
                icon=":material/rocket_launch:",
            ):
                progress_bar = st.progress(0, text="Inicializando...")
                status_text = st.empty()

                tf.keras.backend.clear_session()
                model = build_cnn_bilstm_v2(input_shape, HP_CNN_BILSTM)
                model.compile(
                    optimizer=Adam(learning_rate=HP_CNN_BILSTM["learning_rate"]),
                    loss="mse",
                )

                status_text.text(
                    f"Entrenando ({model.count_params():,} params) — "
                    f"máx. {epochs} épocas..."
                )

                class ProgressCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        pct = min((epoch + 1) / epochs, 1.0)
                        progress_bar.progress(
                            pct,
                            text=f"Época {epoch+1}/{epochs} | "
                            f"loss: {logs.get('loss', 0):.6f} | "
                            f"val_loss: {logs.get('val_loss', 0):.6f}",
                        )

                history = model.fit(
                    X_all,
                    y_all,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=val_split,
                    callbacks=get_callbacks() + [ProgressCallback()],
                    verbose=0,
                )

                progress_bar.progress(1.0, text="Entrenamiento completado")
                status_text.empty()

                best_epoch = np.argmin(history.history["val_loss"]) + 1
                best_val_loss = min(history.history["val_loss"])

                n_val = max(1, int(len(X_all) * val_split))
                X_val = X_all[-n_val:]
                y_val_sc = y_all[-n_val:]
                y_pred_sc = model.predict(X_val, verbose=0).flatten()
                y_val_m3s = np.expm1(
                    scaler_y.inverse_transform(y_val_sc.reshape(-1, 1)).flatten()
                )
                y_pred_m3s = np.maximum(
                    np.expm1(
                        scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).flatten()
                    ),
                    0,
                )
                nse_val = calcular_nse(y_val_m3s, y_pred_m3s)

                st.success(
                    f"Modelo entrenado — Mejor época: {best_epoch} | "
                    f"val_loss: {best_val_loss:.6f} | "
                    f"**NSE (validación): {nse_val:.4f}**"
                )

                st.session_state["model"] = model
                st.session_state["scaler_X"] = scaler_X
                st.session_state["scaler_y"] = scaler_y
                st.session_state["df_full"] = df_full
                st.session_state["history"] = history.history
                st.session_state["nse_val"] = nse_val

                fig_loss = go.Figure()
                fig_loss.add_trace(
                    go.Scatter(
                        y=history.history["loss"],
                        mode="lines",
                        name="Train",
                        line=dict(color=COLOR_OBSERVED),
                    )
                )
                fig_loss.add_trace(
                    go.Scatter(
                        y=history.history["val_loss"],
                        mode="lines",
                        name="Validación",
                        line=dict(color=COLOR_RECONSTRUCTED),
                    )
                )
                fig_loss.add_vline(
                    x=best_epoch - 1,
                    line_dash="dash",
                    line_color="green",
                    annotation_text=f"Mejor: {best_epoch}",
                )
                fig_loss.update_layout(
                    title="Curva de entrenamiento",
                    xaxis_title="Época",
                    yaxis_title="MSE Loss",
                    template="plotly_white",
                    height=300,
                    margin=dict(t=40, b=40, l=60, r=20),
                )
                st.plotly_chart(fig_loss, use_container_width=True)

        if model is None and "model" in st.session_state:
            model = st.session_state["model"]
            scaler_X = st.session_state["scaler_X"]
            scaler_y = st.session_state["scaler_y"]
            df_full = st.session_state["df_full"]

        if model is None:
            st.info("Presione el botón para entrenar el modelo.")
            st.stop()

        # ── Reconstruccion ───────────────────────────────────────────────────

        with st.spinner("Reconstruyendo gaps..."):
            df_final = reconstruir_gaps(model, df_full, scaler_X, scaler_y, umbral_gap)

        obs_mask = df_final["Fuente"] == "Observado"
        rec_mask = df_final["Fuente"].str.startswith("Reconstruido")
        nrec_mask = df_final["Fuente"].str.startswith("No recon")

        n_obs_f = obs_mask.sum()
        n_rec_f = rec_mask.sum()
        n_nrec_f = nrec_mask.sum()

        # ── Metricas ─────────────────────────────────────────────────────────

        m1, m2, m3 = st.columns(3)
        m1.metric("Observados", f"{n_obs_f:,}")
        m2.metric("Reconstruidos", f"{n_rec_f:,}", delta=f"{n_rec_f / n_total * 100:.1f}%")
        m3.metric(
            "No reconstruidos",
            f"{n_nrec_f:,}",
            delta=f"gap >{umbral_gap}d",
            delta_color="inverse",
        )

        cobertura = (n_obs_f + n_rec_f) / n_total * 100
        st.progress(cobertura / 100, text=f"Cobertura total: {cobertura:.1f}%")

        # ── Grafico reconstruido ─────────────────────────────────────────────

        fig = go.Figure()

        q_obs = df_final["Caudal_m3s"].where(obs_mask)
        q_rec = df_final["Caudal_m3s"].where(rec_mask)

        fig.add_trace(
            go.Scatter(
                x=df_final.index,
                y=q_obs,
                mode="lines",
                name=f"Observado ({n_obs_f:,}d)",
                line=dict(color=COLOR_OBSERVED, width=1),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_final.index,
                y=q_rec,
                mode="lines",
                name=f"Reconstruido ({n_rec_f:,}d)",
                line=dict(color=COLOR_RECONSTRUCTED, width=1.5),
            )
        )

        in_gap = False
        gap_ini = None
        for fecha, flag in nrec_mask.items():
            if flag and not in_gap:
                gap_ini = fecha
                in_gap = True
            elif not flag and in_gap:
                fig.add_vrect(
                    x0=gap_ini,
                    x1=fecha,
                    fillcolor="lightgray",
                    opacity=0.35,
                    layer="below",
                    line_width=0,
                )
                in_gap = False
        if in_gap and gap_ini is not None:
            fig.add_vrect(
                x0=gap_ini,
                x1=df_final.index[-1],
                fillcolor="lightgray",
                opacity=0.35,
                layer="below",
                line_width=0,
            )

        fig.update_layout(
            title="Serie de caudal reconstruida",
            xaxis_title="Fecha",
            yaxis_title="Caudal (m³/s)",
            template="plotly_white",
            height=450,
            margin=dict(t=50, b=50, l=60, r=20),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

        # ── Estadisticas comparativas ────────────────────────────────────────

        st.markdown("##### Serie sin reconstrucción vs con reconstrucción")

        q_sin = df_final.loc[obs_mask, "Caudal_m3s"].dropna()
        q_con = df_final.loc[obs_mask | rec_mask, "Caudal_m3s"].dropna()

        if len(q_con) > 0:
            def serie_stats(s):
                return [
                    len(s),
                    s.mean(),
                    s.std(),
                    s.min(),
                    s.quantile(0.25),
                    s.median(),
                    s.quantile(0.75),
                    s.max(),
                ]

            stats = pd.DataFrame(
                {
                    "Sin reconstrucción": serie_stats(q_sin),
                    "Con reconstrucción": serie_stats(q_con),
                },
                index=[
                    "N (días)",
                    "Media",
                    "Desv. Est.",
                    "Mínimo",
                    "Q25",
                    "Mediana",
                    "Q75",
                    "Máximo",
                ],
            )
            stats.iloc[0] = stats.iloc[0].astype(int)
            stats.iloc[1:] = stats.iloc[1:].round(3)

            col_t, col_b = st.columns(2)
            with col_t:
                st.dataframe(stats, use_container_width=True)
            with col_b:
                fig_box = go.Figure()
                fig_box.add_trace(
                    go.Box(
                        y=q_sin,
                        name=f"Sin reconstrucción ({len(q_sin):,}d)",
                        marker_color=COLOR_OBSERVED,
                    )
                )
                fig_box.add_trace(
                    go.Box(
                        y=q_con,
                        name=f"Con reconstrucción ({len(q_con):,}d)",
                        marker_color="#2ca02c",
                    )
                )
                fig_box.update_layout(
                    yaxis_title="Caudal (m³/s)",
                    template="plotly_white",
                    height=380,
                    margin=dict(t=20, b=40, l=60, r=20),
                    showlegend=False,
                )
                st.plotly_chart(fig_box, use_container_width=True)

        st.session_state["df_final"] = df_final

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: DATOS
# ══════════════════════════════════════════════════════════════════════════════

with tab_datos:
    if "df_final" not in st.session_state:
        st.info("Ejecute la reconstrucción primero (pestaña Reconstrucción).")
    else:
        df_final = st.session_state["df_final"]
        df_display = df_final.copy()
        df_display.index = df_display.index.strftime("%Y-%m-%d")

        rec_mask_d = df_display["Fuente"].str.startswith("Reconstruido")

        sub_tab1, sub_tab2 = st.tabs(["Solo reconstruidos", "Serie completa"])

        with sub_tab1:
            df_rec_only = df_display[rec_mask_d]
            st.dataframe(df_rec_only, use_container_width=True, height=450)

        with sub_tab2:
            st.dataframe(df_display, use_container_width=True, height=450)

        st.markdown("")
        st.markdown("##### Descargar resultados")
        dl1, dl2, _ = st.columns([1, 1, 2])

        with dl1:
            buf_csv = io.BytesIO()
            df_final.to_csv(buf_csv, encoding="utf-8-sig")
            st.download_button(
                label="Descargar CSV",
                data=buf_csv.getvalue(),
                file_name="caudal_reconstruido.csv",
                mime="text/csv",
                use_container_width=True,
                icon=":material/download:",
            )

        with dl2:
            buf_xlsx = io.BytesIO()
            with pd.ExcelWriter(buf_xlsx, engine="openpyxl") as writer:
                df_final.to_excel(writer, sheet_name="Serie_Completa")
                df_final[df_final["Fuente"].str.startswith("Reconstruido")].to_excel(
                    writer, sheet_name="Solo_Reconstruidos"
                )
            st.download_button(
                label="Descargar Excel",
                data=buf_xlsx.getvalue(),
                file_name="caudal_reconstruido.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                icon=":material/download:",
            )

# ── Footer ───────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "**CNN-BiLSTM v2** | Reconstrucción de series de caudal diario | "
    "Proyecto Aplicado de Maestría — 2026"
)
