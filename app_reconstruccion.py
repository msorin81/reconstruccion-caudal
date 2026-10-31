r"""
Herramienta de reconstruccion de series de caudal diario
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
        "nse": 0.8125,
    },
    "Santander de Quilichao": {
        "archivo": "modelo_cnn_bilstm_v2_Santander.keras",
        "descripcion": "Cauca",
        "nse": None,
    },
    "Totoro": {
        "archivo": "modelo_cnn_bilstm_v2_Totoro.keras",
        "descripcion": "Cauca",
        "nse": None,
    },
}

# ── Paleta de colores ────────────────────────────────────────────────────────

COLOR_PRIMARY = "#DC143C"
COLOR_OBSERVED = "#1f77b4"
COLOR_RECONSTRUCTED = "#DC143C"
COLOR_BG_CARD = "#f8f9fa"

# ── Configuracion de pagina ──────────────────────────────────────────────────

st.set_page_config(
    page_title="Reconstruccion de Caudal",
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
    """Detecta automaticamente la columna de caudal en el dataset."""
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
    st.markdown("### Reconstruccion de Caudal")
    st.caption("CNN-BiLSTM v2 | Proyecto de Maestria")
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
        ["Modelo pre-entrenado", "Entrenar nuevo modelo"],
        label_visibility="collapsed",
    )

    if modelo_opcion == "Modelo pre-entrenado":
        estacion = st.selectbox(
            "Estacion hidrrologica",
            list(ESTACIONES.keys()),
            label_visibility="collapsed",
        )
        info_est = ESTACIONES[estacion]
        st.caption(f":material/location_on: {info_est['descripcion']}")
        if info_est["nse"]:
            st.caption(f":material/monitoring: NSE = {info_est['nse']}")

    st.divider()
    st.markdown("**3. Parametros**")

    umbral_gap = st.slider(
        "Umbral maximo de gap (dias)",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
    )

    if modelo_opcion == "Entrenar nuevo modelo":
        epochs = st.slider("Epocas", 50, 300, 200, 10)
        batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1)

    st.divider()
    st.caption(
        "Sube un dataset con columnas climaticas y de caudal. "
        "La columna de caudal se detecta automaticamente."
    )

# ── Header principal ─────────────────────────────────────────────────────────

st.markdown(
    '<div class="main-header">'
    "<h1>Reconstruccion de Series de Caudal Diario</h1>"
    "<p>Modelo CNN-BiLSTM v2 &mdash; Proyecto Aplicado de Maestria</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ── Sin archivo: pagina de bienvenida ────────────────────────────────────────

if archivo is None:
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown("#### Como funciona")
        st.markdown(
            """
1. **Suba su dataset** en la barra lateral (Excel o CSV).
2. **Seleccione un modelo** pre-entrenado o entrene uno nuevo.
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
                "Descripcion": [
                    "Fecha (YYYY-MM-DD)",
                    "Precipitacion diaria (mm)",
                    "Temperatura media (C)",
                    "Humedad relativa media (%)",
                    "Evaporacion calculada (mm)",
                    "Radiacion solar (cal/cm2)",
                    "Caudal medio diario (m3/s)",
                ],
            }
        )
        st.dataframe(cols_df, hide_index=True, use_container_width=True)
        st.caption(
            "* La columna de caudal se detecta automaticamente: "
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
            "#### Entrenar para otra estacion\n"
            "Si su dataset corresponde a una estacion diferente, "
            "seleccione *Entrenar nuevo modelo* en la barra lateral."
        )

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
        "No se encontro una columna de caudal en el dataset.\n\n"
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
        "El dataset debe contener las columnas climaticas requeridas."
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

tab_resumen, tab_reconstruccion, tab_datos = st.tabs(
    [
        ":material/bar_chart: Resumen",
        ":material/construction: Reconstruccion",
        ":material/table_chart: Datos",
    ]
)

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
        yaxis_title="Caudal (m3/s)",
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

            st.markdown("##### Analisis de gaps")
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Num. de gaps", f"{len(gap_lengths):,}")
            g2.metric("Longitud media", f"{np.mean(gap_lengths):.1f} d")
            g3.metric("Longitud max.", f"{np.max(gap_lengths):,} d")
            g4.metric("Mediana", f"{np.median(gap_lengths):.0f} d")

            fig_hist = go.Figure()
            fig_hist.add_trace(
                go.Histogram(
                    x=gap_lengths,
                    nbinsx=30,
                    marker_color=COLOR_OBSERVED,
                    opacity=0.8,
                )
            )
            fig_hist.update_layout(
                title="Distribucion de longitud de gaps",
                xaxis_title="Longitud (dias)",
                yaxis_title="Frecuencia",
                template="plotly_white",
                height=280,
                margin=dict(t=40, b=40, l=60, r=20),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    with st.expander("Vista previa del dataset"):
        st.dataframe(df_full.head(30), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: RECONSTRUCCION
# ══════════════════════════════════════════════════════════════════════════════

with tab_reconstruccion:
    if n_faltante == 0:
        st.success("El dataset no tiene datos faltantes. No se requiere reconstruccion.")
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

        if modelo_opcion == "Modelo pre-entrenado":
            with st.spinner(f"Cargando modelo para {estacion}..."):
                model = cargar_modelo_preentrenado(estacion)

            if model is not None:
                st.success(
                    f"Modelo cargado: **CNN_BiLSTM_v2 ({estacion})** — "
                    f"{model.count_params():,} parametros"
                )
            else:
                st.error(
                    f"No se encontro el modelo para {estacion}. "
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
                    f"max {epochs} epocas..."
                )

                class ProgressCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        pct = min((epoch + 1) / epochs, 1.0)
                        progress_bar.progress(
                            pct,
                            text=f"Epoca {epoch+1}/{epochs} | "
                            f"loss: {logs.get('loss', 0):.6f} | "
                            f"val_loss: {logs.get('val_loss', 0):.6f}",
                        )

                history = model.fit(
                    X_all,
                    y_all,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.10,
                    callbacks=get_callbacks() + [ProgressCallback()],
                    verbose=0,
                )

                progress_bar.progress(1.0, text="Entrenamiento completado")
                status_text.empty()

                best_epoch = np.argmin(history.history["val_loss"]) + 1
                best_val_loss = min(history.history["val_loss"])

                st.success(
                    f"Modelo entrenado — Mejor epoca: {best_epoch} | "
                    f"val_loss: {best_val_loss:.6f}"
                )

                st.session_state["model"] = model
                st.session_state["scaler_X"] = scaler_X
                st.session_state["scaler_y"] = scaler_y
                st.session_state["df_full"] = df_full
                st.session_state["history"] = history.history

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
                        name="Validacion",
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
                    xaxis_title="Epoca",
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
            st.info("Presione el boton para entrenar el modelo.")
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
            yaxis_title="Caudal (m3/s)",
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

        st.markdown("##### Estadisticas comparativas")

        q_obs_vals = df_final.loc[obs_mask, "Caudal_m3s"].dropna()
        q_rec_vals = df_final.loc[rec_mask, "Caudal_m3s"].dropna()

        if len(q_rec_vals) > 0:
            stats = pd.DataFrame(
                {
                    "Observado": [
                        q_obs_vals.mean(),
                        q_obs_vals.std(),
                        q_obs_vals.min(),
                        q_obs_vals.quantile(0.25),
                        q_obs_vals.median(),
                        q_obs_vals.quantile(0.75),
                        q_obs_vals.max(),
                    ],
                    "Reconstruido": [
                        q_rec_vals.mean(),
                        q_rec_vals.std(),
                        q_rec_vals.min(),
                        q_rec_vals.quantile(0.25),
                        q_rec_vals.median(),
                        q_rec_vals.quantile(0.75),
                        q_rec_vals.max(),
                    ],
                },
                index=[
                    "Media",
                    "Desv. Est.",
                    "Minimo",
                    "Q25",
                    "Mediana",
                    "Q75",
                    "Maximo",
                ],
            )
            stats = stats.round(3)

            col_t, col_b = st.columns(2)
            with col_t:
                st.dataframe(stats, use_container_width=True)
            with col_b:
                fig_box = go.Figure()
                fig_box.add_trace(
                    go.Box(y=q_obs_vals, name="Observado", marker_color=COLOR_OBSERVED)
                )
                fig_box.add_trace(
                    go.Box(
                        y=q_rec_vals,
                        name="Reconstruido",
                        marker_color=COLOR_RECONSTRUCTED,
                    )
                )
                fig_box.update_layout(
                    yaxis_title="Caudal (m3/s)",
                    template="plotly_white",
                    height=350,
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
        st.info("Ejecute la reconstruccion primero (pestana Reconstruccion).")
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
    "**CNN-BiLSTM v2** | Reconstruccion de series de caudal diario | "
    "Proyecto Aplicado de Maestria — 2026"
)
