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

COLUMNAS_REQUERIDAS = [
    "Fecha_",
    "prcptcion",
    "tmp_mdia",
    "hmd_rel_mdia",
    "evp_clclada",
    "rdcion_slar",
    "Caudal (m^3/s)",
]

COLUMNAS_IMPUTAR = ["tmp_minnma", "tmp_mdia", "hmd_rel_mdia"]

MODELOS_DIR = os.path.join(os.path.dirname(__file__), "modelos")

ESTACIONES = {
    "Buga": {
        "archivo": "modelo_cnn_bilstm_v2_Buga.keras",
        "descripcion": "Estacion Buga — Valle del Cauca",
        "nse": 0.8125,
    },
    "Santander de Quilichao": {
        "archivo": "modelo_cnn_bilstm_v2_Santander.keras",
        "descripcion": "Estacion Santander de Quilichao — Cauca",
        "nse": None,
    },
    "Totoro": {
        "archivo": "modelo_cnn_bilstm_v2_Totoro.keras",
        "descripcion": "Estacion Totoro — Cauca",
        "nse": None,
    },
}

# ── Configuracion de pagina ──────────────────────────────────────────────────

st.set_page_config(
    page_title="Reconstruccion de Caudal — CNN-BiLSTM",
    page_icon="\U0001F30A",
    layout="wide",
)

st.title("\U0001F30A Reconstruccion de Series de Caudal Diario")
st.markdown("**Modelo CNN-BiLSTM v2** — Proyecto Aplicado de Maestria")
st.divider()


# ── Funciones auxiliares ─────────────────────────────────────────────────────


def imputar_promedio_historico(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    df = df.sort_values("Fecha_").copy()
    df["ph"] = df.groupby(["Mes", "Dia"])[columna].transform(
        lambda x: x.shift().expanding().mean()
    )
    df[columna] = df[columna].fillna(df["ph"])
    return df.drop(columns=["ph"])


def preparar_features(df_model: pd.DataFrame) -> pd.DataFrame:
    target = "Caudal (m^3/s)"
    df_model["mes_sin"] = np.sin(2 * np.pi * df_model.index.month / 12)
    df_model["mes_cos"] = np.cos(2 * np.pi * df_model.index.month / 12)
    df_model["dia_sin"] = np.sin(2 * np.pi * df_model.index.dayofyear / 365)
    df_model["dia_cos"] = np.cos(2 * np.pi * df_model.index.dayofyear / 365)

    df_model["Q_log_input"] = np.log1p(df_model[target])
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
    target = "Caudal (m^3/s)"
    serie_caudal = df_full[target]

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
            fuente[idx] = "Reconstruido (CNN_BiLSTM_v2)"

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
    st.header("\U0001F4C2 Datos de Entrada")

    archivo = st.file_uploader(
        "Subir dataset (CSV o Excel)",
        type=["csv", "xlsx", "xls"],
        help="Debe contener las columnas: Fecha_, prcptcion, tmp_mdia, "
        "hmd_rel_mdia, evp_clclada, rdcion_slar, Caudal (m^3/s)",
    )

    st.divider()
    st.header("\U0001F9E0 Modelo")

    modelo_opcion = st.radio(
        "Fuente del modelo",
        [
            "Modelo pre-entrenado (recomendado)",
            "Entrenar nuevo modelo",
        ],
        help="Use el modelo pre-entrenado para resultados inmediatos, "
        "o entrene uno nuevo si tiene un dataset de otra estacion",
    )

    if modelo_opcion == "Modelo pre-entrenado (recomendado)":
        estacion = st.selectbox(
            "Estacion",
            list(ESTACIONES.keys()),
            help="Seleccione la estacion correspondiente al dataset",
        )
        info_est = ESTACIONES[estacion]
        st.caption(f"\U0001F4CD {info_est['descripcion']}")
        if info_est["nse"]:
            st.caption(f"\U0001F4CA NSE validacion: {info_est['nse']}")

    st.divider()
    st.header("⚙️ Parametros")

    umbral_gap = st.slider(
        "Umbral maximo de gap (dias)",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help="Gaps mas largos que este umbral no se reconstruyen",
    )

    if modelo_opcion == "Entrenar nuevo modelo":
        epochs = st.slider(
            "Epocas de entrenamiento",
            min_value=50,
            max_value=300,
            value=200,
            step=10,
        )
        batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1)

# ── Contenido principal ──────────────────────────────────────────────────────

if archivo is None:
    st.info(
        "\U0001F448 Suba un dataset en la barra lateral para comenzar.\n\n"
        "**Columnas requeridas:**\n"
        "- `Fecha_` — fecha en formato YYYY-MM-DD\n"
        "- `prcptcion` — precipitacion diaria (mm)\n"
        "- `tmp_mdia` — temperatura media diaria (°C)\n"
        "- `hmd_rel_mdia` — humedad relativa media (%)\n"
        "- `evp_clclada` — evaporacion calculada (mm)\n"
        "- `rdcion_slar` — radiacion solar (cal/cm²)\n"
        "- `Caudal (m^3/s)` — caudal medio diario (valores faltantes = vacio o NaN)\n\n"
        "**Modelos pre-entrenados disponibles:** Buga, Santander de Quilichao, Totoro"
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

faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df_raw.columns]
if faltantes:
    st.error(
        f"Columnas faltantes en el dataset: **{', '.join(faltantes)}**\n\n"
        "Revise que el archivo tenga las columnas requeridas."
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

    target = "Caudal (m^3/s)"
    df_full = df_model[FEATURES_V2 + [target]].dropna(subset=FEATURES_V2)

n_total = len(df_full)
n_obs = df_full[target].notna().sum()
n_faltante = df_full[target].isna().sum()
pct_faltante = n_faltante / n_total * 100

# ── Vista previa ─────────────────────────────────────────────────────────────

st.header("\U0001F4CA Resumen del Dataset")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Registros totales", f"{n_total:,}")
col2.metric("Datos observados", f"{n_obs:,}")
col3.metric("Datos faltantes", f"{n_faltante:,}")
col4.metric("% Faltante", f"{pct_faltante:.1f}%")

with st.expander("Vista previa del dataset", expanded=False):
    st.dataframe(df_full.head(20), use_container_width=True)

# ── Grafico de datos crudos ──────────────────────────────────────────────────

fig_raw = go.Figure()
fig_raw.add_trace(
    go.Scatter(
        x=df_full.index,
        y=df_full[target],
        mode="lines",
        name="Caudal observado",
        line=dict(color="#4682B4", width=1),
    )
)
fig_raw.update_layout(
    title="Serie de caudal original (con gaps)",
    xaxis_title="Fecha",
    yaxis_title="Caudal (m³/s)",
    template="plotly_white",
    height=350,
    margin=dict(t=40, b=40),
)
st.plotly_chart(fig_raw, use_container_width=True)

if n_faltante == 0:
    st.success("El dataset no tiene datos faltantes. No se requiere reconstruccion.")
    st.stop()

# ── Construccion de secuencias y scalers ─────────────────────────────────────

st.divider()
st.header("\U0001F9E0 Modelo CNN-BiLSTM v2")

with st.spinner("Construyendo secuencias..."):
    X_all, y_all, fechas_all, scaler_X, scaler_y = construir_secuencias(
        df_full, FEATURES_V2, target, WINDOW, HORIZON, fit=True
    )
    input_shape = (TOTAL_LEN, len(FEATURES_V2))

st.info(
    f"Secuencias construidas: **{X_all.shape[0]:,}** "
    f"(shape: {X_all.shape[1]}×{X_all.shape[2]}) — "
    f"Ventana: {WINDOW} dias pasados + 1 objetivo + {HORIZON} dias futuros"
)

# ── Cargar o entrenar modelo ─────────────────────────────────────────────────

model = None

if modelo_opcion == "Modelo pre-entrenado (recomendado)":
    with st.spinner(f"Cargando modelo pre-entrenado para {estacion}..."):
        model = cargar_modelo_preentrenado(estacion)

    if model is not None:
        st.success(
            f"Modelo cargado: **CNN_BiLSTM_v2 — {estacion}** "
            f"({model.count_params():,} parametros)"
        )
    else:
        st.error(
            f"No se encontro el modelo para {estacion}. "
            "Seleccione 'Entrenar nuevo modelo' en la barra lateral."
        )
        st.stop()

elif modelo_opcion == "Entrenar nuevo modelo":
    if st.button("\U0001F680 Entrenar modelo", type="primary", use_container_width=True):
        progress_bar = st.progress(0, text="Inicializando modelo...")
        status_text = st.empty()

        tf.keras.backend.clear_session()
        model = build_cnn_bilstm_v2(input_shape, HP_CNN_BILSTM)
        model.compile(
            optimizer=Adam(learning_rate=HP_CNN_BILSTM["learning_rate"]), loss="mse"
        )

        status_text.text(
            f"Entrenando CNN_BiLSTM_v2 ({model.count_params():,} parametros) — "
            f"{epochs} epocas max..."
        )

        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                pct = min((epoch + 1) / epochs, 1.0)
                progress_bar.progress(
                    pct,
                    text=f"Epoca {epoch+1}/{epochs} — "
                    f"loss: {logs.get('loss', 0):.6f} — "
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
            f"Modelo entrenado — Mejor epoca: {best_epoch} — "
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
                name="Train loss",
                line=dict(color="#4682B4"),
            )
        )
        fig_loss.add_trace(
            go.Scatter(
                y=history.history["val_loss"],
                mode="lines",
                name="Val loss",
                line=dict(color="#DC143C"),
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
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig_loss, use_container_width=True)

if model is None and "model" in st.session_state:
    model = st.session_state["model"]
    scaler_X = st.session_state["scaler_X"]
    scaler_y = st.session_state["scaler_y"]
    df_full = st.session_state["df_full"]

if model is None:
    st.stop()

# ── Reconstruccion ───────────────────────────────────────────────────────────

st.divider()
st.header("\U0001F527 Reconstruccion de Datos Faltantes")

with st.spinner("Reconstruyendo gaps..."):
    df_final = reconstruir_gaps(model, df_full, scaler_X, scaler_y, umbral_gap)

obs_mask = df_final["Fuente"] == "Observado"
rec_mask = df_final["Fuente"].str.startswith("Reconstruido")
nrec_mask = df_final["Fuente"].str.startswith("No recon")

n_obs_f = obs_mask.sum()
n_rec_f = rec_mask.sum()
n_nrec_f = nrec_mask.sum()

# ── Metricas de reconstruccion ───────────────────────────────────────────────

st.subheader("\U0001F4C8 Resultados Cuantitativos")

col1, col2, col3 = st.columns(3)
col1.metric("Dias observados", f"{n_obs_f:,}", help="Datos originales del dataset")
col2.metric(
    "Dias reconstruidos",
    f"{n_rec_f:,}",
    delta=f"{n_rec_f / n_total * 100:.1f}% del total",
    help="Datos imputados por el modelo CNN-BiLSTM v2",
)
col3.metric(
    "Gaps no reconstruidos",
    f"{n_nrec_f:,}",
    delta=f">{umbral_gap} dias consecutivos",
    delta_color="inverse",
    help=f"Gaps mayores a {umbral_gap} dias consecutivos",
)

cobertura = (n_obs_f + n_rec_f) / n_total * 100
st.progress(cobertura / 100, text=f"Cobertura total: {cobertura:.1f}%")

# ── Analisis de gaps ─────────────────────────────────────────────────────────

caudal_orig = df_full[target]
is_gap = caudal_orig.isna()
gap_groups = is_gap.ne(is_gap.shift()).cumsum()
gap_only = gap_groups[is_gap]

if len(gap_only) > 0:
    gap_sizes = gap_only.value_counts().sort_index()
    gap_lengths = gap_sizes.values

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de gaps", f"{len(gap_lengths):,}")
    col2.metric("Longitud media", f"{np.mean(gap_lengths):.1f} dias")
    col3.metric("Longitud maxima", f"{np.max(gap_lengths):,} dias")
    col4.metric("Longitud mediana", f"{np.median(gap_lengths):.0f} dias")

    with st.expander("Distribucion de longitud de gaps", expanded=False):
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=gap_lengths,
                nbinsx=30,
                marker_color="#4682B4",
                opacity=0.8,
            )
        )
        fig_hist.update_layout(
            title="Distribucion de longitud de gaps",
            xaxis_title="Longitud del gap (dias)",
            yaxis_title="Frecuencia",
            template="plotly_white",
            height=300,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# ── Grafico de la serie reconstruida ─────────────────────────────────────────

st.subheader("\U0001F4C9 Serie de Caudal Reconstruida")

fig = go.Figure()

q_obs = df_final["Caudal_m3s"].where(obs_mask)
q_rec = df_final["Caudal_m3s"].where(rec_mask)

fig.add_trace(
    go.Scatter(
        x=df_final.index,
        y=q_obs,
        mode="lines",
        name=f"Observado ({n_obs_f:,} dias)",
        line=dict(color="#4682B4", width=1),
    )
)

fig.add_trace(
    go.Scatter(
        x=df_final.index,
        y=q_rec,
        mode="lines",
        name=f"Reconstruido ({n_rec_f:,} dias)",
        line=dict(color="#DC143C", width=1.5),
    )
)

in_gap = False
gap_ini = None
for i, (fecha, flag) in enumerate(nrec_mask.items()):
    if flag and not in_gap:
        gap_ini = fecha
        in_gap = True
    elif not flag and in_gap:
        fig.add_vrect(
            x0=gap_ini,
            x1=fecha,
            fillcolor="lightgray",
            opacity=0.4,
            layer="below",
            line_width=0,
        )
        in_gap = False
if in_gap and gap_ini is not None:
    fig.add_vrect(
        x0=gap_ini,
        x1=df_final.index[-1],
        fillcolor="lightgray",
        opacity=0.4,
        layer="below",
        line_width=0,
    )

fig.update_layout(
    title=dict(
        text=(
            f"Serie de caudal diario — Observado: {n_obs_f:,} | "
            f"Reconstruido: {n_rec_f:,} | No reconstruido: {n_nrec_f:,}"
        ),
        font=dict(size=14),
    ),
    xaxis_title="Fecha",
    yaxis_title="Caudal (m³/s)",
    template="plotly_white",
    height=450,
    margin=dict(t=50, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)

# ── Estadisticas comparativas ────────────────────────────────────────────────

st.subheader("\U0001F4CB Estadisticas Comparativas")

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
        index=["Media", "Desv. Est.", "Minimo", "Q25", "Mediana", "Q75", "Maximo"],
    )
    stats = stats.round(3)

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(stats, use_container_width=True)
    with col2:
        fig_box = go.Figure()
        fig_box.add_trace(
            go.Box(y=q_obs_vals, name="Observado", marker_color="#4682B4")
        )
        fig_box.add_trace(
            go.Box(y=q_rec_vals, name="Reconstruido", marker_color="#DC143C")
        )
        fig_box.update_layout(
            title="Distribucion de caudales",
            yaxis_title="Caudal (m³/s)",
            template="plotly_white",
            height=350,
            margin=dict(t=40, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig_box, use_container_width=True)

# ── Tabla de datos reconstruidos ─────────────────────────────────────────────

st.subheader("\U0001F4C4 Datos Reconstruidos")

df_display = df_final.copy()
df_display.index = df_display.index.strftime("%Y-%m-%d")

tab1, tab2 = st.tabs(["Solo reconstruidos", "Serie completa"])

with tab1:
    df_rec_only = df_display[df_display["Fuente"].str.startswith("Reconstruido")]
    st.dataframe(df_rec_only, use_container_width=True, height=400)

with tab2:
    st.dataframe(df_display, use_container_width=True, height=400)

# ── Descarga ─────────────────────────────────────────────────────────────────

st.divider()
st.subheader("\U0001F4BE Descargar Resultados")

col1, col2 = st.columns(2)

with col1:
    buf_csv = io.BytesIO()
    df_final.to_csv(buf_csv, encoding="utf-8-sig")
    st.download_button(
        label="\U0001F4E5 Descargar CSV completo",
        data=buf_csv.getvalue(),
        file_name="caudal_reconstruido.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col2:
    buf_xlsx = io.BytesIO()
    with pd.ExcelWriter(buf_xlsx, engine="openpyxl") as writer:
        df_final.to_excel(writer, sheet_name="Serie_Completa")
        df_final[df_final["Fuente"].str.startswith("Reconstruido")].to_excel(
            writer, sheet_name="Solo_Reconstruidos"
        )
    st.download_button(
        label="\U0001F4E5 Descargar Excel",
        data=buf_xlsx.getvalue(),
        file_name="caudal_reconstruido.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

# ── Footer ───────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "\U0001F30A **CNN-BiLSTM v2** — Reconstruccion de series de caudal diario | "
    "Proyecto Aplicado de Maestria — 2026"
)
