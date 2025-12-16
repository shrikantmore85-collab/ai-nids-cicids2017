import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================================
#  PATHS (robust, based on this file location)
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "processed_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "xgb_model.joblib")
SHAP_IMG_PATH = os.path.join(BASE_DIR, "..", "outputs", "shap_values_summary.png")
LIME_HTML_PATH = os.path.join(BASE_DIR, "..", "outputs", "lime_explanation.html")

TEESSIDE_LOGO = os.path.join(BASE_DIR, "teesside.png")
MDIS_LOGO = os.path.join(BASE_DIR, "mdis.png")

# ============================================================================
#  UI CONFIG
# ============================================================================
st.set_page_config(
    page_title="Network Intrusion Detection — Dataset & Model Monitoring",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Colour palette (SOC-style, colourful)
PRIMARY = "#4F46E5"   # indigo
ACCENT1 = "#22C55E"   # green
ACCENT2 = "#F97316"   # orange
ACCENT3 = "#E11D48"   # pink/red
ACCENT4 = "#0EA5E9"   # cyan
DARK_BG = "#020617"

# Small CSS tweak for dark background and card-style look
st.markdown(
    f"""
    <style>
        .main {{
            background-color: {DARK_BG};
            color: #e5e7eb;
        }}
        .stMetric > div {{
            background-color: #020617;
            border-radius: 0.75rem;
            padding: 0.75rem;
            border: 1px solid #1f2937;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
#  HELPERS
# ============================================================================
def find_label_column(df: pd.DataFrame) -> str | None:
    """Find the label column (case-insensitive 'label' / 'class' etc.)."""
    for col in df.columns:
        if col.lower() in ("label", "class", "target", "binary label", "binary_label"):
            return col
    return None


@st.cache_data
def load_data() -> pd.DataFrame | None:
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    return df


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


df = load_data()
model = load_model()

# ============================================================================
#  SIDEBAR
# ============================================================================
st.sidebar.title("Controls")

view = st.sidebar.radio(
    "View",
    ["Overview", "Data", "Model", "Explainability", "Live Monitor"],
)

sample_size = st.sidebar.slider(
    "Sample size for visuals", min_value=200, max_value=5000, value=1000, step=100
)
show_table = st.sidebar.checkbox("Show raw table", value=False)
if st.sidebar.button("Refresh"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.experimental_rerun()

# ============================================================================
#  HEADER (LOGOS + TITLE + TAGLINE)
# ============================================================================
top_left, top_center, top_right = st.columns([1, 4, 1])

with top_left:
    if os.path.exists(TEESSIDE_LOGO):
        st.image(TEESSIDE_LOGO, width=260)
    else:
        st.markdown("**Teesside University**")

with top_center:
    st.markdown(
        f"<h2 style='text-align:center;font-size:40px;color:#e5e7eb;'>"
        f"Network Intrusion Detection — Dataset & Model Monitoring"
        f"</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
    f"""
    <p style="
        text-align:center;
        font-size:22px;
        line-height:1.6;
        margin-top:10px;
        margin-bottom:20px;
    ">
        <span style="color:{ACCENT2}; font-weight:600;">
            EXPLAINABLE AI-ENABLED NIDS
        </span>
        <span style="color:{ACCENT2};">
            USING
        </span>
        <span style="color:{ACCENT2}; font-weight:600;">
            CIC-IDS 2017
        </span>
        <span style="color:{ACCENT2};">
            •
        </span>
        <span style="color:{ACCENT2}; font-weight:600;">
            SOC-STYLE DASHBOARD
        </span>
    </p>
    """,
    unsafe_allow_html=True,
)


with top_right:
    if os.path.exists(MDIS_LOGO):
        st.image(MDIS_LOGO, width=260)
    else:
        st.markdown("**MDIS**")

status_ok = (df is not None) and (model is not None)
status_color = "#22c55e" if status_ok else "#ef4444"
st.markdown(
    f"<div style='text-align:right;font-weight:2000;color:{status_color};'>"
    f"Status: {'Ready' if status_ok else 'Missing data/model'}"
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# ============================================================================
#  OVERVIEW
# ============================================================================
if view == "Overview":
    st.subheader("System Overview")

    if df is None:
        st.error("Processed dataset missing. Run `python src/preprocess.py` first.")
    else:
        label_col = find_label_column(df)
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)

        total_rows = len(df)
        total_features = df.shape[1]
        benign_count = None
        attack_count = None
        attack_rate = None

        if label_col is not None:
            # Treat numeric labels (0/1) specially
            y = df[label_col]
            try:
                y_num = pd.to_numeric(y, errors="coerce")
            except Exception:
                y_num = y

            if set(np.unique(y_num.dropna())) <= {0, 1}:
                benign_count = int((y_num == 0).sum())
                attack_count = int((y_num == 1).sum())
                attack_rate = (attack_count / total_rows) * 100 if total_rows > 0 else 0.0
            else:
                # Multi-class or non-0/1 labels
                benign_count = int((y == 0).sum())  # might be 0
                attack_count = int(total_rows - benign_count)
                attack_rate = (attack_count / total_rows) * 100 if total_rows > 0 else 0.0

        with col1:
            st.metric("Total Flows", f"{total_rows:,}")
        with col2:
            st.metric("Total Features", total_features)
        with col3:
            st.metric("Attack Flows", f"{attack_count:,}" if attack_count is not None else "N/A")
        with col4:
            st.metric(
                "Attack Rate",
                f"{attack_rate:.2f}%" if attack_rate is not None else "N/A",
            )

        st.markdown("")

        # ---- Donut chart for label distribution ----
        if label_col is not None:
            label_series = df[label_col]
            # numeric or string, handle both
            try:
                y_num = pd.to_numeric(label_series, errors="coerce")
            except Exception:
                y_num = None

            if y_num is not None and set(np.unique(y_num.dropna())) <= {0, 1}:
                # binary numeric -> map to Benign/Attack
                class_names = []
                counts = []
                if benign_count is not None and benign_count > 0:
                    class_names.append("Benign")
                    counts.append(benign_count)
                if attack_count is not None and attack_count > 0:
                    class_names.append("Attack")
                    counts.append(attack_count)
                class_df = pd.DataFrame({"Class": class_names, "Count": counts})
            else:
                vc = label_series.value_counts()
                class_df = pd.DataFrame({"Class": vc.index.astype(str), "Count": vc.values})

        else:
            class_df = None

        left_pane, right_pane = st.columns([1.2, 1.8])

        # --- LEFT: Donut chart ---
        with left_pane:
            if class_df is not None and not class_df.empty:
                fig_pie = px.pie(
                    class_df,
                    names="Class",
                    values="Count",
                    hole=0.5,
                    title="Traffic Class Distribution",
                    color="Class",
                    color_discrete_map={
                        "Benign": ACCENT1,
                        "Attack": ACCENT3,
                    },
                )
                fig_pie.update_layout(
                    showlegend=True,
                    legend_title_text="Class",
                    paper_bgcolor=DARK_BG,
                    plot_bgcolor=DARK_BG,
                    font_color="#e5e7eb",
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Could not determine label distribution (no label column found).")

        # --- RIGHT: Curved spline line chart (simulated time buckets) ---
        with right_pane:
            st.markdown("### Attack Trend (sampled, SOC-style)")

            # choose numeric label for computing attack rate
            if label_col is not None:
                try:
                    y_num = pd.to_numeric(df[label_col], errors="coerce")
                except Exception:
                    y_num = None
            else:
                y_num = None

            if y_num is None:
                st.info("Label column not numeric; cannot compute attack trend.")
            else:
                # Sample for performance
                n = min(sample_size, len(df))
                sample_df = df.sample(n=n, random_state=42).copy()
                sample_df["__label_num"] = pd.to_numeric(
                    sample_df[label_col], errors="coerce"
                ).fillna(0)

                # Create artificial "time" index
                sample_df = sample_df.sort_index()
                sample_df["bucket"] = np.linspace(0, 100, len(sample_df)).astype(int)

                agg = (
                    sample_df.groupby("bucket")["__label_num"]
                    .agg(["mean", "count"])
                    .reset_index()
                )
                agg.rename(
                    columns={
                        "bucket": "Time Bucket",
                        "mean": "Attack Rate",
                        "count": "Flow Count",
                    },
                    inplace=True,
                )
                agg["Attack Rate (%)"] = agg["Attack Rate"] * 100.0

                fig_line = go.Figure()

                # Attack rate spline line
                fig_line.add_trace(
                    go.Scatter(
                        x=agg["Time Bucket"],
                        y=agg["Attack Rate (%)"],
                        mode="lines",
                        name="Attack Rate (%)",
                        line=dict(color=ACCENT3, width=3, shape="spline"),
                    )
                )

                # Flow count as secondary, lighter line
                fig_line.add_trace(
                    go.Scatter(
                        x=agg["Time Bucket"],
                        y=agg["Flow Count"],
                        mode="lines",
                        name="Flow Count",
                        line=dict(color=ACCENT4, width=2, dash="dot", shape="spline"),
                        yaxis="y2",
                    )
                )

                fig_line.update_layout(
                    xaxis_title="Time Bucket (simulated)",
                    yaxis=dict(title="Attack Rate (%)", color=ACCENT3),
                    yaxis2=dict(
                        title="Flow Count",
                        overlaying="y",
                        side="right",
                        showgrid=False,
                        color=ACCENT4,
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.02),
                    paper_bgcolor=DARK_BG,
                    plot_bgcolor="#020617",
                    font_color="#e5e7eb",
                )
                st.plotly_chart(fig_line, use_container_width=True)

        if show_table:
            st.markdown("### Sample of Processed Data")
            st.dataframe(df.head(100), use_container_width=True)

# ============================================================================
#  DATA PAGE
# ============================================================================
if view == "Data":
    st.subheader("Dataset Explorer")

    if df is None:
        st.error("No dataset found. Run `preprocess.py`.")
    else:
        st.markdown("#### Columns Summary")
        info_df = pd.DataFrame(
            {
                "Column": df.columns,
                "Non-null Count": df.notna().sum().values,
                "Dtype": df.dtypes.astype(str).values,
            }
        )
        st.dataframe(info_df, use_container_width=True, height=400)

        st.markdown("#### Feature Distribution")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.info("No numeric columns detected.")
        else:
            col_sel = st.selectbox("Choose feature", numeric_cols)
            fig_hist = px.histogram(
                df,
                x=col_sel,
                nbins=60,
                title=f"Distribution of {col_sel}",
                marginal="box",
                color_discrete_sequence=[ACCENT4],
            )
            fig_hist.update_layout(
                paper_bgcolor=DARK_BG,
                plot_bgcolor="#020617",
                font_color="#e5e7eb",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

# ============================================================================
#  MODEL PAGE
# ============================================================================
if view == "Model":
    st.subheader("Model Analysis")

    if df is None or model is None:
        st.error("Model or data missing. Run preprocess & train scripts first.")
    else:
        label_col = find_label_column(df)
        features_df = df.drop(columns=[label_col], errors="ignore") if label_col else df

        st.markdown("#### Sample Predictions")
        sample = features_df.sample(
            n=min(sample_size, len(features_df)), random_state=123
        )

        if st.button("Run Predictions"):
            preds = model.predict(sample)
            pred_df = sample.copy()
            pred_df["Prediction"] = preds
            st.dataframe(pred_df.head(50), use_container_width=True)
            st.write("Prediction counts:")
            st.write(pred_df["Prediction"].value_counts())

        st.markdown("#### Feature Importance")
        if hasattr(model, "feature_importances_"):
            imp = pd.DataFrame(
                {
                    "Feature": features_df.columns,
                    "Importance": model.feature_importances_,
                }
            ).sort_values("Importance", ascending=False)[:20]

            fig_imp = px.bar(
                imp,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Viridis",
                title="Top 20 Features by Importance",
            )
            fig_imp.update_layout(
                paper_bgcolor=DARK_BG,
                plot_bgcolor="#020617",
                font_color="#e5e7eb",
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Model does not expose feature_importances_ attribute.")

# ============================================================================
#  EXPLAINABILITY PAGE
# ============================================================================
if view == "Explainability":
    st.subheader("Explainability — SHAP & LIME")

    col_shap, col_lime = st.columns([2, 1])

    with col_shap:
        st.markdown("#### Global Feature Importance (SHAP)")
        if os.path.exists(SHAP_IMG_PATH):
            st.image(
                SHAP_IMG_PATH,
                caption="SHAP Summary Plot (Top Features Driving Model Output)",
                use_container_width=True,
            )
        else:
            st.warning("SHAP summary image not found. Run `python src/explain.py` first.")

    with col_lime:
        st.markdown("#### Local Explanation (LIME)")
        if os.path.exists(LIME_HTML_PATH):
            st.markdown(
                f"[Open detailed LIME explanation report]({LIME_HTML_PATH})",
                unsafe_allow_html=True,
            )
        else:
            st.warning("LIME explanation file not found. Run `python src/explain.py`.")

# ============================================================================
#  LIVE MONITOR PAGE
# ============================================================================
if view == "Live Monitor":
    st.subheader("Live Flow Monitoring (Simulated)")

    if df is None or model is None:
        st.warning("Provide dataset and trained model to enable live monitoring.")
    else:
        label_col = find_label_column(df)
        features_df = df.drop(columns=[label_col], errors="ignore") if label_col else df

        flow = features_df.sample(1, random_state=999)
        st.markdown("#### Live Sample Flow (simulated)")
        st.dataframe(flow.T, use_container_width=True)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(flow)[0]
            # If binary classifier, use probability of class 1 as attack
            if probs.shape[0] == 2:
                attack_prob = float(probs[1] * 100.0)
            else:
                attack_prob = float(np.max(probs) * 100.0)

            st.markdown("#### Attack Probability Gauge")
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=attack_prob,
                    title={"text": "Attack Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": ACCENT3},
                        "steps": [
                            {"range": [0, 40], "color": "#16a34a"},
                            {"range": [40, 70], "color": "#facc15"},
                            {"range": [70, 100], "color": "#b91c1c"},
                        ],
                    },
                )
            )
            gauge.update_layout(
                paper_bgcolor=DARK_BG,
                font_color="#e5e7eb",
            )
            st.plotly_chart(gauge, use_container_width=True)
        else:
            st.info(
                "Model does not support `predict_proba`. Cannot display probability gauge."
            )
