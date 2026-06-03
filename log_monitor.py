import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Log Analyzer",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
    }

    .stApp {
        background: linear-gradient(135deg, #020617 0%, #0f172a 50%, #111827 100%);
        color: white;
    }

    .title-box {
        padding: 25px;
        border-radius: 18px;
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        margin-bottom: 25px;
    }

    .title-text {
        font-size: 36px;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 5px;
    }

    .subtitle-text {
        font-size: 16px;
        color: #cbd5e1;
    }

    .metric-card {
        background: #1e293b;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #334155;
        text-align: center;
    }

    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #38bdf8;
    }

    .metric-label {
        font-size: 14px;
        color: #cbd5e1;
    }

    .info-box {
        background: #1e293b;
        padding: 18px;
        border-radius: 14px;
        border-left: 5px solid #38bdf8;
        color: #e2e8f0;
    }

    .warning-box {
        background: #422006;
        padding: 16px;
        border-radius: 14px;
        border-left: 5px solid #f59e0b;
        color: #fde68a;
    }

    .danger-box {
        background: #450a0a;
        padding: 16px;
        border-radius: 14px;
        border-left: 5px solid #ef4444;
        color: #fecaca;
    }

    section[data-testid="stSidebar"] {
        background-color: #020617;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="title-box">
    <div class="title-text">🛡️ AI-Based Server Log Anomaly Detection</div>
    <div class="subtitle-text">
        Upload any server/application log file and detect abnormal logs using Machine Learning.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Control Panel")

uploaded_file = st.sidebar.file_uploader(
    "Upload Log File",
    type=["log", "txt", "csv"]
)

sensitivity = st.sidebar.slider(
    "Anomaly Detection Sensitivity",
    min_value=0.05,
    max_value=0.50,
    value=0.20,
    step=0.05
)

show_only_anomalies = st.sidebar.checkbox("Show only anomalies")
search_text = st.sidebar.text_input("Search in logs")

st.sidebar.markdown("---")
st.sidebar.info(
    "Supported files: .log, .txt, .csv\n\n"
    "Model Used: TF-IDF + Isolation Forest"
)

# ---------------- FUNCTIONS ----------------
def read_uploaded_file(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)

        if len(df.columns) == 1:
            df.columns = ["log_message"]
        else:
            text_columns = df.select_dtypes(include=["object"]).columns

            if len(text_columns) > 0:
                df["log_message"] = df[text_columns].astype(str).agg(" ".join, axis=1)
            else:
                df["log_message"] = df.astype(str).agg(" ".join, axis=1)

        df = df[["log_message"]]

    else:
        content = file.read().decode("utf-8", errors="ignore")
        lines = content.splitlines()
        df = pd.DataFrame(lines, columns=["log_message"])

    df = df[df["log_message"].astype(str).str.strip() != ""]
    df.reset_index(drop=True, inplace=True)
    df["line_number"] = df.index + 1

    return df


def detect_log_level(message):
    msg = str(message).upper()

    if "CRITICAL" in msg or "FATAL" in msg:
        return "CRITICAL"
    elif "ERROR" in msg or "EXCEPTION" in msg:
        return "ERROR"
    elif "FAILED" in msg or "FAILURE" in msg:
        return "FAILED"
    elif "WARNING" in msg or "WARN" in msg:
        return "WARNING"
    elif "INFO" in msg:
        return "INFO"
    elif "DEBUG" in msg:
        return "DEBUG"
    else:
        return "UNKNOWN"


def detect_timestamp(message):
    patterns = [
        r"\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}",
        r"\d{2}/\d{2}/\d{4}\s\d{2}:\d{2}:\d{2}",
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    ]

    for pattern in patterns:
        match = re.search(pattern, str(message))
        if match:
            return match.group()

    return "N/A"


def severity_score(level):
    scores = {
        "CRITICAL": 5,
        "ERROR": 4,
        "FAILED": 4,
        "WARNING": 3,
        "UNKNOWN": 2,
        "INFO": 1,
        "DEBUG": 1
    }
    return scores.get(level, 2)


def analyze_logs(df, contamination):
    df["timestamp"] = df["log_message"].apply(detect_timestamp)
    df["log_level"] = df["log_message"].apply(detect_log_level)
    df["severity_score"] = df["log_level"].apply(severity_score)

    if len(df) < 5:
        df["ml_status"] = "Need More Logs"
        df["anomaly_score"] = 0
        df["risk_score"] = df["severity_score"] * 20
        return df

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=1000
    )

    features = vectorizer.fit_transform(df["log_message"].astype(str))

    model = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    predictions = model.fit_predict(features)
    anomaly_scores = model.decision_function(features)

    df["prediction"] = predictions
    df["anomaly_score"] = anomaly_scores
    df["ml_status"] = df["prediction"].apply(
        lambda x: "Anomaly" if x == -1 else "Normal"
    )

    min_score = df["anomaly_score"].min()
    max_score = df["anomaly_score"].max()

    if max_score != min_score:
        df["ml_risk"] = 100 - (
            (df["anomaly_score"] - min_score) / (max_score - min_score) * 100
        )
    else:
        df["ml_risk"] = 0

    df["risk_score"] = ((df["ml_risk"] * 0.7) + (df["severity_score"] * 20 * 0.3)).round(2)

    return df


def get_risk_label(score):
    if score >= 75:
        return "High Risk"
    elif score >= 45:
        return "Medium Risk"
    else:
        return "Low Risk"


# ---------------- MAIN APP ----------------
if uploaded_file is None:
    st.markdown("""
    <div class="info-box">
        <h3>📂 Upload a log file to start analysis</h3>
        <p>You can upload <b>.log</b>, <b>.txt</b>, or <b>.csv</b> files.</p>
        <p>The system will detect errors, warnings, failed jobs, critical logs, and ML-based anomalies.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Sample Log Format")
    st.code("""
2026-06-03 10:15:22 INFO Server started successfully
2026-06-03 10:16:10 WARNING High memory usage detected
2026-06-03 10:17:45 ERROR Database connection failed
2026-06-03 10:22:01 CRITICAL Server disk space almost full
2026-06-03 10:25:18 FAILED Batch job execution failed
    """)

else:
    try:
        df = read_uploaded_file(uploaded_file)
        result_df = analyze_logs(df, sensitivity)
        result_df["risk_label"] = result_df["risk_score"].apply(get_risk_label)

        total_logs = len(result_df)
        normal_logs = len(result_df[result_df["ml_status"] == "Normal"])
        anomalies = len(result_df[result_df["ml_status"] == "Anomaly"])
        critical_logs = len(result_df[result_df["log_level"] == "CRITICAL"])
        error_logs = len(result_df[result_df["log_level"] == "ERROR"])
        warning_logs = len(result_df[result_df["log_level"] == "WARNING"])

        st.markdown("## 📊 Analysis Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_logs}</div>
                <div class="metric-label">Total Logs</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{normal_logs}</div>
                <div class="metric-label">Normal Logs</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{anomalies}</div>
                <div class="metric-label">ML Anomalies</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{critical_logs}</div>
                <div class="metric-label">Critical Logs</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if critical_logs > 0 or anomalies > 0:
            st.markdown("""
            <div class="danger-box">
                ⚠️ Attention required: Critical logs or abnormal patterns were detected.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("No major critical issues detected.")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["🚨 Anomalies", "📄 Full Logs", "📈 Log Level Chart", "⬇️ Download Report"]
        )

        with tab1:
            st.subheader("Detected Anomalies")

            anomaly_df = result_df[result_df["ml_status"] == "Anomaly"]

            if search_text:
                anomaly_df = anomaly_df[
                    anomaly_df["log_message"].str.contains(search_text, case=False, na=False)
                ]

            if len(anomaly_df) > 0:
                st.dataframe(
                    anomaly_df[
                        [
                            "line_number",
                            "timestamp",
                            "log_level",
                            "ml_status",
                            "risk_label",
                            "risk_score",
                            "log_message"
                        ]
                    ],
                    use_container_width=True
                )
            else:
                st.success("No anomalies found based on current sensitivity.")

        with tab2:
            st.subheader("Complete Log Analysis")

            display_df = result_df.copy()

            if show_only_anomalies:
                display_df = display_df[display_df["ml_status"] == "Anomaly"]

            if search_text:
                display_df = display_df[
                    display_df["log_message"].str.contains(search_text, case=False, na=False)
                ]

            st.dataframe(
                display_df[
                    [
                        "line_number",
                        "timestamp",
                        "log_level",
                        "ml_status",
                        "risk_label",
                        "risk_score",
                        "log_message"
                    ]
                ],
                use_container_width=True
            )

        with tab3:
            st.subheader("Log Level Distribution")

            level_count = result_df["log_level"].value_counts().reset_index()
            level_count.columns = ["Log Level", "Count"]

            st.bar_chart(level_count.set_index("Log Level"))

            st.markdown("### Quick Counts")
            col5, col6, col7 = st.columns(3)
            col5.metric("Errors", error_logs)
            col6.metric("Warnings", warning_logs)
            col7.metric("Critical", critical_logs)

        with tab4:
            st.subheader("Download Analysis Report")

            csv_report = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download CSV Report",
                data=csv_report,
                file_name="ai_log_analysis_report.csv",
                mime="text/csv"
            )

            st.markdown("""
            <div class="info-box">
                Report includes line number, timestamp, log level, ML status, risk score, and complete log message.
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error("Something went wrong while analyzing the file.")
        st.code(str(e))