
from pathlib import Path
import os
import re
import subprocess
import sys

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import f1_score


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Encrypted Traffic Classification",
    page_icon="🌐",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
TRAINING_SCRIPT = ROOT / "train_self_supervised.py"

st.title("Encrypted Traffic Classification")
st.subheader("Self-Supervised Learning Under Network Distribution Shifts")

st.write(
    "Compare a supervised temporal baseline with an "
    "autoencoder-based representation learning pipeline."
)


# --------------------------------------------------
# 2. CHECK REQUIRED FILES
# --------------------------------------------------
summary_path = RESULTS / "validated_model_comparison.csv"
baseline_path = RESULTS / "fair_temporal_baseline_predictions.csv"
original_proposed_path = RESULTS / "self_supervised_predictions.csv"

required_files = [
    summary_path,
    baseline_path,
    original_proposed_path,
    TRAINING_SCRIPT,
]

missing_files = [str(path) for path in required_files if not path.is_file()]

if missing_files:
    st.error("Required project files are missing:")
    for filename in missing_files:
        st.write(f"- {filename}")
    st.stop()

try:
    summary = pd.read_csv(summary_path)
    baseline = pd.read_csv(baseline_path)
except Exception as error:
    st.error(f"Could not read a required results file: {error}")
    st.stop()

required_summary_columns = {
    "Model", "Accuracy (%)", "Macro-F1 (%)"
}

if not required_summary_columns.issubset(summary.columns):
    st.error("The validated model comparison CSV has unexpected columns.")
    st.stop()

required_prediction_columns = {"actual", "predicted"}

if not required_prediction_columns.issubset(baseline.columns):
    st.error("The baseline predictions CSV has unexpected columns.")
    st.stop()

if baseline.empty:
    st.error("The baseline predictions file contains no rows.")
    st.stop()


# --------------------------------------------------
# 3. RUN A NEW EXPERIMENT
# --------------------------------------------------
st.header("1. Run a New Experiment")

st.write(
    "Select the number of autoencoder training epochs and click "
    "**Run Experiment** to train the model."
)

epochs = st.slider(
    "Autoencoder training epochs",
    min_value=5,
    max_value=30,
    value=10,
    step=5,
    key="epoch_slider",
)

st.caption(
    "Each epoch setting saves separate result files. "
    "Your original validated results will be kept separate."
)

if st.button("Run Experiment", type="primary"):
    with st.spinner(
        f"Training the autoencoder for {epochs} epochs. Please wait..."
    ):
        environment = os.environ.copy()
        environment["AE_EPOCHS"] = str(epochs)

        expected_report = (
            RESULTS / f"self_supervised_{epochs}epochs_report.txt"
        )
        expected_predictions = (
            RESULTS / f"self_supervised_{epochs}epochs_predictions.csv"
        )
        expected_weights = (
            RESULTS / f"autoencoder_{epochs}epochs_weights.pt"
        )

        expected_outputs = (
            expected_report,
            expected_predictions,
            expected_weights,
        )

        # Record existing file modification times before training.
        previous_mtimes = {
            path: path.stat().st_mtime_ns if path.is_file() else None
            for path in expected_outputs
        }

        st.session_state["last_run_epochs"] = epochs
        st.session_state["last_run_success"] = False

        try:
            completed = subprocess.run(
                [sys.executable, str(TRAINING_SCRIPT)],
                cwd=str(ROOT),
                env=environment,
                capture_output=True,
                text=True,
                timeout=1800,
                check=False,
            )

            output_text = completed.stdout or ""

            if completed.stderr:
                output_text += "\n\nSTDERR:\n" + completed.stderr

            if completed.returncode != 0:
                output_text += (
                    f"\n\nTraining exited with code "
                    f"{completed.returncode}."
                )

            missing_outputs = [
                str(path)
                for path in expected_outputs
                if not path.is_file()
            ]

            stale_outputs = [
                str(path)
                for path in expected_outputs
                if path.is_file()
                and previous_mtimes[path] is not None
                and path.stat().st_mtime_ns <= previous_mtimes[path]
            ]

            run_success = (
                completed.returncode == 0
                and not missing_outputs
                and not stale_outputs
            )

            if missing_outputs:
                output_text += (
                    "\n\nMissing expected output files:\n"
                    + "\n".join(missing_outputs)
                )

            if stale_outputs:
                output_text += (
                    "\n\nOutput files were not freshly updated:\n"
                    + "\n".join(stale_outputs)
                )

            st.session_state["last_run_success"] = run_success
            st.session_state["last_run_output"] = output_text

        except subprocess.TimeoutExpired as error:
            partial_output = error.stdout or ""
            if isinstance(partial_output, bytes):
                partial_output = partial_output.decode(
                    "utf-8", errors="replace"
                )

            st.session_state["last_run_output"] = (
                "Training exceeded the 30-minute time limit.\n\n"
                + partial_output
            )

        except Exception as error:
            st.session_state["last_run_output"] = str(error)



if "last_run_success" in st.session_state:
    if st.session_state["last_run_success"]:
        st.success(
            f"The {st.session_state['last_run_epochs']}-epoch "
            "experiment finished successfully, and all expected "
            "output files exist."
        )
    else:
        st.error(
            "The experiment failed or one or more expected output "
            "files are missing. Check the training output below."
        )

    with st.expander("View training output"):
        st.code(st.session_state["last_run_output"])


# --------------------------------------------------
# 4. SELECT PREDICTIONS TO DISPLAY
# --------------------------------------------------
active_predictions_path = original_proposed_path
active_experiment_name = "Original validated experiment"

if (
    st.session_state.get("last_run_success", False)
    and "last_run_epochs" in st.session_state
):
    last_epochs = st.session_state["last_run_epochs"]

    candidate_predictions = (
        RESULTS / f"self_supervised_{last_epochs}epochs_predictions.csv"
    )

    if candidate_predictions.is_file():
        active_predictions_path = candidate_predictions
        active_experiment_name = (
            f"Latest {last_epochs}-epoch experiment"
        )

try:
    proposed = pd.read_csv(active_predictions_path)
except Exception as error:
    st.error(f"Could not read the proposed predictions file: {error}")
    st.stop()

if not required_prediction_columns.issubset(proposed.columns):
    st.error(
        "The selected proposed predictions CSV has unexpected columns."
    )
    st.stop()

if proposed.empty:
    st.error("The selected proposed predictions file contains no rows.")
    st.stop()


# --------------------------------------------------
# 5. OVERALL MODEL PERFORMANCE
# --------------------------------------------------
st.header("2. Overall Model Performance")

baseline_row = summary[
    summary["Model"] == "Fair Temporal Baseline"
]

proposed_row = summary[
    summary["Model"] == "Self-Supervised Seed 42"
]

if baseline_row.empty or proposed_row.empty:
    st.error("Expected model names were not found in the results file.")
    st.stop()

b = baseline_row.iloc[0]
p = proposed_row.iloc[0]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Baseline Accuracy", f'{b["Accuracy (%)"]:.2f}%')
col2.metric("Original Proposed Accuracy", f'{p["Accuracy (%)"]:.2f}%')
col3.metric("Baseline Macro-F1", f'{b["Macro-F1 (%)"]:.2f}%')
col4.metric("Original Proposed Macro-F1", f'{p["Macro-F1 (%)"]:.2f}%')

st.caption(
    "These are the original validated results. "
    "Newly run experiment results are shown separately below."
)

st.subheader("Validated Model Comparison")

chart_data = summary.melt(
    id_vars="Model",
    value_vars=["Accuracy (%)", "Macro-F1 (%)"],
    var_name="Metric",
    value_name="Score",
)

fig = px.bar(
    chart_data,
    x="Model",
    y="Score",
    color="Metric",
    barmode="group",
    text="Score",
    title="Accuracy vs Macro-F1",
    labels={
        "Model": "Model",
        "Score": "Score (%)",
        "Metric": "Metric",
    },
    hover_data={"Score": ":.2f"},
)

fig.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside",
    cliponaxis=False,
)

fig.update_layout(
    yaxis_title="Score (%)",
    xaxis_title="",
    yaxis_range=[0, 108],
    legend_title="Metric",
    xaxis_tickangle=-12,
    margin=dict(l=20, r=20, t=70, b=30),
)

st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# 6. LATEST EXPERIMENT RESULTS
# --------------------------------------------------
st.header("3. Latest Experiment Results")

if (
    st.session_state.get("last_run_success", False)
    and "last_run_epochs" in st.session_state
):
    last_epochs = st.session_state["last_run_epochs"]

    report_path = (
        RESULTS / f"self_supervised_{last_epochs}epochs_report.txt"
    )

    if report_path.is_file():
        try:
            report_text = report_path.read_text(encoding="utf-8")
        except OSError as error:
            st.error(f"Could not read the experiment report: {error}")
            st.stop()

        accuracy = None
        macro_f1 = None
        final_loss = None

        for line in report_text.splitlines():
            lowered = line.lower()

            if ":" not in line:
                continue

            value_text = line.split(":", 1)[1].strip()

            if not value_text:
                continue

            numbers = re.findall(
                r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?",
                value_text,
            )

            if not numbers:
                continue

            try:
                if "accuracy" in lowered:
                    accuracy = float(numbers[0])

                elif "macro" in lowered and "f1" in lowered:
                    macro_f1 = float(numbers[0])

                elif "reconstruction" in lowered and "loss" in lowered:
                    final_loss = float(numbers[-1])

            except (ValueError, IndexError):
                continue

        metric_cols = st.columns(3)

        if accuracy is not None:
            accuracy_percent = (
                accuracy * 100 if accuracy <= 1 else accuracy
            )
            metric_cols[0].metric(
                "Latest Accuracy",
                f"{accuracy_percent:.2f}%",
            )
        else:
            metric_cols[0].metric("Latest Accuracy", "See report")

        if macro_f1 is not None:
            macro_percent = (
                macro_f1 * 100 if macro_f1 <= 1 else macro_f1
            )
            metric_cols[1].metric(
                "Latest Macro-F1",
                f"{macro_percent:.2f}%",
            )
        else:
            metric_cols[1].metric("Latest Macro-F1", "See report")

        if final_loss is not None:
            metric_cols[2].metric(
                "Final Reconstruction Loss",
                f"{final_loss:.6f}",
            )
        else:
            metric_cols[2].metric(
                "Final Reconstruction Loss",
                "See report",
            )

        with st.expander("View full experiment report"):
            st.text(report_text)

        if accuracy is None or macro_f1 is None or final_loss is None:
            st.info(
                "Some metrics could not be extracted automatically. "
                "Open the full report above to inspect the exact values."
            )

    else:
        st.warning(
            f"Report not found: {report_path.name}. "
            "Check the training output above."
        )
else:
    st.info(
        "Run an experiment above to display its metrics here. "
        "The original validated results remain available."
    )


# --------------------------------------------------
# 7. CLASS-WISE PERFORMANCE
# --------------------------------------------------
st.header("4. Class-wise Performance")

st.caption(
    f"Proposed predictions displayed: {active_experiment_name}"
)

if (
    len(baseline) != len(proposed)
    or not baseline["actual"].equals(proposed["actual"])
):
    st.error(
        "The baseline and proposed prediction files do not contain "
        "the same test labels in the same order."
    )
    st.stop()

classes = [
    "audio-streaming",
    "browsing",
    "file-transfer",
    "p2p",
]

baseline_f1 = f1_score(
    baseline["actual"],
    baseline["predicted"],
    labels=classes,
    average=None,
    zero_division=0,
)

proposed_f1 = f1_score(
    proposed["actual"],
    proposed["predicted"],
    labels=classes,
    average=None,
    zero_division=0,
)

class_results = pd.DataFrame({
    "Traffic class": classes,
    "Temporal baseline F1": baseline_f1,
    "Proposed model F1": proposed_f1,
})

st.dataframe(
    class_results.style.format({
        "Temporal baseline F1": "{:.4f}",
        "Proposed model F1": "{:.4f}",
    }),
    width="stretch",
    hide_index=True,
)

class_chart_data = class_results.melt(
    id_vars="Traffic class",
    value_vars=["Temporal baseline F1", "Proposed model F1"],
    var_name="Model",
    value_name="F1 score",
)

class_fig = px.bar(
    class_chart_data,
    x="Traffic class",
    y="F1 score",
    color="Model",
    barmode="group",
    text="F1 score",
    title="Class-wise F1 Comparison",
    labels={
        "Traffic class": "Traffic class",
        "F1 score": "F1 score",
        "Model": "Model",
    },
    hover_data={"F1 score": ":.4f"},
)

class_fig.update_traces(
    texttemplate="%{text:.3f}",
    textposition="outside",
    cliponaxis=False,
)

class_fig.update_layout(
    yaxis_title="F1 score",
    xaxis_title="",
    yaxis_range=[0, 1.12],
    legend_title="Model",
    xaxis_tickangle=-15,
    margin=dict(l=20, r=20, t=70, b=30),
)

st.plotly_chart(class_fig, use_container_width=True)


# --------------------------------------------------
# 8. EXPERIMENTAL SETUP AND LIMITATIONS
# --------------------------------------------------
st.header("5. Experimental Setup")

st.write("- Dataset: CIC-Darknet2020 (Darknet.CSV)")
st.write("- Training period: before 24 February 2016")
st.write("- Test period: 24 February 2016 onward")
st.write("- Common traffic classes: four")
st.write("- Original numeric features: 64")
st.write("- Autoencoder embedding dimension: 16")
st.write("- Random seeds evaluated in the original experiments: 42 and 123")

st.warning(
    "The test classes are highly imbalanced. Audio-streaming and "
    "file-transfer remain difficult to classify. The baseline uses "
    "original features, while the proposed classifier uses learned embeddings."
)

st.caption(
    "Research prototype — results reflect this dataset and temporal split; "
    "they do not establish generalization to every network environment."
)
