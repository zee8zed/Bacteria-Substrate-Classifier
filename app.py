import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
import gradio as gr
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Model path (relative — the .keras file must be uploaded into this same
#    Space repo, e.g. alongside app.py, or in a subfolder you reference here)
# ---------------------------------------------------------------------------
MODEL_PATH = "Trained_Transformer_Model.keras"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        "Make sure it has been uploaded to the Space repo."
    )

# ---------------------------------------------------------------------------
# 2. Custom layers (unchanged from your notebook)
# ---------------------------------------------------------------------------
@keras.saving.register_keras_serializable(package="Custom")
class TokenPositionalEmbedding(layers.Layer):
    def __init__(self, maxlen, d_model, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.d_model = d_model
        self.pos_emb = layers.Embedding(
            input_dim=maxlen,
            output_dim=d_model,
            name="pos_emb"
        )

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = self.pos_emb(positions)
        return x + positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.maxlen,
            "d_model": self.d_model
        })
        return config


@keras.saving.register_keras_serializable(package="Custom")
class TinyTransformerBlock(layers.Layer):
    def __init__(
        self,
        d_model=None,
        num_heads=None,
        ffn_dim=None,
        ff_dim=None,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        if ffn_dim is None:
            ffn_dim = ff_dim

        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout

        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            name="mha"
        )

        self.ffn = keras.Sequential(
            [
                layers.Dense(ffn_dim, activation="relu"),
                layers.Dense(d_model),
            ],
            name="ffn"
        )

        self.ln1 = layers.LayerNormalization(epsilon=1e-6, name="ln1")
        self.ln2 = layers.LayerNormalization(epsilon=1e-6, name="ln2")

        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        attn_output = self.mha(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)

        out1 = self.ln1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        return self.ln2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout
        })
        return config


# ---------------------------------------------------------------------------
# 3. Load the model once, at Space startup
# ---------------------------------------------------------------------------
custom_objects = {
    "TokenPositionalEmbedding": TokenPositionalEmbedding,
    "TinyTransformerBlock": TinyTransformerBlock,
    "Custom>TokenPositionalEmbedding": TokenPositionalEmbedding,
    "Custom>TinyTransformerBlock": TinyTransformerBlock,
}

model = keras.models.load_model(
    MODEL_PATH,
    custom_objects=custom_objects,
    compile=False,
    safe_mode=False
)

CLASS_0 = "Substrate"
CLASS_1 = "Bacteria"
THRESHOLD = 0.5
TARGET_FEATURES = int(model.input_shape[1])


# ---------------------------------------------------------------------------
# 4. File readers (unchanged from your notebook)
# ---------------------------------------------------------------------------
def extract_numbers(line):
    return re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)


def read_horiba_raman_txt(file_path, target_features=500):
    with open(file_path, "rb") as f:
        raw = f.read()

    text = raw.decode("latin1", errors="ignore")
    lines = text.splitlines()

    header_idx = None
    wavenumbers = None

    for i, line in enumerate(lines):
        if line.startswith("#"):
            continue

        nums = extract_numbers(line)

        if len(nums) > 100:
            values = np.array([float(x) for x in nums], dtype="float32")

            if 500 <= values[0] <= 600 and 2100 <= values[-1] <= 2300:
                header_idx = i
                wavenumbers = values
                break

    if header_idx is None:
        raise ValueError(
            "Could not find the wavenumber header row. "
            "Please check if this is a Horiba Raman TXT file."
        )

    expected_len = len(wavenumbers)

    spectra = []
    coordinates = []

    for line in lines[header_idx + 1:]:
        if not line.strip():
            continue

        if line.startswith("#"):
            continue

        nums = extract_numbers(line)

        if len(nums) >= expected_len + 2:
            values = np.array(
                [float(x) for x in nums[:expected_len + 2]],
                dtype="float32"
            )

            coordinates.append(values[:2])
            spectra.append(values[2:])

    if len(spectra) == 0:
        raise ValueError("No spectral intensity rows found in the TXT file.")

    X_raw = np.array(spectra, dtype="float32")
    coordinates = np.array(coordinates, dtype="float32")

    original_spectra_count = X_raw.shape[0]
    original_features = X_raw.shape[1]

    nonzero_mask = np.sum(np.abs(X_raw), axis=1) > 0

    X_raw = X_raw[nonzero_mask]
    coordinates = coordinates[nonzero_mask]

    removed_zero_spectra = original_spectra_count - X_raw.shape[0]

    if X_raw.shape[0] == 0:
        raise ValueError("All spectra are zero. No usable spectra found.")

    if original_features != target_features:
        target_wavenumbers = np.linspace(
            wavenumbers.min(),
            wavenumbers.max(),
            target_features
        ).astype("float32")

        X_resampled = np.array(
            [
                np.interp(target_wavenumbers, wavenumbers, row)
                for row in X_raw
            ],
            dtype="float32"
        )
    else:
        X_resampled = X_raw

    X_model = np.expand_dims(X_resampled, axis=-1)

    info = {
        "original_spectra_count": int(original_spectra_count),
        "used_spectra_count": int(X_model.shape[0]),
        "removed_zero_spectra": int(removed_zero_spectra),
        "original_features": int(original_features),
        "model_features": int(target_features),
        "wavenumber_start": float(wavenumbers.min()),
        "wavenumber_end": float(wavenumbers.max())
    }

    return X_model, info


def read_generic_table_file(file_path, target_features=500):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, encoding="latin1")
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type for generic table reader.")

    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    remove_cols = [
        "X", "Y", "x", "y",
        "X_coord", "Y_coord", "x_coord", "y_coord",
        "Label", "label", "Class", "class"
    ]

    df = df.drop(columns=[c for c in remove_cols if c in df.columns], errors="ignore")

    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.dropna(axis=0, how="all")
    numeric_df = numeric_df.dropna(axis=1, how="all")
    numeric_df = numeric_df.fillna(0)

    if numeric_df.shape[1] == 0:
        raise ValueError("No numeric spectral columns found.")

    X_raw = numeric_df.values.astype("float32")

    nonzero_mask = np.sum(np.abs(X_raw), axis=1) > 0
    X_raw = X_raw[nonzero_mask]

    if X_raw.shape[0] == 0:
        raise ValueError("All rows are zero after cleaning.")

    original_features = X_raw.shape[1]

    if original_features != target_features:
        old_axis = np.linspace(0, 1, original_features)
        new_axis = np.linspace(0, 1, target_features)

        X_resampled = np.array(
            [
                np.interp(new_axis, old_axis, row)
                for row in X_raw
            ],
            dtype="float32"
        )
    else:
        X_resampled = X_raw

    X_model = np.expand_dims(X_resampled, axis=-1)

    info = {
        "original_spectra_count": int(numeric_df.shape[0]),
        "used_spectra_count": int(X_model.shape[0]),
        "removed_zero_spectra": int(numeric_df.shape[0] - X_model.shape[0]),
        "original_features": int(original_features),
        "model_features": int(target_features),
        "wavenumber_start": None,
        "wavenumber_end": None
    }

    return X_model, info


def read_and_prepare_file(file_path, target_features=500):
    file_path_lower = file_path.lower()

    if file_path_lower.endswith(".txt"):
        return read_horiba_raman_txt(file_path, target_features=target_features)
    elif file_path_lower.endswith(".csv") or file_path_lower.endswith(".xlsx") or file_path_lower.endswith(".xls"):
        return read_generic_table_file(file_path, target_features=target_features)
    else:
        raise ValueError("Please upload a TXT, CSV, XLSX, or XLS file.")


# ---------------------------------------------------------------------------
# 5. Plotting + prediction (unchanged from your notebook)
# ---------------------------------------------------------------------------
def make_probability_histogram(values, title, xlabel):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=20, alpha=0.8, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    return fig


def make_percentage_bar_plot(class_0_count, class_1_count, total):
    class_0_percent = (class_0_count / total) * 100
    class_1_percent = (class_1_count / total) * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([CLASS_0, CLASS_1], [class_0_percent, class_1_percent])
    ax.set_title("Percentage of Samples per Predicted Class")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)

    for i, value in enumerate([class_0_percent, class_1_percent]):
        ax.text(i, value + 1, f"{value:.2f}%", ha="center")

    ax.grid(True, axis="y", alpha=0.3)
    return fig


def predict_substrate_vs_bacteria(file):
    try:
        if file is None:
            return "Please upload a file first.", None, None, None, None

        # Gradio may hand back either a filepath string or an object with .name
        file_path = file if isinstance(file, str) else file.name

        X, info = read_and_prepare_file(
            file_path,
            target_features=TARGET_FEATURES
        )

        probabilities = model.predict(X, verbose=0).flatten()

        bacteria_probabilities = probabilities * 100
        substrate_probabilities = (1 - probabilities) * 100

        predicted_labels = []
        confidences = []

        for p in probabilities:
            p = float(p)
            if p >= THRESHOLD:
                predicted_labels.append(CLASS_1)
                confidences.append(p)
            else:
                predicted_labels.append(CLASS_0)
                confidences.append(1.0 - p)

        result_df = pd.DataFrame({
            "Spectrum Number": np.arange(1, len(predicted_labels) + 1),
            "Predicted Class": predicted_labels,
            "Bacteria Probability (%)": bacteria_probabilities,
            "Substrate Probability (%)": substrate_probabilities,
            "Confidence (%)": np.array(confidences) * 100
        })

        class_0_count = predicted_labels.count(CLASS_0)
        class_1_count = predicted_labels.count(CLASS_1)
        total = len(predicted_labels)

        mean_probability = float(np.mean(probabilities))

        if mean_probability >= THRESHOLD:
            final_prediction_by_probability = CLASS_1
            final_probability_confidence = mean_probability
        else:
            final_prediction_by_probability = CLASS_0
            final_probability_confidence = 1.0 - mean_probability

        if class_1_count > class_0_count:
            final_prediction_by_majority = CLASS_1
            majority_confidence = class_1_count / total
        elif class_0_count > class_1_count:
            final_prediction_by_majority = CLASS_0
            majority_confidence = class_0_count / total
        else:
            final_prediction_by_majority = "Tie"
            majority_confidence = 0.5

        if info["wavenumber_start"] is not None:
            wavenumber_text = f'{info["wavenumber_start"]:.2f} to {info["wavenumber_end"]:.2f} cm⁻¹'
        else:
            wavenumber_text = "Not available"

        summary = f"""
Final File-Level Prediction: {final_prediction_by_majority}

Majority Confidence: {majority_confidence * 100:.2f}%

Prediction by Mean Probability: {final_prediction_by_probability}
Mean Probability Confidence: {final_probability_confidence * 100:.2f}%

Total spectra detected in file: {info["original_spectra_count"]}
Zero spectra removed: {info["removed_zero_spectra"]}
Spectra used for prediction: {info["used_spectra_count"]}

Original spectral features: {info["original_features"]}
Model input features: {info["model_features"]}

Wavenumber range: {wavenumber_text}

{CLASS_0} spectra: {class_0_count}
{CLASS_1} spectra: {class_1_count}

Mean bacteria probability: {mean_probability * 100:.2f}%
Mean substrate probability: {(1 - mean_probability) * 100:.2f}%
Decision threshold: {THRESHOLD}
"""

        bacteria_hist_fig = make_probability_histogram(
            bacteria_probabilities,
            "Predicted Probabilities for Bacteria",
            "Probability (%)"
        )

        substrate_hist_fig = make_probability_histogram(
            substrate_probabilities,
            "Predicted Probabilities for Substrate",
            "Probability (%)"
        )

        percentage_bar_fig = make_percentage_bar_plot(
            class_0_count,
            class_1_count,
            total
        )

        return summary, result_df, bacteria_hist_fig, substrate_hist_fig, percentage_bar_fig

    except Exception as e:
        return f"Error: {str(e)}", None, None, None, None


# ---------------------------------------------------------------------------
# 6. Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Substrate vs Bacteria Classifier") as app:

    gr.Markdown("# Substrate vs Bacteria Classifier")

    gr.Markdown(
        """
Upload a Raman/SERS spectral file.

Supported formats:
- Horiba/XploRA `.txt`
- `.csv`
- `.xlsx`
- `.xls`

The model classifies each spectrum as **Substrate** or **Bacteria**.
"""
    )

    file_input = gr.File(
        label="Upload TXT / CSV / Excel file",
        file_types=[".txt", ".csv", ".xlsx", ".xls"]
    )

    predict_button = gr.Button("Predict")

    summary_output = gr.Textbox(
        label="Prediction Summary",
        lines=16
    )

    table_output = gr.Dataframe(
        label="Per-Spectrum Predictions",
        wrap=True
    )

    gr.Markdown("## Prediction Graphs")

    bacteria_plot = gr.Plot(label="Predicted Probabilities for Bacteria")
    substrate_plot = gr.Plot(label="Predicted Probabilities for Substrate")
    percentage_plot = gr.Plot(label="Percentage of Samples per Predicted Class")

    predict_button.click(
        fn=predict_substrate_vs_bacteria,
        inputs=file_input,
        outputs=[
            summary_output,
            table_output,
            bacteria_plot,
            substrate_plot,
            percentage_plot
        ]
    )

# No share=True / debug=True needed — Hugging Face Spaces hosts it for you
if __name__ == "__main__":
    app.launch()
