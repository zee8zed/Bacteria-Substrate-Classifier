---
title: Substrate vs Bacteria Classifier
emoji: 🦠
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
pinned: false
---

# Substrate vs Bacteria Classifier

A Gradio-based web application for classifying Raman/SERS spectral data as either **Substrate** or **Bacteria** using a trained Transformer model.

## Overview

This project provides a simple interface for uploading Raman/SERS spectral files and obtaining spectrum-level classification results. The application is intended for demonstration and evaluation of a trained deep learning model that distinguishes between substrate/background spectra and bacterial spectra.

## Supported File Formats

The application supports the following input file formats:

- `.txt`
- `.csv`
- `.xlsx`
- `.xls`

Each uploaded file should contain Raman/SERS spectral data in a format compatible with the preprocessing pipeline used during model development.

## Workflow

The application follows the general workflow below:

1. Upload a Raman/SERS spectral file.
2. Parse and preprocess the spectral data.
3. Apply the trained Transformer-based classification model.
4. Classify each spectrum as either **Substrate** or **Bacteria**.
5. Display the prediction results through the Gradio interface.

## Model Output

For each input spectrum, the model predicts one of the following classes:

- **Substrate**
- **Bacteria**

The output can be used to identify spectra that are likely to contain bacterial Raman/SERS signatures while separating them from substrate or background signals.

## Application

The application is built using [Gradio](https://www.gradio.app/) and is configured for deployment on Hugging Face Spaces.

## Project Files

Typical project files include:

```text
app.py              # Main Gradio application
README.md           # Project documentation
model files         # Trained Transformer model and related assets
preprocessing files # Supporting preprocessing utilities, if included
```

## Usage

Run the application locally with:

```bash
python app.py
```

After launching, open the local Gradio URL in a web browser, upload a supported Raman/SERS file, and view the predicted class labels.

## Notes

- The model should be used with spectral data that follows the same preprocessing assumptions used during training.
- Predictions are intended for research and demonstration purposes.
- Model performance should be validated on independent datasets before use in a production or diagnostic setting.

## License

Add the appropriate license information for this repository.
