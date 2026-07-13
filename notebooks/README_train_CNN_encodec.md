# EnCodec CNN Training Pipeline (`train_CNN_encodec.ipynb`)

This notebook serves as the core training and evaluation engine for testing Convolutional Neural Network (CNN) performance on bioacoustic species identification using audio compressed and reconstructed via Meta's **EnCodec** model.

## Prerequisites
Before running this notebook, `encodec_pipeline_v2.ipynb` must be executed to handle the audio compression and the extraction of `.pkl` dataset features. Once the datasets are prepared, this notebook handles the downstream training and inference.

## Automation & Workflow
The notebook is fully automated to perform the following:
1. Iterate over the defined EnCodec bandwidths (e.g., `1.5`, `3.0`, `6.0`).
2. Execute complete training loops for each bandwidth to generate statistically significant F1-scores.
3. Dynamically configure paths and network architectures based on the specified `TARGET_SPECIES` variable.

*(Note: The `main_vs2.ipynb` notebook is a legacy benchmarking script for standard codecs like AAC, MP3, and FLAC, and is independent of this EnCodec pipeline.)*

---

## Data Leakage & Split Integrity

**Are `test` and `val` sets used during training?**  
**No.** The 3-way split (Train / Val / Test) is strictly enforced and isolated.

- **Training (`X_train`):** Used exclusively by the CNN for gradient updates and weight learning.
- **Validation (`X_val`):** Used exclusively during the training loop for **Early Stopping** (evaluating validation loss) to prevent model overfitting. It is never used to update model weights.
- **Testing (`test_type="testing_dataset"` & `test_type="entire_files"`):** The test datasets are strictly excluded from the `.train()` cycle. They are loaded exclusively for inference (`evaluation.run()`) after the optimal model state has been saved to disk.

---

## Notebook Structure

### 1. Configuration & Architecture Setup
Configuration is strictly managed at the beginning of the notebook via the target species definition:
```python
TARGET_SPECIES = "PTW" # or "Thyolo", "Gibbon", etc.
```
This ensures the script dynamically retrieves the correct CNN architecture and audio metrics tailored to the designated species.

### 2. Training Phase
Within the main loop, the notebook initializes the model and begins the training sequence:
```python
model.train(X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val, ...)
```
This phase relies exclusively on the `.pkl` array files generated during the prior dataset creation step.

### 3. Evaluation Phase
Two distinct evaluation methodologies are applied to the unseen **Test** set:

#### A. Segmented Testing Dataset
Following training, the model evaluates the specific sliced spectrograms extracted from the test set:
```python
f1_score = evaluation.run(model, test_type="testing_dataset")[0]
```

#### B. Continuous Audio "Entire Files" Evaluation
To simulate real-world deployment conditions, the model is evaluated against continuous, uncut `.wav` files. This step utilizes the reconstructed EnCodec `.wav` files located in the `Compressed_Audio` directory to generate `_predictions.svl` files and the final `_f1score_csv.csv` outputs:
```python
f1_score_full = evaluation.run(model, test_type="entire_files", preprocessing_arg=True)[0]
```
