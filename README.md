# Bioacoustics Neural Compression (EnCodec Branch)

*Note: The legacy Compressed Sensing implementation is located in the `main` branch. This specific branch focuses entirely on neural compression using Meta's EnCodec.*

This repository contains the research codebase for applying deep learning-based audio compression to bioacoustics monitoring. The primary goal is to compress environmental audio recordings (such as the sounds of Gibbons, Thyolo, and Pyntailed Wydah) while maintaining enough acoustic fidelity to successfully run classification and detection Convolutional Neural Networks (CNNs).

## EnCodec Compression Pipeline

This branch integrates **Meta's EnCodec** neural compression into the acoustic pipeline. This allows for various bitrates (e.g., 1.5kbps, 3kbps, 6kbps), which reduces the storage footprint of long-term ecological data.

The EnCodec implementation:
* Splits the audio into 1-second chunks (with a 10ms overlap to prevent edge artifacts).
* Processes the audio entirely on the CPU (optimized for resource-constrained environments).
* Encodes the data into latent spaces (`.pt` files) or reconstructs them as `.wav` files to be fed directly into the CNN models.
* Re-adjusts the dimensions of the input arrays so the CNN layers are fully compatible with the EnCodec outputs.

## Setup Instructions

To run the pipeline locally or on a new machine, install the exact Python dependencies required for the project.

1. (Optional but recommended) Create a virtual environment using a preferred tool (e.g. `venv`, `conda`).
2. Install the required dependencies from the `requirements.txt` file by running:
   ```bash
   pip install -r requirements.txt
   ```
   *(This will install necessary packages including `torch`, `encodec`, `librosa`, etc.)*

## How to Run the Pipeline

The core logic is stored within the `src/` directory, but the easiest way to interact with the code is through the provided Jupyter Notebooks located in the `notebooks/` directory.

- **`notebooks/encodec_pipeline_v2.ipynb`**: The main pipeline for applying the EnCodec compression to the raw dataset.
- **`notebooks/train_CNN_encodec.ipynb`**: Used to train the CNN using the newly compressed EnCodec data.
- **`notebooks/encodec_comparative_analysis.ipynb`**: Runs experiments across different bitrates to analyze the trade-off between file size and model accuracy.


