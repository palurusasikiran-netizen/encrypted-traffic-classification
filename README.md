# Encrypted Traffic Classification Under Network Distribution Shifts

**Course:** Computer Networks Lab — Innovative Evaluation Practices
**Institution:** Vellore Institute of Technology, School of Computer Science and Engineering
**Semester:** Fall Semester 2026–27
**Team:** F2 — Team 28

## 1. Project Overview

This project studies encrypted network traffic classification when the traffic distribution changes over time. It compares:

1. **Temporal supervised baseline:** a `HistGradientBoostingClassifier` trained on the original preprocessed numeric traffic features.
2. **Self-supervised pipeline:** an autoencoder learns a compact representation of the training traffic without using class labels during representation learning; a `HistGradientBoostingClassifier` then classifies those learned embeddings.

The experiment uses the CIC-Darknet2020 `Darknet.CSV` dataset and a chronological train/test split. It is a research prototype evaluated on this dataset and split; it does not establish generalization to every network or live deployment environment.

## 2. Main Results

The project keeps the original validated comparison separate from a later dashboard run. Do not combine these result sets as though they were the same experiment.

### Original validated comparison

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Fair temporal baseline | 77.81% | 38.57% |
| Self-supervised pipeline, seed 42 | 86.47% | 48.81% |
| Self-supervised pipeline, seed 123 | 86.32% | 48.43% |

For seed 42, the self-supervised pipeline improved accuracy by about **8.66 percentage points** and Macro-F1 by about **10.24 percentage points** compared with the fair temporal baseline.

### Separate latest dashboard run (10 epochs)

| Metric | Value |
|---|---:|
| Accuracy | 86.66% |
| Macro-F1 | 48.92% |
| Final autoencoder reconstruction loss | 0.060400 |

These dashboard-run values are a separate run and should not replace the original validated values in comparisons unless the associated output files are explicitly used and validated.

### Interpretation and limitation

The test set is highly imbalanced. In the original seed-42 comparison, F1 scores were approximately:

| Traffic class | Baseline F1 | Self-supervised F1 |
|---|---:|---:|
| audio-streaming | 0.0132 | 0.0935 |
| browsing | 0.6541 | 0.9268 |
| file-transfer | 0.0155 | 0.0155 |
| p2p | 0.8598 | 0.9163 |

The file-transfer class remains difficult to classify, and audio-streaming performance is also weak. Accuracy alone therefore does not describe performance across all classes; Macro-F1 and class-wise F1 are important when interpreting the results.

## 3. Dataset

- **Expected file:** `data/Darknet.CSV`
- **Dataset:** CIC-Darknet2020
- **Original shape:** 158,616 rows × 85 columns
- **Numeric traffic features:** 64 after excluding identifiers, timestamps and label columns
- **Time range observed:** 30 March 2015 to 25 February 2016

The project uses four traffic classes that are represented in both temporal periods: `audio-streaming`, `browsing`, `file-transfer`, and `p2p`.

### Dataset access and responsible use

The dataset is not included in this repository by default. Obtain it from the dataset provider and place the CSV at `data/Darknet.CSV`, after checking the provider's terms and any redistribution restrictions. Do not commit the dataset unless you have confirmed that redistribution is permitted.

## 4. Experimental Design

### Chronological split

The temporal cutoff is **24 February 2016**:

- Training period: timestamps before 24 February 2016
- Test period: 24 February 2016 onward
- Training rows: 60,778
- Test rows: 66,731

The four common traffic classes are used for the main temporal comparison. This split is intended to examine performance under a change in traffic distribution over time.

### Preprocessing

The main temporal pipeline:
1. Normalizes class labels and parses timestamps.
2. Excludes identifiers, IP addresses, timestamp and label columns from the numeric feature matrix.
3. Replaces infinite values and imputes missing values using statistics fitted on training data.
4. Removes features that are constant in the training set.
5. Standardizes using training-set statistics and clips standardized values to `[-10, 10]`.
6. Applies the fitted preprocessing to the test set.

Fitting preprocessing on the training period avoids using test-set statistics during training.

### Models

**Fair temporal baseline**
- Input: preprocessed numeric traffic features
- Classifier: `HistGradientBoostingClassifier`
- Main configuration: `max_iter=100`, `learning_rate=0.1`, `max_leaf_nodes=31`, `random_state=42`

**Self-supervised pipeline**
- Autoencoder architecture: input → 32 → 16 latent dimensions → 32 → input
- Activations: ReLU
- Optimizer: Adam, learning rate `0.001`
- Reconstruction objective: mean squared error (MSE)
- Batch size: `512`
- Dashboard default experiment: `10` epochs
- The autoencoder is trained on training-period features without class labels for the representation-learning step.
- A `HistGradientBoostingClassifier` is trained on the learned embeddings using the training labels.

The dashboard also displays the original validated results and the latest run separately.

## 5. Repository Structure

The following is the intended high-level structure. The exact filenames in `results/` depend on the scripts that have been run.

```text
EncryptedTrafficProject/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── Darknet.CSV              # obtain separately; do not commit without permission
├── results/
│   ├── validated_model_comparison.csv
│   ├── final_model_comparison_verified.csv
│   ├── validated_model_comparison.png
│   ├── validated_class_f1_comparison.png
│   └── ...                      # reports, predictions and other outputs
├── experiments/                 # optional: exploratory/alternative analysis scripts
└── *.py                         # training, validation and analysis scripts
```

Keep the existing scripts and results. Before publishing, check that the main report's numbers correspond to the validated result files and that exploratory experiments are clearly identified as exploratory.

## 6. Setup

Use Python 3.12 if possible, matching the development environment. From a terminal opened in the project directory:

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks environment activation, use the Python executable inside `.venv` directly or follow your institution's approved PowerShell policy; do not weaken system security settings just to activate the environment.

### Dataset placement

Create a `data` folder if it does not exist, then place the dataset file at:

```text
data/Darknet.CSV
```

The scripts expect this relative path. Keep the filename consistent with the code.

## 7. Run the Dashboard

With the virtual environment active and the dataset in place, run:

```powershell
streamlit run app.py
```

Open the local URL printed in the terminal (typically `http://localhost:8501`) in your browser. In the dashboard, choose the number of autoencoder epochs and select **Run Experiment**. The dashboard's default setting used for the latest run was 10 epochs.

The dashboard indicates whether the run completed and displays the latest run's metrics separately from the original validated comparison. A success message and output-file existence are useful checks, but they do not independently prove that the predictions are correct. Use the validation scripts and inspect the generated outputs as well.

## 8. Validation and Supporting Scripts

The project includes scripts for training, temporal-shift evaluation, class-wise analysis, comparison and output checks. Important scripts include:

- `train_temporal_baseline_fair.py` — fair temporal baseline
- `train_self_supervised.py` — autoencoder representation learning and downstream classifier
- `train_self_supervised_repeat.py` — repeated run with a different seed
- `validate_model_comparison.py` — checks the comparison and prediction alignment
- `final_results_summary.py` — summarizes final results
- `compare_models.py` and `compare_class_f1.py` — generate comparison visualizations
- `verify_day1_outputs.py` — checks expected output files

Run a script from the project directory, for example:

```powershell
python validate_model_comparison.py
python final_results_summary.py
```

Script outputs depend on the local dataset, installed packages and files already present in `results/`. Check each script's configuration and output filenames before rerunning it. Some exploratory scripts use different approaches or settings and should not be treated as the reported final experiment unless their results have been independently validated.

## 9. Limitations

- The temporal test distribution is highly imbalanced.
- Minority classes, especially `file-transfer`, have low F1 scores.
- The experiment uses a single dataset and a specific chronological split.
- Results do not demonstrate generalization to every network environment or live traffic.
- The self-supervised stage does not use labels for autoencoder training, but the downstream classifier is supervised.
- The dashboard's latest run and the original validated comparison are distinct result sets.

## 10. Team

**F2 — Team 28**

- P Sasikiran — 24BCE0349
- G Monish — 24BCE2050
- A Harsha — 24BCE2292

## 11. References

1. Velan et al. (2015). Survey of encrypted traffic classification methods. *International Journal of Network Management*. DOI: https://doi.org/10.1002/nem.1901
2. Cui et al. (2023). Semi-2DCAE: semi-supervised encrypted traffic classification research. *PeerJ Computer Science*. DOI: https://doi.org/10.7717/peerj-cs.1635
3. Xu et al. (2025). Research on encrypted traffic classification using autoencoder and CNN methods. *PLOS ONE*. DOI: https://doi.org/10.1371/journal.pone.0333276
4. Rust-Nguyen et al. (2023). “Darknet traffic classification and adversarial attacks using machine learning.” *Computers & Security*, 127, 103098.
5. Canadian Institute for Cybersecurity. CIC-Darknet2020 dataset page: https://www.unb.ca/cic/datasets/darknet2020.html
6. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O’Reilly Media.

---

**Note:** This README describes the implementation and results documented for this project. Before submission, verify the reference metadata against the original publications and ensure the final report, code, screenshots and result files all refer to the intended experiment.
