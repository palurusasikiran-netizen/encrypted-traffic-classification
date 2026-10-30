# Self-Supervised Encrypted Traffic Classification Under Network Distribution Shifts

**Course:** Computer Networks Lab – Innovative Evaluation Practices  
**Institution:** Vellore Institute of Technology (VIT), School of Computer Science and Engineering  
**Semester:** Fall Semester 2026–2027  
**Slot:** F2  
**Team:** Team 28

---

## 1. Project Overview

This project studies encrypted network traffic classification under changing network traffic distributions over time.

Traditional machine learning models may experience performance degradation when the distribution of network traffic changes. This project investigates whether self-supervised representation learning can improve classification performance under temporal distribution shifts.

Two approaches are implemented and compared:

1. **Temporal Supervised Baseline:** A HistGradientBoostingClassifier trained using preprocessed numerical network traffic features.
2. **Self-Supervised Pipeline:** An autoencoder learns compact representations of network traffic without using class labels during representation learning. A HistGradientBoostingClassifier subsequently classifies the learned representations using training labels.

The project uses the CIC-Darknet2020 dataset and a chronological training/testing split.

A Streamlit dashboard is also implemented to display experimental results, model comparisons and performance metrics.

**Project Type:** Offline machine-learning research prototype using an existing encrypted network traffic dataset. It is not a live packet-capture or real-time network monitoring system.

---

## 2. Project Objectives

The main objectives are:

- Study encrypted network traffic classification under temporal distribution shifts.
- Implement a supervised machine-learning baseline.
- Develop a self-supervised autoencoder for traffic feature representation learning.
- Train a downstream classifier using learned representations.
- Compare the performance of supervised and self-supervised approaches.
- Evaluate models using Accuracy, Macro-F1 and class-wise F1 scores.
- Develop a Streamlit dashboard to visualize experimental results.
- Analyze the limitations of encrypted traffic classification under changing traffic distributions.

---

## 3. Technologies and Tools Used

| Technology / Tool | Purpose |
|---|---|
| Python 3.12 | Main programming language |
| NumPy | Numerical computations |
| Pandas | Dataset processing and analysis |
| Scikit-learn | Machine-learning algorithms and evaluation |
| PyTorch | Autoencoder implementation and training |
| Streamlit | Interactive project dashboard |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Visual Studio Code | Development environment |
| Git and GitHub | Source-code management and project submission |

---

## 4. Dataset Description

**Dataset:** CIC-Darknet2020  
**Dataset File:** `Darknet.CSV`  
**Expected Location:** `data/Darknet.CSV`

Dataset details:

- Original dataset shape: 158,616 rows × 85 columns.
- Numerical traffic features used: 64.
- Time range observed: 30 March 2015 to 25 February 2016.
- Four traffic classes used in the main temporal comparison:
  - audio-streaming
  - browsing
  - file-transfer
  - p2p

Identifiers, timestamps and label columns are excluded from the numerical model input features.

### Dataset Access

The dataset is not included in this GitHub repository by default because of its size.

Obtain the CIC-Darknet2020 dataset from the Canadian Institute for Cybersecurity:

https://www.unb.ca/cic/datasets/darknet2020.html

After obtaining the dataset, place `Darknet.CSV` inside the `data/` directory.

Do not redistribute the dataset unless its applicable usage and redistribution terms permit it.

---

## 5. Experimental Design

### 5.1 Chronological Train-Test Split

The project uses a temporal cutoff of **24 February 2016**.

| Description | Details |
|---|---|
| Training period | Before 24 February 2016 |
| Testing period | 24 February 2016 onward |
| Training rows | 60,778 |
| Testing rows | 66,731 |
| Total rows in temporal comparison | 127,509 |

The main temporal comparison uses four traffic classes represented in both temporal periods.

The original dataset contains 158,616 rows, whereas the reported temporal comparison contains 127,509 rows. The remaining 31,107 rows are outside the reported comparison. Their exact filtering and exclusion accounting should be interpreted using the preprocessing scripts.

### 5.2 Data Preprocessing

The main temporal pipeline performs the following operations:

1. Normalize traffic class labels.
2. Parse timestamps.
3. Exclude identifiers, IP addresses, timestamps and label columns from the model input.
4. Replace infinite values.
5. Impute missing values using statistics fitted on the training data.
6. Remove features that are constant in the training set.
7. Standardize features using training-set statistics.
8. Clip standardized values to the range [-10, 10].
9. Apply the fitted preprocessing steps to the test dataset.

Preprocessing statistics are fitted using the training period to avoid using test-set statistics during training.

### 5.3 Supervised Baseline

The supervised baseline uses:

- Algorithm: HistGradientBoostingClassifier
- Input: Preprocessed numerical traffic features
- Maximum iterations: 100
- Learning rate: 0.1
- Maximum leaf nodes: 31
- Random state: 42

### 5.4 Self-Supervised Pipeline

The self-supervised approach consists of two main stages.

**Stage 1: Autoencoder Representation Learning**

The autoencoder learns compact representations from the training-period traffic features without using class labels during this representation-learning stage.

Architecture:

Input → 32 → 16 latent dimensions → 32 → Output

Configuration:

- Activation: ReLU
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 512
- Reconstruction objective: Mean Squared Error (MSE)
- Dashboard default training duration: 10 epochs

**Stage 2: Downstream Classification**

The learned representations are used as input features for a HistGradientBoostingClassifier.

The downstream classifier uses training labels for supervised classification.

---

## 6. Model Performance and Experimental Results

### 6.1 Original Validated Model Comparison

The following values represent the original validated temporal comparison.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Fair Temporal Baseline | 77.81% | 38.57% |
| Self-Supervised Pipeline (Seed 42) | 86.47% | 48.81% |
| Self-Supervised Pipeline (Seed 123) | 86.32% | 48.43% |

For seed 42, the self-supervised pipeline achieved:

- Accuracy improvement: approximately 8.66 percentage points.
- Macro-F1 improvement: approximately 10.24 percentage points.

### 6.2 Separate Latest Dashboard Run

The dashboard also contains a separate later experimental run.

| Metric | Value |
|---|---:|
| Accuracy | 86.66% |
| Macro-F1 | 48.92% |
| Final Autoencoder Reconstruction Loss | 0.060400 |

These values belong to a separate run and should not be mixed with the original validated model comparison.

### 6.3 Class-Wise Performance

Original seed-42 comparison:

| Traffic Class | Baseline F1 | Self-Supervised F1 |
|---|---:|---:|
| audio-streaming | 0.0132 | 0.0935 |
| browsing | 0.6541 | 0.9268 |
| file-transfer | 0.0155 | 0.0155 |
| p2p | 0.8598 | 0.9163 |

The results show that the file-transfer class remains difficult to classify. Audio-streaming performance is also weak.

Therefore, accuracy alone is insufficient to evaluate the model. Macro-F1 and class-wise F1 scores are also important.

---

## 7. Repository Structure

The GitHub repository contains the Python implementation, analysis scripts, experiment results, screenshots and supporting resources.

The high-level structure is:

```text
encrypted-traffic-classification/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── app.py
│
├── *.py
│   └── Training, evaluation, validation and analysis scripts
│
├── results/
│   └── Experimental outputs, reports and comparison results
│
├── screenshots/
│   └── Dashboard screenshots
│
├── F2_Team28_SourceCode.zip
├── F2_Team28_Screenshots.zip
│
└── submission/
    ├── F2_Team28_Report.pdf
    └── F2_Team28_PPT.pptx
```

The `submission/` folder will contain the final report and presentation after they are ready.

The original Python scripts and screenshot folder are retained separately from their ZIP archives for convenient inspection.

The dataset is stored locally and is not committed to the repository by default.

---

## 8. Software Requirements

The following software and libraries are required:

- Python 3.12
- pip
- Visual Studio Code or another Python-compatible development environment
- NumPy
- Pandas
- Scikit-learn
- PyTorch
- Matplotlib
- Seaborn
- Streamlit

The exact package dependencies are listed in `requirements.txt`.

**Recommended environment:** Python 3.12. The project was developed and tested using a CPU-based PyTorch environment.

---

## 9. Installation Procedure

### Step 1: Clone the GitHub Repository

```bash
git clone https://github.com/palurusasikiran-netizen/encrypted-traffic-classification.git
```

### Step 2: Open the Project Directory

```bash
cd encrypted-traffic-classification
```

### Step 3: Create a Virtual Environment

For Windows:

```bash
py -3.12 -m venv .venv
```

### Step 4: Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the virtual environment's Python executable directly or follow your institution's approved PowerShell policy.

### Step 5: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 6: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 10. Dataset Placement

Create a directory named `data` inside the project directory if it does not already exist.

Place the downloaded dataset at:

```text
data/Darknet.CSV
```

Ensure the filename and directory structure match the paths expected by the Python scripts.

---

## 11. Project Execution

### 11.1 Run the Streamlit Dashboard

After installing dependencies and placing the dataset in the correct location, execute:

```bash
streamlit run app.py
```

The terminal displays a local URL, typically:

```text
http://localhost:8501
```

Open this address in your web browser.

### 11.2 Run the Experimental Scripts

The repository contains Python scripts for:

- Dataset inspection and preprocessing.
- Supervised baseline training.
- Self-supervised autoencoder training.
- Temporal distribution shift evaluation.
- Class-wise performance analysis.
- Model comparison.
- Experimental result validation.

Important scripts include:

| Script | Purpose |
|---|---|
| `train_temporal_baseline_fair.py` | Trains the fair temporal baseline |
| `train_self_supervised.py` | Trains the self-supervised pipeline |
| `train_self_supervised_repeat.py` | Performs a repeated self-supervised run |
| `evaluate_controlled_shift.py` | Evaluates the controlled temporal split |
| `evaluate_temporal_shift.py` | Evaluates temporal distribution shifts |
| `validate_model_comparison.py` | Validates model comparison outputs |
| `final_results_summary.py` | Summarizes experimental results |
| `compare_models.py` | Generates model comparison visualizations |
| `verify_day1_outputs.py` | Checks expected experimental output files |

Example commands:

```bash
python validate_model_comparison.py
```

```bash
python final_results_summary.py
```

Run the required scripts from the project directory.

**Note:** Individual scripts may depend on the dataset, installed libraries and previously generated result files. Check each script's configuration and expected inputs before execution. Exploratory scripts may use different settings and should not automatically be treated as part of the final validated experiment.

---

## 12. Project Execution Workflow

The overall project workflow is:

1. Obtain the CIC-Darknet2020 dataset.
2. Place the dataset inside the `data/` directory.
3. Inspect the dataset and traffic labels.
4. Preprocess the numerical traffic features.
5. Perform the chronological train-test split.
6. Train the fair supervised temporal baseline.
7. Train the autoencoder using training-period features.
8. Generate learned feature representations.
9. Train the downstream classifier using training labels.
10. Evaluate both approaches on the chronological test set.
11. Compare Accuracy, Macro-F1 and class-wise F1 scores.
12. Validate and save the experimental outputs.
13. Launch the Streamlit dashboard to visualize the results.

---

## 13. Testing and Verification

The project includes scripts for checking experimental outputs, comparing models and inspecting classification performance.

The following aspects are evaluated:

- Dataset preprocessing and label handling.
- Chronological training and testing separation.
- Supervised baseline training.
- Autoencoder training.
- Learned representation generation.
- Downstream classifier training.
- Accuracy and Macro-F1 comparison.
- Class-wise classification performance.
- Model comparison and prediction alignment.
- Dashboard execution and result visualization.

The validation scripts and generated files are available in the repository.

Important supporting scripts include:

```text
validate_model_comparison.py
final_results_summary.py
compare_models.py
compare_class_f1.py
verify_day1_outputs.py
```

The reported experimental results are documented in the Results section.

Successful script execution and the presence of output files are useful verification checks, but they do not independently establish prediction correctness. The validation outputs should also be inspected.

---

## 14. Dashboard Screenshots

The `screenshots/` directory contains visual evidence of the implemented Streamlit dashboard.

It is intended to demonstrate:

- Dashboard interface.
- Experiment configuration.
- Model execution.
- Experimental result visualization.
- Performance comparison.
- Classification metrics.

The complete screenshot collection is also provided separately as:

```text
F2_Team28_Screenshots.zip
```

---

## 15. Project Submission Files

The following files are included or will be included in the GitHub repository according to the faculty's submission requirements.

| Filename | Description |
|---|---|
| `F2_Team28_SourceCode.zip` | Complete Python source code |
| `F2_Team28_Screenshots.zip` | Dashboard screenshots |
| `F2_Team28_Report.pdf` | Complete project report |
| `F2_Team28_PPT.pptx` | Project presentation |

The original Python source files, `screenshots/` directory, `results/` directory, README and dependency files are also retained in the repository.

A Cisco Packet Tracer `.pkt` file is not applicable because this project is an offline machine-learning and dashboard-based implementation rather than a Packet Tracer simulation.

---

## 16. Limitations

The project has the following limitations:

- The evaluation uses a single dataset and a specific chronological split.
- The test distribution is highly imbalanced.
- Minority classes, particularly file-transfer, have low F1 scores.
- The results do not establish generalization to every network environment.
- The implementation does not perform live packet capture or real-time network traffic monitoring.
- The autoencoder's representation-learning stage does not use class labels, but the downstream classifier is supervised.
- Only two self-supervised runs were included in the original comparison; their variation should be interpreted descriptively.
- The separate dashboard run should not be confused with the original validated comparison.

---

## 17. Future Scope

Possible future improvements include:

- Evaluating the approach on additional encrypted traffic datasets.
- Improving classification performance for minority traffic classes.
- Investigating additional self-supervised learning methods.
- Evaluating robustness under different temporal and network distribution shifts.
- Increasing the number of repeated experimental runs.
- Exploring real-time encrypted traffic classification in a suitable deployment environment.

---

## 18. Team Details

**Slot:** F2  
**Team:** 28

| Student Name | Register Number |
|---|---|
| P. Sasikiran | 24BCE0349 |
| G. Monish | 24BCE2050 |
| A. Harsha | 24BCE2292 |

---

## 19. References

1. Velan et al. (2015). Survey of encrypted traffic classification methods. *International Journal of Network Management*. https://doi.org/10.1002/nem.1901

2. Cui et al. (2023). Semi-2DCAE: Semi-supervised encrypted traffic classification research. *PeerJ Computer Science*. https://doi.org/10.7717/peerj-cs.1635

3. Xu et al. (2025). Research on encrypted traffic classification using autoencoder and CNN methods. *PLOS ONE*. https://doi.org/10.1371/journal.pone.0333276

4. Rust-Nguyen et al. (2023). Darknet traffic classification and adversarial attacks using machine learning. *Computers & Security*, 127, 103098.

5. Canadian Institute for Cybersecurity. CIC-Darknet2020 Dataset. https://www.unb.ca/cic/datasets/darknet2020.html

6. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.

---

**Project:** Self-Supervised Encrypted Traffic Classification Under Network Distribution Shifts

**Course:** Computer Networks Lab – BCSE308P

**Institution:** Vellore Institute of Technology (VIT)

**Slot and Team:** F2 – Team 28
