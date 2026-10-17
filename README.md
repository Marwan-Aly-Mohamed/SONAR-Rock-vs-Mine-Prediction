# Sonar: Rock vs Mine Classification

A machine learning project that classifies a sonar return as a **rock (R)** or a **metal mine (M)** from 60 frequency-band energy readings. It covers the full workflow: data exploration, model comparison, final evaluation, a saved model, and a small web demo.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/sonar-rock-vs-mine/blob/main/notebooks/SONAR_Rock_vs_Mine_Prediction.ipynb)

<!-- Add a screenshot of the demo here: -->
<!-- ![Demo](images/demo_screenshot.png) -->

## Dataset

- **Source:** UCI Machine Learning Repository, *Connectionist Bench (Sonar, Mines vs. Rocks)*.
- **Size:** 208 samples, 60 numeric features, 1 label column (the file has no header row).
- **Features:** each number is the echo energy in one frequency band, with values between 0 and 1.
- **Classes:** 111 mines (53.4%) and 97 rocks (46.6%), so the classes are nearly balanced.
- **Quality:** no missing values and no duplicate rows.

## Approach

1. **Exploratory analysis:** class balance, mean spectrum per class, box plots, and a correlation heatmap. Mines show stronger average echoes in roughly bands 9 to 21, and rocks around bands 34 to 36, but individual samples vary a lot. Neighboring bands are strongly correlated.
2. **Split:** 75% train and 25% test (156 / 52 rows), stratified so both parts keep the same rock/mine ratio.
3. **Preprocessing:** `RobustScaler`, fitted on the training data only to avoid data leakage. Outliers were kept, because they look like genuine measurements and the dataset is small.
4. **Models compared:** Logistic Regression (`C=1`), SVM (RBF kernel, `C=1`), and Random Forest (default settings).
5. **Final model:** a scikit-learn `Pipeline` (scaler + Random Forest), saved with `joblib`.

## Results (test set, 52 samples)

| Model | Train accuracy | Test accuracy | Missed mines | False alarms |
|---|---|---|---|---|
| Logistic Regression | 0.923 | 0.808 | 5 | 5 |
| SVM | 0.968 | 0.808 | 4 | 6 |
| **Random Forest** | 1.000 | **0.846** | **1** | 7 |

**Why Random Forest?** The three models are close on accuracy, but a missed mine is the costly mistake, and Random Forest missed only 1 of 28 mines.

Final model, classification report:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Rock | 0.94 | 0.71 | 0.81 |
| Mine | 0.79 | 0.96 | 0.87 |

The trade-off is more false alarms: 7 of 24 rocks were flagged as mines.

## Limitations

- The dataset is small (208 rows), so the test score is noisy. One test sample is about 1.9 percentage points.
- The models were compared on the test set, so the reported score is slightly optimistic. Cross-validation would give a steadier estimate.
- All three models overfit somewhat (train accuracy is higher than test accuracy), and no hyperparameter tuning was done.
- There is no external validation on other seabeds or sonar settings, so the results may not carry over.

## Project structure

```
sonar-rock-vs-mine/
├── README.md
├── requirements.txt
├── notebooks/
│   └── SONAR_Rock_vs_Mine_Prediction.ipynb
├── data/
│   └── Dataset.csv
├── models/
│   └── sonar_model.joblib
├── app/
│   └── app.py
└── images/
```

## How to run

**In Google Colab:** click the badge above, upload `Dataset.csv` to `/content`, and run all cells.

**Locally:**

```bash
git clone https://github.com/YOUR_USERNAME/sonar-rock-vs-mine.git
cd sonar-rock-vs-mine
pip install -r requirements.txt
jupyter notebook notebooks/SONAR_Rock_vs_Mine_Prediction.ipynb
```

When running locally, change the data path in the notebook from `/content/Dataset.csv` to `../data/Dataset.csv`.

## Demo app

The notebook includes a small Gradio app: paste 60 comma-separated sonar values and it returns **Rock** or **Mine** with the probability of a mine.

```bash
python app/app.py
```

Note: the saved `.joblib` model should be loaded with the same scikit-learn version it was trained with (see `requirements.txt`).

## Tech stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Gradio.

## Acknowledgments

Dataset: Gorman, R. P., and Sejnowski, T. J. (1988), via the UCI Machine Learning Repository.
