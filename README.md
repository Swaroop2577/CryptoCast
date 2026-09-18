# BitFusion: Fusing Market and Alternative Data for Bitcoin

**BitFusion** is a machine learning project for predicting the **next-day direction of Bitcoin prices** by combining traditional market data with alternative signals derived from **Wikipedia activity and sentiment**.

The project explores whether publicly available behavioral signals, when combined with historical Bitcoin market data and engineered temporal features, can provide useful information for short-term price direction prediction.

## 📌 Project Overview

Instead of predicting the exact future price of Bitcoin, the problem is formulated as a **binary classification task**:

* `1` → Bitcoin's closing price is higher the next day
* `0` → Bitcoin's closing price is not higher the next day

The model uses historical Bitcoin market data along with activity and sentiment information from the Bitcoin Wikipedia page.

## 🔍 Approach

### Market Data

Historical Bitcoin data is collected using **Yahoo Finance (`yfinance`)**, including:

* Open
* High
* Low
* Close
* Trading Volume

### Wikipedia Signals

Wikipedia revision history for the Bitcoin page is used as an alternative source of information.

The project extracts:

* Daily Wikipedia edit count
* Sentiment from revision comments
* Negative sentiment proportion

These signals are aggregated on a daily basis and merged with the Bitcoin market data.

### Feature Engineering

To capture market behavior across different time scales, the project introduces **multi-horizon rolling features** using windows of:

`2, 7, 60, and 365 days`

These include:

* Price-to-rolling-average ratios
* Rolling Wikipedia edit activity
* Historical directional trends

This allows the model to capture both short-term and long-term patterns in the data.

## 🤖 Machine Learning

The project experiments with **Random Forest** and **XGBoost** classifiers.

XGBoost is used as the primary model for the final evaluation.

```python
XGBClassifier(
    learning_rate=0.1,
    n_estimators=200,
    random_state=1
)
```

## 🔄 Walk-Forward Backtesting

Since Bitcoin data is time-dependent, the project uses an **expanding-window walk-forward backtesting approach** rather than randomly shuffling the dataset.

At each step, the model is trained on historical observations and evaluated on a subsequent unseen period.

This provides a more realistic evaluation of how the model would perform when predicting future observations.

## 📊 Results

| Model   | Features                          |  Precision |
| ------- | --------------------------------- | ---------: |
| XGBoost | Initial features                  | **49.90%** |
| XGBoost | Engineered multi-horizon features | **52.83%** |

Feature engineering improved the walk-forward precision from **49.90% to 52.83%**, an improvement of approximately **2.93 percentage points**.

> **Note:** Precision is reported here rather than accuracy. The initial Random Forest experiment used a different holdout evaluation setup, so its 78.57% precision is not directly comparable to the walk-forward XGBoost results.

## 🛠️ Tech Stack

* **Python**
* **Pandas** – Data processing and feature engineering
* **Scikit-learn** – Machine learning utilities and Random Forest
* **XGBoost** – Gradient boosting classifier
* **Hugging Face Transformers** – Sentiment analysis
* **yFinance** – Bitcoin market data
* **mwclient** – Wikipedia revision data
* **Matplotlib** – Visualization
* **Jupyter Notebook** – Development and experimentation

## 📂 Project Structure

```text
BitcoinPricePrediction/
│
├── prediction.ipynb
│   └── Data processing, feature engineering,
│       model training and backtesting
│
├── sentimental.ipynb
│   └── Wikipedia data collection and
│       sentiment analysis
│
├── wikipedia_edits.csv
│   └── Processed Wikipedia activity data
│
└── README.md
```

## 🚀 Getting Started

Clone the repository:

```bash
git clone https://github.com/balu1305/BitcoinPricePrediction.git
cd BitcoinPricePrediction
```

Install the required packages:

```bash
pip install pandas scikit-learn xgboost yfinance matplotlib mwclient transformers torch
```

Run Jupyter Notebook:

```bash
jupyter notebook
```

Run `sentimental.ipynb` first to generate the Wikipedia-based features, followed by `prediction.ipynb` for model training and evaluation.

