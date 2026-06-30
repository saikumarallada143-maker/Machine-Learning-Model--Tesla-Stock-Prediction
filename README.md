# Machine-Learning-Model--Tesla-Stock-Prediction
ML pipeline for forecasting Tesla (TSLA) stock prices using historical price/volume data and regression modeling
# Tesla Stock Price Prediction

A machine learning project that predicts Tesla's (TSLA) next-day closing stock price using historical OHLCV (Open, High, Low, Close, Volume) data and regression models.

## Overview

This project performs exploratory data analysis on Tesla's historical stock data, engineers a next-day closing price target, and trains/compares three regression models to find the best predictor.

## Dataset

- **File:** `Tesla.csv`
- **Columns:** `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`
- Place the dataset in the project root (or update the file path in the code) before running.

## Exploratory Data Analysis

- Closing price trend over time
- Trading volume trend over time
- Correlation heatmap across `Open`, `High`, `Low`, `Close`, `Volume`, `Adj Close`
- Close price vs. Volume scatter plot

## Feature Engineering

The target variable is next day's closing price, created by shifting `Close` back by one row:

```python
df['Target'] = df['Close'].shift(-1)
```

**Features used:** `Open`, `High`, `Low`, `Close`, `Volume`

Data is split chronologically (80% train / 20% test) rather than randomly, to respect time-series order.

## Models Compared

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Linear Regression** | **3.66** | **5.01** | **0.958** |
| Random Forest (tuned via GridSearchCV) | 4.07 | 5.50 | 0.950 |
| Gradient Boosting | 4.13 | 5.65 | 0.947 |

Linear Regression performed best on this dataset and was selected as the final model.

Random Forest hyperparameters were tuned with `GridSearchCV` over:
```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10]
}
```

## Project Structure

```
tesla-stock-prediction/
├── Tesla.csv                       # Historical stock data (not included, add your own)
├── tesla_stock_prediction.py       # Main script / notebook source
├── tesla_stock_model.pkl           # Saved trained model (Linear Regression)
└── README.md
```

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## Usage

1. Place `Tesla.csv` in the project directory.
2. Run the script (or notebook cells) in order:
   - Load and inspect data
   - Generate EDA charts
   - Engineer features and split data
   - Train and evaluate Linear Regression, Random Forest, and Gradient Boosting models
   - Save the final model

```bash
python tesla_stock_prediction.py
```

3. Load the saved model for inference:
```python
import joblib
model = joblib.load('tesla_stock_model.pkl')
prediction = model.predict(sample_features)
```

## Results Summary

Linear Regression achieved the strongest performance (R² of 0.958), suggesting that next-day Tesla closing price is largely well-approximated by a linear combination of the same day's Open, High, Low, Close, and Volume — likely because these features are highly autocorrelated with the target by construction. Tree-based ensemble models did not outperform the simpler linear approach on this dataset.

## Disclaimer

This project is for educational and research purposes only. The model is trained on historical price data and predictions should **not** be used as financial advice. Stock markets are influenced by many factors not captured in this dataset, and past performance does not guarantee future results.

## License

MIT License (or specify your preferred license)
