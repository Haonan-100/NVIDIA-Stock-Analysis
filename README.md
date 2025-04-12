# NVIDIA Stock Forecasting and Analysis

This repository provides a comprehensive analysis and forecasting project focused on **NVIDIA (NVDA) stock**. It combines data cleaning, feature engineering, correlation analysis, and multiple predictive models (including ARIMA, SARIMAX, GARCH, and LSTM with Attention) to understand NVIDIA’s historical performance and forecast its future price movements.

## Table of Contents
- [Project Overview](#project-overview)
- [Data](#data)
- [Key Features and Models](#key-features-and-models)
- [Environment and Dependencies](#environment-and-dependencies)
- [Usage](#usage)
- [Results and Visualization](#results-and-visualization)
- [References](#references)

---

## Project Overview

NVIDIA is a leading technology company whose stock shows notable performance and volatility. This project aims to:
1. Predict NVDA stock prices using a variety of models ranging from time series methods (ARIMA/SARIMAX/GARCH) to deep learning architectures (LSTM, Bidirectional LSTM, and Attention).
2. Identify important market factors (e.g., S&P 500, Google, Microsoft, Intel, inflation rates) and their correlation with NVIDIA’s movements.
3. Incorporate sentiment or event impact scores (`ImpactScore`) to capture the effect of market news and critical events on price dynamics.

## Data

- **Primary CSV Files**: 
  - `cleaned_data.csv` and `merged_with_impact_score.csv` contain the cleaned and merged historical data.
  - Each row typically includes information such as `Date`, `NVDA_adj_close`, `SP500`, `Google_Adj_Close`, `MSFT_Adj_Close`, `INTC_Adj_Close`, and additional fields like `ImpactScore`.
- **Source**: 
  - Yahoo Finance (historical stock data for NVDA, S&P 500, etc.)
  - World Bank/FRED (macro indicators)
  - Custom scripts or third-party data for event-driven sentiment/impact score.

## Key Features and Models

1. **Feature Engineering**  
   - Daily returns, log returns, short-term (20-day) and long-term (80-day) moving averages, etc.
   - Principal Component Analysis (PCA) to reduce dimensionality.
   - Lasso and Random Forest for feature importance scoring.

2. **Time Series Models**  
   - **ARIMA/SARIMAX** for autocorrelation-based forecasting.
   - **GARCH** (via `arch` library) for volatility modeling and capturing financial time series volatility clustering.

3. **Deep Learning Models**  
   - **LSTM / Bidirectional LSTM** to handle long-term dependencies in time-series.
   - **Attention mechanism** to focus on the most relevant time steps and improve predictive accuracy.
   - Comparison with a baseline T-2 model (using the price from two days ago as a naive forecast).

## Environment and Dependencies

- **Python 3.x**  
- Key packages:  
  - `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`  
  - `statsmodels`, `arch`  
  - `tensorflow` (Keras)  
  - `PyPDF2` (optional, if you need to parse PDF files)  
- It is recommended to use a virtual environment or `conda` to manage dependencies:
  ```bash
  conda create -n nvda_env python=3.9
  conda activate nvda_env
  pip install -r requirements.txt
  ```

## Usage

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/YourUsername/NVIDIA-Forecasting.git
   cd NVIDIA-Forecasting
   ```
2. **Place the Data**  
   - Ensure that `cleaned_data.csv`, `merged_with_impact_score.csv`, and other relevant CSV files are located at the correct file paths specified in the code.
3. **Run the Code**  
   - The main script is `Total Code.py`. You can execute it directly:
     ```bash
     python "Total Code.py"
     ```
   - Various plots, metrics, and model summaries will be printed or displayed in the console/plots.
4. **Inspect the Results**  
   - Model predictions, MSE, RMSE, MAE, R², etc. are shown in the console output.
   - Visualizations (e.g., correlation heatmaps, forecast vs. actual plots) are generated to illustrate how well each model captures NVDA price movements.

## Results and Visualization

- **ARIMA / SARIMAX**: Demonstrates baseline time-series performance; includes residual diagnostics and confidence intervals.
- **GARCH**: Helpful for modeling volatility and capturing periods of high price fluctuations.
- **LSTM + Attention**: Showcases the potential for improved accuracy by leveraging deep learning architectures, capturing nonlinear dynamics and long-range dependencies.

Example forecasting visualization:
```
Observed (Blue) vs. Forecast (Red) lines
Confidence intervals (Pink area)
```
A comparison with a naive T-2 baseline is also provided to highlight the added value of advanced models.

## References

- Xiao, Q., & Ihnaini, B. (2023). *Stock trend prediction using sentiment analysis.* PeerJ Computer Science.  
- [Yahoo Finance](https://finance.yahoo.com/quote/NVDA/news/)  
- Mnih, V., et al. (2016). *Asynchronous Methods for Deep Reinforcement Learning*, arXiv:1607.01958.  

---

### License

This project is for educational and research purposes. Refer to individual data sources for license and usage details.

```
