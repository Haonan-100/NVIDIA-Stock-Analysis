
# NVIDIA Stock Forecasting and Analysis

This repository provides an end‑to‑end pipeline—data cleaning, feature engineering, classical econometrics, sentiment fusion, and deep learning—for **NVIDIA (NVDA)** price forecasting.

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Data](#data)  
3. [Environment and Setup](#environment-and-setup)
4. [Usage](#usage)  
5. [Methods](#methods)  
   - [Data Pre-processing & Feature Selection](#data-pre-processing-feature-selection)  
   - [Classical Time-Series Models](#classical-time-series-models)  
   - [Sentiment Integration](#sentiment-integration)  
   - [Hybrid RF → LSTM-Attention](#hybrid-rf-lstm-attention)  
6. [Results](#results)  
7. [Figures](#figures)  
8. [References](#references)  
9. [License](#license)


---

## Project Overview
1. **Forecast** NVDA prices with ARIMA, SARIMAX, GARCH, and a hybrid Random‑Forest ➔ Bidirectional LSTM + Attention network.  
2. **Explain** drivers such as S&P 500, Google, Microsoft, Intel, macro factors, and a news‑based `ImpactScore`.  
3. **Compare** each layer of model complexity against a simple T‑2 baseline (price from two days prior).

---

## Data
| File | Columns (sample) | Source |
|------|------------------|--------|
| `cleaned_data.csv` | `Date, NVDA_adj_close, SP500, Google_Adj_Close, …` | Yahoo Finance, World Bank, FRED |
| `merged_with_impact_score.csv` | above + `ImpactScore` | News sentiment scrape (VADER / FinBERT) |

---

## Environment and Setup
```bash
conda create -n nvda_env python=3.9
conda activate nvda_env
pip install -r requirements.txt
```

Key libraries – `pandas  •  numpy  •  scikit-learn  •  statsmodels  •  arch  •  tensorflow (keras)`

---

## Usage
### 1 . Clone
```bash
git clone https://github.com/YourUsername/NVIDIA-Forecasting.git
cd NVIDIA-Forecasting
```

### 2 . Drop data into `data/`
```bash
mkdir -p data
# place cleaned_data.csv and merged_with_impact_score.csv here
```

### 3 . Run full notebook‑style script **(quick start)**
```bash
python "Total Code.py"
```

### 4 . Modular workflow **(recommended)**
| Step | Script | Core Techniques |
|------|--------|-----------------|
| ① Pre‑processing | `src/data_prep.py` | drop NA/∞, `StandardScaler`, add 20‑/80‑day MA |
| ② Feature ranking | `src/feature_select.py` | Pearson corr, **PCA 95 %**, **LassoCV**, **Random Forest** |
| ③ ARIMA | `src/arima.py` | ADF test → ARIMA(4,1,0) rolling forecast |
| ④ SARIMAX & GARCH | `src/sarimax_garch.py` | exog = `[SP500_log, ImpactScore, INTC_ret]` |
| ⑤ Sentiment scrape | `src/news_sentiment.py` | VADER / FinBERT → daily `ImpactScore` |
| ⑥ Deep model | `src/lstm_attention.py` | RF meta‑feature ➔ 2×50 bi‑LSTM + Attention |
| ⑦ Plots | `src/visualization.py` | saves all figures to `docs/img/` |

Execute all:
```bash
for m in data_prep feature_select arima sarimax_garch lstm_attention visualization
do
  python -m src.$m
done
```

Artifacts (models, metrics JSON, PNGs) appear in `outputs/` and `docs/img/`.

---

## Methods

### Data Pre‑processing & Feature Selection
* Remove nulls/inf, scale features.  
* **PCA** keeps components explaining ≥ 95 % variance.  
* **LassoCV** (α cross‑validated) + **Random Forest** importance confirm top drivers: `SP500`, `MSFT_Adj_Close`, `ImpactScore`, 20‑/80‑day MA.

### Classical Time‑Series Models
* **ARIMA(4,1,0)** chosen via ACF/PACF and AIC grid search.  
* **SARIMAX(1,0,1)(0,1,1,12)** with exogenous factors.  
* **GARCH(1,1)** captures volatility clustering in log‑returns.

### Sentiment Integration
Daily news are scored; `ImpactScore` fed as an exogenous regressor and shown to correlate 0.43 with next‑day return.

### Hybrid RF → LSTMAttention
* Random Forest predictions appended as an extra feature.  
* Model: *Input → [bi‑LSTM × 2] → Attention → LSTM → Dense*  
* Regularised by Dropout 0.3 and L2 0.01, EarlyStopping patience 10.

---

## Results
| Model | Inputs | Best Test Metric | Baseline (T‑2) |
|-------|--------|------------------|----------------|
| ARIMA(4,1,0) | Close | MSE 34.2 | — |
| SARIMAX | Close + exog | MSE 22.5 | — |
| GARCH(1,1) | log‑σ² | LLH ↑ –2156 | — |
| **LSTM‑Attention** | 20 features, *time_steps = 2* | **MAE 2.42** | 3.44 |

* LSTM reduces MAE by **32 %** vs. baseline.  
* SARIMAX halves ARIMA error by adding macro + sentiment.  
* Residuals from SARIMAX nearly i.i.d.; tails addressed by GARCH.

---

## Figures
| Description | Image |
|-------------|-------|
| ACF (lag = 3) confirms short‑memory | ![acf lag3](docs/img/arima_acf_lag3.png) |
| Full ACF of Adjusted Close | ![acf full](docs/img/arima_acf_full.png) |
| SARIMAX residual diagnostics | ![sarimax](docs/img/sarimax_diagnostics.png) |
| Daily return vs. ImpactScore | ![impact](docs/img/sentiment_impact.png) |
| LSTM vs. Actual vs. Baseline | ![lstm](docs/img/lstm_pred_ts2.png) |

---

## References
* Xiao, Q., & Ihnaini, B. (2023). *Stock trend prediction using sentiment analysis*. PeerJ Computer Science.  
* [Yahoo Finance – NVDA](https://finance.yahoo.com/quote/NVDA/news/)  
* Mnih, V. et al. (2016). *Asynchronous Methods for Deep Reinforcement Learning*. arXiv:1607.01958.

---

## License
This repository is released for academic and educational use.  
Please verify licensing terms of individual data sources before commercial deployment.
```
