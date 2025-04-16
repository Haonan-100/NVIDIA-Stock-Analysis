
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
Below is a list of each data file contained in the `Data` directory:

- **Backup.csv**
- **Database.csv.csv**
- **cleaned_data.csv**
- **merged_data.csv**
- **merged_data2.csv**
- **merged_with_impact.csv**
- **merged_with_impact_score.csv**
- **nvidia_events_filtered.csv**

Each file is used at different stages of the data processing and analysis pipeline. Please refer to the specific sections of the project documentation for details on how each file is utilized


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

### Data Pre‑processing & Feature Selection
We start by removing `NaN`/`Inf` rows and standardising all numerical columns.  
Two complementary feature‑ranking tracks are applied:

| Technique | Purpose | Outcome |
|-----------|---------|---------|
| **PCA (95 % var)** | Orthogonalise & compress | 9 principal components retained |
| **LassoCV + Random Forest** | Sparse, non‑parametric importance | Top drivers: `SP500`, `MSFT_Adj_Close`, `ImpactScore`, 20‑/80‑day MA |

> These features feed every downstream model to ensure consistency and avoid look‑ahead bias.

---

### Classical Time‑Series Models

#### ARIMA
The **ADF test** rejects the unit‑root hypothesis after one differencing, leading to an `(4,1,0)` specification chosen via AIC grid search.  
The autocorrelation structure is visualised below:

![ACF lag = 3](https://github.com/Haonan-100/NVIDIA-Stock-Analysis/blob/main/Photo/ARIMA%20Model%20-%20Nvidia%20Stock%20-%20Autocorrelation%20plot%20with%20lag%20%3D%203%20.png)

![Full ACF](https://github.com/Haonan-100/NVIDIA-Stock-Analysis/blob/main/Photo/ARIMA%20Model-%20NVIDIA%20Adjusted%20Close%20-%20Autocorrelation%20plot.png)

Both plots confirm strong short‑memory up to three lags, justifying the AR term.

#### SARIMAX
Seasonality (12‑month) and exogenous regressors—`SP500_log`, `ImpactScore`, `INTC_ret`—are introduced in a **SARIMAX(1,0,1)(0,1,1,12)** framework.

<div align="center">

<img src="https://github.com/Haonan-100/NVIDIA-Stock-Analysis/blob/main/Photo/SARIMAX%20Model.png" width="48%">
<img src="https://github.com/Haonan-100/NVIDIA-Stock-Analysis/blob/main/Photo/SARIMAX%20Model2.png" width="48%">

</div>

*Left panel*: fitted vs. observed shows tight tracking.  
*Right panel*: standardised residuals & Q‑Q plot indicate near‑normality with mild tail risk—later captured by GARCH.

#### GARCH
A **GARCH(1,1)** layer is fitted to log‑return residuals, reducing volatility clustering and yielding a log‑likelihood of −2156 (↑ vs. ARIMA).

---

### Sentiment Integration
Daily news headlines are scored by **VADER** and **FinBERT**; scores are averaged into an **`ImpactScore`** that enters SARIMAX and LSTM as a leading indicator.

<div align="center">

<img src="https://github.com/Haonan-100/NVIDIA-Stock-Analysis/blob/main/Photo/Sentiment%20Analysis%20-%20impact%20of%20events%20on%20NVDA%20Stock%20Price.png" width="48%">
<img src="https://github.com/Haonan-100/NVIDIA-Stock-Analysis/blob/main/Photo/Sentiment%20Analysis%20-%20NVDA%20Stock%20Price%20Around%20Events.png" width="48%">

</div>

*Scatter* illustrates a Pearson‑r = 0.43 between `ImpactScore` and next‑day return.  
*Event overlay* shows price jumps aligning with major positive (green) and negative (red) news.

---

### Hybrid RF → LSTM‑Attention
1. **Random Forest** predicts next‑day price to create a meta‑feature.  
2. **Neural net architecture**:  

```
Input (20 features, T‑2 window)
     └─ bi‑LSTM (50) ─┐
     └─ bi‑LSTM (50) ─┘ → Attention → LSTM (50) → Dense(1)
```

3. **Regularisation**: Dropout 0.3, L2 0.01, EarlyStopping (patience 10).

![LSTM Performance](https://github.com/Haonan-100/NVIDIA-Stock-Analysis/blob/main/Photo/LSTM.png)

The plot compares Actual (blue), LSTM (red), and the naive T‑2 baseline (green).

---

## Results

| Model | Inputs | Best Test Metric | T‑2 Baseline |
|-------|--------|------------------|--------------|
| **ARIMA(4,1,0)** | Close | MSE 34.2 | — |
| **SARIMAX** | Close + exog | MSE 22.5 | — |
| **GARCH(1,1)** | log‑σ² | LLH ↑ −2156 | — |
| **LSTM‑Attention** | 20 features, *time_steps = 2* | **MAE 2.42** | 3.44 |

* **32 % MAE reduction** from baseline to LSTM.  
* SARIMAX halves ARIMA’s error by injecting macro + sentiment.  
* Residual heavy tails in SARIMAX are mostly neutralised after the GARCH layer.

```

## References
* Xiao, Q., & Ihnaini, B. (2023). *Stock trend prediction using sentiment analysis*. PeerJ Computer Science.  
* [Yahoo Finance – NVDA](https://finance.yahoo.com/quote/NVDA/news/)  
* Mnih, V. et al. (2016). *Asynchronous Methods for Deep Reinforcement Learning*. arXiv:1607.01958.

---

Thank you for visiting this project. If you have any questions or suggestions, feel free to open an issue or submit a pull request.
