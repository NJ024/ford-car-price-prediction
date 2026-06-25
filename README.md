# 🚗 Ford Used Car Price Prediction

Predicting used Ford car listing prices from vehicle attributes, with a direct comparison of two categorical encoding strategies — one-hot encoding vs. label encoding — to show their real impact on model performance.

**Best model: R² = 0.846 | Adjusted R² = 0.844 (Linear Regression, one-hot encoded features)**

---

## 📌 Project Overview

Used car pricing depends on a mix of mechanical specs (engine size, mileage, fuel type) and market factors (model, year, transmission). This project builds a Linear Regression model to predict listing price for used Ford vehicles, and — rather than just picking one encoding method — deliberately trains the same model on two different categorical encodings to measure which one actually performs better and why.

## 📊 Dataset

- **Source:** [100,000 UK Used Car Data Set (Kaggle)](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes) — `ford.csv`
- **Size:** 17,966 records, 9 columns
- **Features:** `model`, `year`, `transmission`, `mileage`, `fuelType`, `tax`, `mpg`, `engineSize`
- **Target:** `price` (GBP)

## 🔍 Workflow

1. **Exploratory Data Analysis** — price distribution, correlation heatmap, boxplots of price across year/model/transmission/fuel type/engine size/tax band, and a mileage-vs-price scatterplot
2. **Data Cleaning** — verified no missing values across all 9 columns
3. **Categorical Encoding (two approaches, compared head-to-head):**
   - **One-hot encoding** (`pd.get_dummies`) on `model`, `transmission`, `fuelType`
   - **Label encoding** (`sklearn.LabelEncoder`) on the same three columns
4. **Feature Scaling** — `StandardScaler` applied to numeric features (`year`, `mileage`, `tax`, `mpg`, `engineSize`)
5. **Modeling** — Linear Regression trained separately on each encoded dataset, 80/20 train-test split
6. **Evaluation** — R² and Adjusted R² compared across both encoding strategies

## 💡 Key Findings

| Encoding Strategy | R² (test set) | Adjusted R² |
|---|---|---|
| **One-hot encoding** | **0.846** | **0.844** |
| Label encoding | 0.737 | — |

- **One-hot encoding outperformed label encoding by ~11 percentage points of R².** This is expected for Linear Regression: label encoding assigns an arbitrary numeric order to categories like `model` (e.g. Fiesta=3, Focus=7), which falsely implies a magnitude relationship the linear model treats as real — one-hot encoding avoids that distortion.
- Strongest visual price drivers from EDA: vehicle `year` (newer cars command higher prices) and `engineSize`, while `mileage` shows a clear negative relationship with price.

## 📷 Exploratory Visuals

| Price Distribution | Correlation Heatmap |
|---|---|
| ![Price Distribution](images/price_distribution.png) | ![Correlation Heatmap](images/correlation_heatmap.png) |

| Price by Year | Mileage vs. Price |
|---|---|
| ![Price by Year](images/price_by_year_boxplot.png) | ![Mileage vs Price](images/mileage_vs_price_scatter.png) |

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Scikit-learn`

## 📁 Repository Structure

```
ford-car-price-prediction/
├── data/
│   └── ford.csv
├── notebooks/
│   └── ford_price_eda_and_modeling.ipynb
├── images/
│   ├── price_distribution.png
│   ├── correlation_heatmap.png
│   ├── price_by_year_boxplot.png
│   └── mileage_vs_price_scatter.png
├── README.md
├── requirements.txt
└── LICENSE
```

## ▶️ How to Run

```bash
git clone https://github.com/<your-username>/ford-car-price-prediction.git
cd ford-car-price-prediction
pip install -r requirements.txt
jupyter notebook notebooks/ford_price_eda_and_modeling.ipynb
```

## 🚀 Future Improvements

- Add Random Forest / Gradient Boosting / XGBoost and compare against the Linear Regression baseline
- Use target encoding or frequency encoding for `model` as a third comparison point alongside one-hot and label encoding
- Add cross-validation rather than a single train/test split
- Residual analysis to check linear regression assumptions
- Deploy as a simple Streamlit app for live price estimation

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
