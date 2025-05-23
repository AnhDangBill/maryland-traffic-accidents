﻿
# Maryland Traffic Accident Prediction

## Project Overview
This data science project predicts traffic accidents in Maryland using machine learning. By analyzing historical crash data, weather, time, and road characteristics, we build predictive models to identify high-risk scenarios and improve road safety outcomes.

## Motivation
As a Maryland driving school owner, I regularly observe how adverse conditions—like poor visibility or bad road design—endanger new drivers. With data analytics, we can go beyond instinct and experience to proactively prevent accidents, support policy changes, and improve driver training.

## Data Sources
- Maryland Road Closures Dataset
- MDTA Accidents Dataset (geolocation, timestamps, severity)
- Prince George's County Crash Summary Report
- NOAA historical weather data

## Methodology
- Data preprocessing and feature engineering
- Exploratory data analysis to find correlations
- Supervised learning models:
  - Random Forest
  - Logistic Regression
  - XGBoost Classifier
  - Deep Neural Network
  - Ensemble Voting Classifier
- SMOTE to handle class imbalance
- Evaluation using accuracy and confusion matrices
- Visualizations: feature importance, heatmaps, confusion matrix

## Key Findings
- **Time of day** is the strongest predictor: peak hours are 7–9 AM and 4–7 PM.
- **Precipitation increases accident likelihood by 32%**
- **Urban hotspots**, especially near Baltimore and the I-495 corridor, have elevated crash frequencies.
- **Ensemble model** achieved the best accuracy: **80.8%** (after cleaning and balancing).

## Project Structure

📦 maryland-traffic-accidents/  
├── data/ # Cleaned CSV files (not uploaded)  
├── notebooks/ # Jupyter notebooks  
│   ├── 01_preprocessing.ipynb  
│   ├── 02_modeling.ipynb  
│   └── 03_visualizations.ipynb  
├── figures/ # Saved images (feature importance, confusion matrix, etc.)  
├── requirements.txt # Python dependencies  
└── README.md # Project documentation  

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/AnhDangBill/maryland-traffic-accidents
cd maryland-traffic-accidents
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open the notebooks folder and run them in order:

* `01_preprocessing.ipynb` → clean and engineer features
* `02_modeling.ipynb` → train and evaluate models
* `03_visualizations.ipynb` → generate figures

---

## Author

**Anh Dang**
Senior, B.S. Information Science
University of Maryland
GitHub: [@AnhDangBill](https://github.com/AnhDangBill)

```

