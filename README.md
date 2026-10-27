# car_price_predictor

# 🚗 Car Price Predictor

A machine learning practice project that predicts the estimated selling price of a used car based on its specifications, including company, model, year of purchase, fuel type, and kilometres driven.

The project uses **Linear Regression** to learn relationships between car features and prices and provides a simple interface for entering vehicle details and viewing a price estimate.

## 📌 Project Overview

Buying or selling a used car can be challenging because prices depend on several factors. This project explores how machine learning can estimate a car's price using historical used-car data.

The project covers the complete basic machine learning workflow, from data cleaning and preprocessing to model training, evaluation, and prediction through a Streamlit application.

## 🎯 Objectives

- Clean and preprocess a used-car dataset.
- Explore the relationship between car features and selling prices.
- Convert categorical variables into numerical features using One-Hot Encoding.
- Train a Linear Regression model.
- Evaluate the model using the R² score.
- Build an interactive web application for car price prediction.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and cleaning |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and preprocessing |
| Matplotlib / Seaborn | Data visualization, if used in the notebook |
| Streamlit | Interactive web interface |
| Google Colab | Model development and experimentation |
| Git & GitHub | Version control and project hosting |

## 📂 Project Structure

```text
car-price-predictor/
│
├── Untitled7.ipynb
├── quikr_car.csv
├── Cleaned car.csv
├── LinearRegressionModel.pkl
├── app.py
├── requirements.txt
└── README.md
```

*Note: Update the filenames in this structure to match the files included in your repository.*

## 📊 Dataset Description

The project uses a used-car dataset containing information about vehicles and their selling prices.

| Feature | Description |
|---|---|
| `name` | Car model name |
| `company` | Car manufacturer |
| `year` | Manufacturing year |
| `Price` | Selling price of the car |
| `kms_driven` | Distance travelled in kilometres |
| `fuel_type` | Fuel type, such as Petrol or Diesel |

The target variable is `Price`, which the model attempts to predict.

## ⚙️ Project Workflow

1. **Data loading:** Load the original CSV dataset.
2. **Data cleaning:** Handle invalid prices, missing values, and inconsistent data formats.
3. **Feature preparation:** Select the relevant car attributes
