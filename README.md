# California Housing Price Prediction (Model Comparison)

This project builds an end-to-end machine learning pipeline to predict housing prices using the California Housing dataset.  
Multiple regression models are trained and evaluated to understand how different approaches perform on real-world, tabular data.

The project emphasizes **data preprocessing, model comparison, evaluation, visualization, and evidence-based model selection**.

---

## Project Objectives
- Clean and preprocess raw housing data
- Explore geographic and socioeconomic patterns in housing prices
- Train and compare multiple regression models
- Select the best-performing model using quantitative evaluation metrics
- Communicate results through visualizations

---

## Dataset
The dataset contains housing information for districts in California, including:
- Location (longitude, latitude)
- Housing characteristics (total rooms, bedrooms, households)
- Demographics (population, median income)
- Proximity to the ocean (categorical feature)
The target variable is median house value
- `housing.csv` — full raw dataset

---

## Data Preparation
The following preprocessing steps were applied:
- Handled missing values using imputation
- One-hot encoded categorical variables (e.g., `ocean_proximity`)
- Scaled numerical features using MinMaxScaler

---

## Models 

### Linear Regression (Baseline)
Linear Regression is the simplest model used in this project and serves as a baseline.  
It assumes a straight-line relationship between input features (such as income and location) and housing prices.
While it cannot capture complex interactions or nonlinear patterns, it provides a useful reference point to determine whether more advanced models are justified.
House prices change proportionally as features increase or decrease.

---

### K-Nearest Neighbors (KNN)
KNN predicts a house’s price based on the prices of the **most similar houses** in the dataset.  
Similarity is determined by distance in feature space.
This model can capture nonlinear relationships but may struggle with noisy data and does not generalize as well when the dataset is large or unevenly distributed.
Houses with similar characteristics tend to have similar prices.

---

### Random Forest Regressor
Random Forest is an ensemble model that builds many decision trees and averages their predictions.  
Each tree is trained on a random subset of the data, which improves robustness and reduces overfitting.
Random Forests are effective at modeling nonlinear relationships and feature interactions.
Ask many different decision trees for their opinion and average the result.

---

### XGBoost Regressor
XGBoost is a gradient-boosted tree model that builds trees sequentially, where each new tree focuses on correcting the errors made by previous trees.
It is highly optimized for tabular data and often achieves superior performance when tuned carefully.
Build trees step-by-step, each one learning from past mistakes.

---

## Model Evaluation
All models were evaluated using:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)

Evaluation was performed on **training and test sets**, with final model selection based on **test performance**.

---

## Results

### Test Set Performance (MAPE)

| Model | Test MAPE |
|------|-----------|
| Linear Regression | 0.290 |
| KNN | 0.233 |
| Random Forest | 0.210 |
| **XGBoost** | **0.179** |

### Model Comparison Visualization
<img width="870" height="483" alt="Screenshot 2025-12-21 at 7 39 38 PM" src="https://github.com/user-attachments/assets/73c5db4c-431b-47c7-b0b7-4856ad444a84" />

**XGBoost achieved the lowest test MAPE**, outperforming Random Forest, KNN, and the linear baseline.  
Based on this comparison, XGBoost was selected as the final model.

---

## Feature Importance
Feature importance analysis indicates that:
- **Median income** is the strongest predictor of housing prices
- **Location-related features** (longitude, latitude, inland vs coastal) also play a major role
These findings align with known real-world housing trends in California.
<img width="826" height="463" alt="Screenshot 2025-12-21 at 7 40 08 PM" src="https://github.com/user-attachments/assets/8dfa4ce7-509f-4502-b7ec-a6561909d646" />

---

## Geographic Visualization
Predicted housing prices were visualized on a California map and compared against ground truth values.  
The spatial distribution of predictions closely matches observed high-value coastal regions and lower inland prices.
<img width="1125" height="604" alt="Screenshot 2025-12-21 at 7 40 32 PM" src="https://github.com/user-attachments/assets/b74332a8-35b0-4dce-9b1c-1026cf31066e" />


---

## Repository Structure
- `Housing_Price_Prediction_EDA.ipynb` — data cleaning, preprocessing, and exploratory analysis
- `Housing_Price_Prediction_Models.ipynb` — model training, evaluation, and comparison
- `model_comparison_mape.png` — final model comparison chart
- `housing.csv`, `housing_train.csv`, `housing_test.csv`, `housing_test_y.csv` — dataset files

---

## Technologies Used
Python, Pandas, NumPy, scikit-learn, XGBoost, Matplotlib
