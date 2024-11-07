# Customer Segmentation using RFM Analysis and Logistic Regression

This project uses RFM (Recency, Frequency, Monetary) analysis to classify customers into segments based on their purchasing behavior. We leverage logistic regression to predict customer segments based on RFM scores and additional features.

## Project Overview

Customer segmentation is an essential part of targeted marketing and personalization. By classifying customers into different segments based on their transaction history, businesses can design specific marketing strategies for each group, boosting retention and satisfaction.

In this project, we:
- Conducted RFM analysis to score each customer.
- Defined a binary target variable (e.g., “high value” or “low value”) based on RFM scores.
- Trained a logistic regression model to classify customers as high-value or low-value, depending on their RFM behavior.

## RFM Model

The RFM model assigns scores to customers based on:
- **Recency**: How recently a customer made a purchase.
- **Frequency**: How often a customer makes a purchase.
- **Monetary**: The amount a customer spends on purchases.

## Features and Target Variable

The dataset used contains the following main features:
- `Recency`: Number of days since the last purchase.
- `Frequency`: Number of transactions made by a customer.
- `Monetary`: Total monetary value of the customer's purchases.
- `recency_score`, `frequency_score`, `monetary_score`: Scores assigned to each RFM factor.
- **Target**: A binary label indicating whether a customer has an RFM score of 10 or above (`target=1`) or below (`target=0`).

## Project Structure

The main components of this project include:

- **Data Loading and Preprocessing**: Load data from CSV and define the target variable based on RFM scores.
- **Data Splitting**: Split the data into training and test sets.
- **Feature Scaling**: Standardize the data using `StandardScaler`.
- **Model Training**: Train a logistic regression model on the training data.
- **Evaluation**: Assess model accuracy and other metrics like precision, recall, and F1-score.

## Installation

To run this project, you need Python 3.6+ and the following Python libraries:
- `pandas`
- `scikit-learn`

You can install these dependencies with:
```bash
pip install pandas scikit-learn
```

## Usage

1. **Clone the repository** (or download the project files):
   ```bash
   git clone https://github.com/mdakilahmed/loan_prob_regession
   cd customer-segmentation
   ```

2. **Place your data** in the root folder or specify the correct path for your dataset in the code.

3. **Run the code** by executing the Python script:
   ```bash
   jupyter notebook resges.ipynb
   ```

### Sample Code Structure

The primary code script contains the following main steps:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load data and define target
file_path = './loan_rfm.csv'
df = pd.read_csv(file_path)
df['target'] = df['RFM_Score'].apply(lambda x: 1 if x >= 10 else 0)

# Define features and target
X = df[['Recency', 'Frequency', 'Monetary', 'recency_score', 'frequency_score', 'monetary_score']]
y = df['target']

# Split, scale, and train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=200, solver='saga')
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

## Data Insights

After loading and preprocessing, here are some key insights:
- **Data Types and Missing Values**: Ensure all columns have appropriate types and handle any missing values.
- **Target Distribution**: Review the distribution of the target variable to check for class imbalances.

## Model Training and Evaluation

The logistic regression model achieved a performance measured in terms of accuracy and classification metrics. Further evaluations include:

- **Accuracy**: Percentage of correct predictions out of total predictions.
- **Precision, Recall, F1-score**: Metrics provided in the classification report to assess performance for both classes.
- **Confusion Matrix**: A matrix showing the true positives, true negatives, false positives, and false negatives.

## Future Enhancements

Possible future improvements include:
- Experimenting with other classification algorithms (e.g., Random Forest, SVM).
- Using techniques like oversampling or undersampling if there's a class imbalance.
- Hyperparameter tuning for optimizing logistic regression performance.

