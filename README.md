# Customer Churn Prediction

This project aims to predict whether a customer will churn from a telecommunications company using the "Telco Customer Churn" dataset from Kaggle.

## Project Overview

Customer churn is a critical problem for many businesses, particularly in telecommunications, where the cost of acquiring new customers often exceeds the cost of retaining existing ones. This project uses a logistic regression model to predict customer churn and evaluates its performance using various metrics.

## Project Steps

1. **Data Loading and Exploration**:
   - Loaded the dataset and explored its features and the target variable.
   - Visualized the distribution of the target variable.
   
2. **Data Preprocessing**:
   - Handled missing values and encoded categorical variables.
   - Scaled the numerical features.
   
3. **Model Training**:
   - Trained a logistic regression model with L2 regularization to prevent overfitting.
   
4. **Model Evaluation**:
   - Evaluated the model using metrics like accuracy, precision, recall, and F1 score.
   - Visualized the confusion matrix and ROC curve.

## Data

The dataset used in this project is the "Telco Customer Churn" dataset from Kaggle. It contains information about customer behavior, including services signed up for, account information, and demographics.

## Model

The model used is a logistic regression model with L2 regularization. The model was chosen for its simplicity and effectiveness in binary classification tasks.

## Evaluation Metrics

The model was evaluated using the following metrics:

- **Accuracy**: The proportion of correctly predicted instances.
- **Precision**: The proportion of positive predictions that were actually correct.
- **Recall**: The proportion of actual positives that were correctly predicted.
- **F1 Score**: The harmonic mean of precision and recall.

## Results

The model's performance on the test set was as follows:

- **Accuracy**: 0.80
- **Precision**: 0.73
- **Recall**: 0.65
- **F1 Score**: 0.69

## Conclusion

The project demonstrated that a logistic regression model can reasonably predict customer churn. However, there's potential for improvement using more advanced techniques or hyperparameter tuning.

## Dependencies

The project uses the following dependencies:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## Instructions

1. Clone the repository.
2. Install the dependencies using `pip install -r requirements.txt`.
3. Open and run the Jupyter Notebook `Customer_churn_prediction.ipynb`.

## License

This project is licensed under the MIT License.
