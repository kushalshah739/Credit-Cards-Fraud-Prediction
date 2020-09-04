# Credit-Cards-Fraud-Prediction

Created a machine learning model in Python that predicts the fraud credit cards using the Light Gradient Boosting Machine algorithm. 

Overview: 

- Credit card frauds have significantly increased over the past few years. To combat such crimes, cyber security has highly been invested by several banks and corporations with the objective to minimize fraud transactions. Recently, the trends and attention have shifted to Machine Learning which can train a model and predict a fraud transaction from the historical data.

- This project is based on a machine learning model in Python that predicts the fraud credit cards using the Light Gradient Boosting Machine algorithm and the key parameters or factors.

- The dataset was initially split in 90-10 ratio where 90% is used for the training set and 10% is used as a test set. The 90% dataset was further split into a 70-30 ratio in Pycaret's setup() function. 

- The model's performance was based on the AUC value rather than the accuracy score because:
  - Accuracy ignores probability estimations of classification in favor of class labels.
  - ROC curves show the trade off between false positive and true positive rates.
  - AUC of ROC is a better measure than accuracy.
  - AUC as a criteria for comparing learning algorithms.
  - AUC replaces accuracy when comparing classifiers.
  
- Lastly, the model had a training AUC of 0.9935 and a testing AUC of 0.9926.



Dataset:

- The datasets contains transactions made by credit cards in September 2013 by european cardholders.
It presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. This dataset was balanced using the undersampling technique which equated the Fradulents and Non-Fradulents. 

- It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, the original feature names are hidden along with more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. 

- Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.



