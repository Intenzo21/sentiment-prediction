# Sentiment Prediction Platform

For the purpose of the project, we considered a real-life balanced dataset consisting of 50,000 labeled gourmet food reviews from Amazon extracted from the work of McAuley and Leskovec. A food review is labeled as 1 (positive) if it received four or more stars and 0 (negative) otherwise. 

We developed, trained, and tuned a number of Machine Learning (ML) models such as Logistic Regression, Support Vector Machine (SVM), and Long Short-Term Memory (LSTM). These models were then evaluated using a holdout test set previously created from the whole available data. For the evaluation, we considered various metrics such as accuracy, precision, recall, ROC AUC score, etc. Finally, we compared the models using to determine the best performing model. The best model (Logistic Regression) was then deployed using Flask and Heroku.
