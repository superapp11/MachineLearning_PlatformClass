import pandas as pd
from sklearn.linear_model import LogisticRegression

def train_logistic_regression(data):
    X = data[['student_id', 'course_id', 'total_score']]
    y = data['grade_name']
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X, y)
    return logistic_model
