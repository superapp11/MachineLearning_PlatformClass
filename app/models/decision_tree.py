import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(data):
    X = data[['student_id', 'course_id', 'total_score']]
    y = data['grade_name']
    tree_model = DecisionTreeClassifier()
    tree_model.fit(X, y)
    return tree_model
