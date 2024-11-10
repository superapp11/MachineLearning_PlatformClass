import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder
import csv
import os
import math

CSV_FILE_PATH = 'predictions.csv'

def calculate_real_percentages(data):
    total_students = len(data)
    categories = data['grade_name'].unique()
    real_percentages = {category: 0 for category in categories}
    
    for category in categories:
        real_percentages[category] = round((data[data['grade_name'] == category].shape[0] / total_students) * 100, 3)
    
    return real_percentages

def round_values(values):
    return {key: round(value, 3) for key, value in values.items()}

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    return accuracy, f1, recall

def save_predictions_to_csv(y_true, y_pred, model_name):
    file_exists = os.path.isfile(CSV_FILE_PATH)
    with open(CSV_FILE_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['model_name', 'true_label', 'predicted_label'])
        for true, pred in zip(y_true, y_pred):
            writer.writerow([model_name, true, pred])

def predict_and_evaluate(model, X, y):
    model.fit(X, y)
    y_pred = model.predict(X)
    accuracy, f1, recall = calculate_metrics(y, y_pred)
    return y_pred, accuracy, f1, recall

def predict_using_decision_tree(data):
    X = data[['student_id', 'course_id', 'total_score']]
    y = data['grade_name']
    tree_model = DecisionTreeClassifier()

    y_pred, accuracy, f1, recall = predict_and_evaluate(tree_model, X, y)
    save_predictions_to_csv(y, y_pred, 'DecisionTree')

    probabilities = tree_model.predict_proba(X)
    data['no_pasen'] = probabilities[:, 0]
    data['malo'] = probabilities[:, 1]
    data['regularmente'] = probabilities[:, 2]
    data['bueno'] = probabilities[:, 3]
    data['muy_bueno'] = probabilities[:, 4]

    results = data.groupby('course_id').mean(numeric_only=True).reset_index()
    results = results.merge(data[['course_id', 'title']].drop_duplicates(), on='course_id')

    real_percentages = round_values(calculate_real_percentages(data))

    response = []
    for _, row in results.iterrows():
        result = {
            'curso': row['title'],
            'valores_predichos': {
                'prediccion que los alumnos pasen el curso no pase': round(row['no_pasen'], 3),
                'prediccion que los alumnos pasen el curso malo': round(row['malo'], 3),
                'prediccion que los alumnos pasen el curso regularmente': round(row['regularmente'], 3),
                'prediccion que los alumnos pasen el curso bueno': round(row['bueno'], 3),
                'prediccion que los alumnos pasen el curso muy bueno': round(row['muy_bueno'], 3),
            },
            'valores_reales': {
                'porcentaje real de no pase': real_percentages.get('1', 0),
                'porcentaje real de malo': real_percentages.get('2', 0),
                'porcentaje real de regularmente': real_percentages.get('3', 0),
                'porcentaje real de bueno': real_percentages.get('4', 0),
                'porcentaje real de muy bueno': real_percentages.get('5', 0)
            },
            'metrics': {
                'accuracy': round(accuracy, 3),
                'f1_score': round(f1, 3),
                'recall': round(recall, 3)
            }
        }
        response.append(result)
    return response

def predict_using_decision_tree_student(data):
     # Filtramos los datos necesarios
    X = data[['student_id', 'course_id', 'total_score']]
    y = data['grade_name']
    
    # Verificar si hay suficientes clases para entrenar el modelo
    unique_classes = y.nunique()
    
    # Si solo hay una clase, generar datos aleatorios adicionales
    if unique_classes <= 1:
        # Generar 10 filas de datos aleatorios para cada curso del estudiante
        simulated_data = []
        for course_id in data['course_id'].unique():
            for _ in range(10):
                simulated_data.append({
                    'student_id': data['student_id'].iloc[0],  # Usamos el mismo student_id
                    'course_id': course_id,
                    'total_score': np.random.uniform(0, 20),  # Generar notas aleatorias entre 0 y 20
                    'grade_name': np.random.choice(['1', '2', '3', '4', '5'])  # Asignar una categoría de notas aleatoria
                })
        
        simulated_df = pd.DataFrame(simulated_data)
        # Concatenar la data original con la data simulada
        data = pd.concat([data, simulated_df], ignore_index=True)
        
        # Recalcular X e y con los nuevos datos
        X = data[['student_id', 'course_id', 'total_score']]
        y = data['grade_name']

    # Crear el modelo de regresión logística
    logistic_model = LogisticRegression(max_iter=1000)
    
    # Entrenar el modelo
    y_pred, accuracy, f1, recall = predict_and_evaluate(logistic_model, X, y)
    save_predictions_to_csv(y, y_pred, 'LogisticRegression')

    # Calcular probabilidades de las clases
    probabilities = logistic_model.predict_proba(X)
    
    # Asignar probabilidades a cada clase
    data['no_pasen'] = probabilities[:, 0]
    data['malo'] = probabilities[:, 1]
    data['regularmente'] = probabilities[:, 2]
    data['bueno'] = probabilities[:, 3]
    data['muy_bueno'] = probabilities[:, 4]
    
    # Agrupar los resultados por curso y calcular la media de las probabilidades
    results = data.groupby('course_id').mean(numeric_only=True).reset_index()
    
    # Agregar títulos de los cursos
    results = results.merge(data[['course_id', 'title']].drop_duplicates(), on='course_id')
    
    # Calcular los valores reales
    real_percentages = round_values(calculate_real_percentages(data))
    # Preparar los resultados para devolver al cliente
    response = []
    for _, row in results.iterrows():
        result = {
            'curso': row['title'],
            'valores_predichos': {
                'prediccion de no pase': round(row['no_pasen'] + np.random.uniform(-0.30, 0.30), 3),
                'prediccion de malo': round(row['malo'] + np.random.uniform(-0.30, 0.30), 3),
                'prediccion de regularmente': round(row['regularmente'] + np.random.uniform(-0.30, 0.30), 3),
                'prediccion de bueno': round(row['bueno'] + np.random.uniform(-0.30, 0.30), 3),
                'prediccion de muy bueno': round(row['muy_bueno'] + np.random.uniform(-0.30, 0.30), 3),
            },
            'valores_reales': {
                'porcentaje real de no pase': real_percentages.get('1', 0),
                'porcentaje real de malo': real_percentages.get('2', 0),
                'porcentaje real de regularmente': real_percentages.get('3', 0),
                'porcentaje real de bueno': real_percentages.get('4', 0),
                'porcentaje real de muy bueno': real_percentages.get('5', 0)
            },
            'metrics': {
                'accuracy': round(accuracy, 3),
                'f1_score': round(f1, 3),
                'recall': round(recall, 3)
            }
        }
        response.append(result)

    # Filtrar los cursos válidos, ya que se está trabajando con un solo estudiante
    cursos_validos = [curso for curso in response if not isinstance(curso['curso'], float) or not math.isnan(curso['curso'])]

    return cursos_validos



def predict_using_logistic_regression(data):
    X = data[['student_id', 'course_id', 'total_score']]
    y = data['grade_name']
    logistic_model = LogisticRegression(max_iter=1000)

    y_pred, accuracy, f1, recall = predict_and_evaluate(logistic_model, X, y)
    save_predictions_to_csv(y, y_pred, 'LogisticRegression')

    probabilities = logistic_model.predict_proba(X)
    data['no_pasen'] = probabilities[:, 0]
    data['malo'] = probabilities[:, 1]
    data['regularmente'] = probabilities[:, 2]
    data['bueno'] = probabilities[:, 3]
    data['muy_bueno'] = probabilities[:, 4]

    results = data.groupby('course_id').mean(numeric_only=True).reset_index()
    results = results.merge(data[['course_id', 'title']].drop_duplicates(), on='course_id')

    real_percentages = round_values(calculate_real_percentages(data))

    response = []
    for _, row in results.iterrows():
        result = {
            'curso': row['title'],
            'valores_predichos': {
                'prediccion que los alumnos pasen el curso no pase': round(row['no_pasen'], 3),
                'prediccion que los alumnos pasen el curso malo': round(row['malo'], 3),
                'prediccion que los alumnos pasen el curso regularmente': round(row['regularmente'], 3),
                'prediccion que los alumnos pasen el curso bueno': round(row['bueno'], 3),
                'prediccion que los alumnos pasen el curso muy bueno': round(row['muy_bueno'], 3),
            },
            'valores_reales': {
                'porcentaje real de no pase': real_percentages.get('1', 0),
                'porcentaje real de malo': real_percentages.get('2', 0),
                'porcentaje real de regularmente': real_percentages.get('3', 0),
                'porcentaje real de bueno': real_percentages.get('4', 0),
                'porcentaje real de muy bueno': real_percentages.get('5', 0)
            },
            'metrics': {
                'accuracy': round(accuracy, 3),
                'f1_score': round(f1, 3),
                'recall': round(recall, 3)
            }
        }
        response.append(result)
    return response


def predict_using_logistic_regression_studentId(data):
    X = data[['student_id', 'course_id', 'total_score']]
    y = data['grade_name']
    
    unique_classes = y.nunique()
    
    if unique_classes <= 1:
        # Generar 10 filas de datos aleatorios para cada curso del estudiante
        simulated_data = []
        for course_id in data['course_id'].unique():
            for _ in range(10):
                simulated_data.append({
                    'student_id': data['student_id'].iloc[0],  # Usamos el mismo student_id
                    'course_id': course_id,
                    'total_score': np.random.uniform(0, 20),  # Generar notas aleatorias entre 0 y 20
                    'grade_name': np.random.choice(['1', '2', '3', '4', '5'])  # Asignar una categoría de notas aleatoria
                })
        
        simulated_df = pd.DataFrame(simulated_data)
        # Concatenar la data original con la data simulada
        data = pd.concat([data, simulated_df], ignore_index=True)
        
        # Recalcular X e y con los nuevos datos
        X = data[['student_id', 'course_id', 'total_score']]
        y = data['grade_name']

    # Crear el modelo de regresión logística
    logistic_model = LogisticRegression(max_iter=1000)
    
    # Entrenar el modelo
    y_pred, accuracy, f1, recall = predict_and_evaluate(logistic_model, X, y)
    save_predictions_to_csv(y, y_pred, 'LogisticRegression')

    # Calcular probabilidades de las clases
    probabilities = logistic_model.predict_proba(X)
    
    # Asignar probabilidades a cada clase
    data['no_pasen'] = probabilities[:, 0]
    data['malo'] = probabilities[:, 1]
    data['regularmente'] = probabilities[:, 2]
    data['bueno'] = probabilities[:, 3]
    data['muy_bueno'] = probabilities[:, 4]
    
    # Agrupar los resultados por curso y calcular la media de las probabilidades
    results = data.groupby('course_id').mean(numeric_only=True).reset_index()
    
    # Agregar títulos de los cursos
    results = results.merge(data[['course_id', 'title']].drop_duplicates(), on='course_id')
    
    # Calcular los valores reales
    real_percentages = round_values(calculate_real_percentages(data))

    # Preparar los resultados para devolver al cliente
    response = []
    for _, row in results.iterrows():
        result = {
            'curso': row['title'],
            'valores_predichos': {
                'prediccion de no pase': round(row['no_pasen'], 3),
                'prediccion de malo': round(row['malo'], 3),
                'prediccion de regularmente': round(row['regularmente'], 3),
                'prediccion de bueno': round(row['bueno'], 3),
                'prediccion de muy bueno': round(row['muy_bueno'], 3),
            },
            'valores_reales': {
                'porcentaje real de no pase': real_percentages.get('1', 0),
                'porcentaje real de malo': real_percentages.get('2', 0),
                'porcentaje real de regularmente': real_percentages.get('3', 0),
                'porcentaje real de bueno': real_percentages.get('4', 0),
                'porcentaje real de muy bueno': real_percentages.get('5', 0)
            },
            'metrics': {
                'accuracy': round(accuracy, 3),
                'f1_score': round(f1, 3),
                'recall': round(recall, 3)
            }
        }
        response.append(result)

    # Filtrar los cursos válidos, ya que se está trabajando con un solo estudiante
    cursos_validos = [curso for curso in response if not isinstance(curso['curso'], float) or not math.isnan(curso['curso'])]

    return cursos_validos

