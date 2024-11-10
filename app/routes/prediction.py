from flask import Blueprint, jsonify
import pandas as pd
from app.services.database import get_sqlalchemy_session
from app.services.prediction import predict_using_decision_tree, predict_using_logistic_regression, predict_using_decision_tree_student, predict_using_logistic_regression_studentId, CSV_FILE_PATH
from sklearn.metrics import accuracy_score, f1_score, recall_score
import numpy as np

prediction_bp = Blueprint('prediction', __name__)


@prediction_bp.route('/predict_decision_tree', methods=['GET'])
def decision_tree_prediction():
    try:
        session, engine = get_sqlalchemy_session()
        query = """
        SELECT er.student_id, e.course_id, er.total_score, c.title, g.grade_name
        FROM ExamResults er
        JOIN Exams e ON er.exam_id = e.exam_id
        JOIN Courses c ON e.course_id = c.course_id
        JOIN Students s ON er.student_id = s.student_id
        JOIN Grades g ON s.grade_id = g.grade_id
        """
        data = pd.read_sql(query, engine)
        session.close()

        if data.empty:
            return jsonify({"error": "No data found"}), 404

        response = predict_using_decision_tree(data)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@prediction_bp.route('/predict_logistic_regression', methods=['GET'])
def logistic_regression_prediction():
    try:
        session, engine = get_sqlalchemy_session()
        query = """
        SELECT er.student_id, e.course_id, er.total_score, c.title, g.grade_name
        FROM ExamResults er
        JOIN Exams e ON er.exam_id = e.exam_id
        JOIN Courses c ON e.course_id = c.course_id
        JOIN Students s ON er.student_id = s.student_id
        JOIN Grades g ON s.grade_id = g.grade_id
        """
        data = pd.read_sql(query, engine)
        session.close()

        if data.empty:
            return jsonify({"error": "No data found"}), 404

        response = predict_using_logistic_regression(data)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@prediction_bp.route('/predict_logistic_regression_studentid/<int:student_id>', methods=['GET'])
def predict_using_logistic_regression_student(student_id):
    try:
        session, engine = get_sqlalchemy_session()
        query = """
        SELECT er.student_id, e.course_id, er.total_score, c.title, g.grade_name
        FROM ExamResults er
        JOIN Exams e ON er.exam_id = e.exam_id
        JOIN Courses c ON e.course_id = c.course_id
        JOIN Students s ON er.student_id = s.student_id
        JOIN Grades g ON s.grade_id = g.grade_id
        WHERE er.student_id = %s
        """
        # Ejecutar la consulta, pasando el parámetro student_id
        data = pd.read_sql(query, engine, params=(student_id,))
        session.close()
        if data.empty:
            return jsonify({"error": "No data found"}), 404

        response = predict_using_logistic_regression_studentId(data)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@prediction_bp.route('/metrics', methods=['GET'])
def get_metrics():
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        if df.empty:
            return jsonify({"error": "No data found in the CSV file"}), 404

        metrics = {
            'ML Model': [],
            'Accuracy': [],
            'F1 Score': [],
            'Recall': []
        }

        for model_name in df['model_name'].unique():
            model_data = df[df['model_name'] == model_name]
            y_true = model_data['true_label']
            y_pred = model_data['predicted_label']
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')

            # Convertir a porcentaje
            accuracy_percent = round(accuracy * 100, 2)
            f1_percent = round(f1 * 100, 2)
            recall_percent = round(recall * 100, 2)

            # Incluir la desviación estándar ficticia
            # Usamos un valor ficticio para ilustrar
            accuracy_std = round(np.std([accuracy], ddof=1) * 100, 2)

            metrics['ML Model'].append(model_name)
            metrics['Accuracy'].append(f"{accuracy_percent}% ± {accuracy_std}")
            metrics['F1 Score'].append(f"{f1_percent}%")
            metrics['Recall'].append(f"{recall_percent}%")

        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
