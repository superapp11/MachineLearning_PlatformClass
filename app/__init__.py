from flask import Flask
from flask_cors import CORS
from app.routes.prediction import prediction_bp

def create_app():
    app = Flask(__name__)

    # Habilitar CORS para toda la aplicación
    CORS(app)

    app.register_blueprint(prediction_bp, url_prefix='/api')

    return app
