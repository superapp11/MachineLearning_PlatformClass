import mysql.connector
from mysql.connector import Error
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URI = 'mysql+mysqlconnector://username:password@localhost/udemy'

def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="username",
            password="password",
            database="udemy"
        )
        if connection.is_connected():
            print("Connected to the database")
        return connection
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return None

def get_sqlalchemy_session():
    engine = create_engine(DATABASE_URI)
    Session = sessionmaker(bind=engine)
    session = Session()
    return session, engine
