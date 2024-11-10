# Usa una imagen de Python como base
FROM python:3.11.3

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia el archivo de requisitos a la imagen
COPY requirements.txt .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código a la imagen
COPY . .

# Expone el puerto 5000 para que Gunicorn pueda escuchar las solicitudes
EXPOSE 5000

# Comando para ejecutar la aplicación con Gunicorn cuando el contenedor se inicie
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:create_app()"]
