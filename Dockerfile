# Gunakan image Python sebagai base
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy semua file ke work directory
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port yang akan digunakan Flask
EXPOSE 80

# Perintah untuk menjalankan aplikasi Flask
CMD ["python", "app.py"]
