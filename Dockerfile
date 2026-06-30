FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY backend /app/backend
COPY frontend /app/frontend
COPY data /app/data
COPY app.py /app/app.py

CMD streamlit run app.py --server.port ${PORT:-8000} --server.address 0.0.0.0
