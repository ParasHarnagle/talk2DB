FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py constants.py test_sql.py gunicorn.conf.py /app/

EXPOSE 9000
CMD ["gunicorn", "app:app", "-c", "gunicorn.conf.py"]
