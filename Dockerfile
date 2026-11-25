FROM python:3.11-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY churn churn
RUN python -m churn.train   # train at build time so the image ships with a registered model
EXPOSE 8000
CMD ["uvicorn", "churn.serve:app", "--host", "0.0.0.0", "--port", "8000"]
