FROM python:3.12

WORKDIR /app

COPY requirements.txt .
COPY . .

RUN pip install -r requirements.txt

EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0"]

