FROM python:3.9-slim-buster

RUN apt update -y && apt install -y awscli

WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

CMD ["sh", "-c", "python main.py && streamlit run app.py"]