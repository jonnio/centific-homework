FROM python:3.10.14

ENV PYTHONUNBUFFERED=True

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV APP_HOME=/root
WORKDIR $APP_HOME
COPY /main $APP_HOME/app
COPY /build/tokenizer.pth $APP_HOME/app/build/tokenizer.pth
COPY /build/trained_model.pth $APP_HOME/app/build/trained_model.pth

ENV TOKEN_PATH=$APP_HOME/app/build/tokenizer.pth
ENV MODEL_PATH=$APP_HOME/app/build/trained_model.pth

EXPOSE 8080
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080"]
