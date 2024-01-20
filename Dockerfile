FROM python:3.11-slim

RUN pip install pipenv

RUN pip install gunicorn
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "/"]
COPY ["model.bin", "/"]
COPY ["X_test.pkl", "/"]

EXPOSE 8000

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:8000", "predict:app"]

#to build, go to directory in CMD and run (excl quotes): "docker build -t model ."
#to run, go to directory in CMD and run (excl quotes): "docker run -it --rm -p 8000:8000 model"