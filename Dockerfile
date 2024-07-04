FROM python:3.8-slim-buster

WORKDIR /app

COPY api/requirements.txt /app/api/

RUN pip install -U pip && pip install -r /app/api/requirements.txt

COPY api/ /app/api/
COPY model/model.pkl /app/model/model.pkl
COPY initializer.sh /app/

RUN chmod +x /app/initializer.sh

EXPOSE 8000

ENTRYPOINT ["./initializer.sh"]
