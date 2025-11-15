FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1

COPY flask_app/ /app/

COPY bow_vectorizer.pkl /app/bow_vectorizer.pkl
COPY logreg_bow_model.pkl /app/logreg_bow_model.pkl


RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

CMD ["python", "app.py"]