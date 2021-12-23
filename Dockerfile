FROM python:3.9-slim
RUN mkdir /app
WORKDIR /app
ADD requirements.txt application.py entrypoint.sh /app/
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install -r requirements.txt
EXPOSE 5000
RUN chmod +x ./entrypoint.sh
ENTRYPOINT ["sh", "entrypoint.sh"]