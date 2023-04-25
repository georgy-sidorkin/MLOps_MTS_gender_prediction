FROM --platform=linux/amd64 python:3.9-slim-buster

WORKDIR /app/frontend
COPY . .


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        gcc

RUN pip install --upgrade pip && \
    pip install --ignore-installed -r requirements.txt


EXPOSE 8501

ENTRYPOINT [ "streamlit", "run" ]
CMD ["main.py", "--server.headless", "true", "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false"]
