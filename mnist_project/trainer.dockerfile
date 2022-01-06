# Base image
FROM python:3.7-slim

# install python 
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/*

# copy over our application (the essential parts)
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY reports/ reports/

# set the working directory
WORKDIR /

# add commands that install the dependencies
RUN pip install -r requirements.txt --no-cache-dir

# The entrypoint is the application that we want to run when the image is being executed
ENTRYPOINT ["python", "-u", "src/models/train_model.py"]