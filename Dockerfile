FROM ambolt/emily:1.0.0-tf-slim

WORKDIR /workspace

# Install Pip requirements
COPY requirements.txt requirements.txt
RUN pip install --disable-pip-version-check -r requirements.txt

COPY . .

