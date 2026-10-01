FROM python:3.12-slim

WORKDIR /app

# Dependencies first: this layer is rebuilt only if requirements change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then the project itself
COPY pyproject.toml cfg.yaml ./
COPY campaigns ./campaigns
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

ENTRYPOINT ["flroad"]
CMD ["--help"]
