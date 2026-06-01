FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY configs ./configs
COPY scripts ./scripts
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install -e .

CMD ["python", "scripts/run_all.py"]
