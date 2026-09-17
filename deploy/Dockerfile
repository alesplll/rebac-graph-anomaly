# The service only. Training happens on a workstation and its artefact is mounted in.
FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY web ./web

RUN uv sync --extra cpu --extra service --extra neo4j --extra ml --frozen

EXPOSE 8000
CMD ["uv", "run", "rga", "serve", "--config", "configs/service/opens3.yaml", "--host", "0.0.0.0"]
