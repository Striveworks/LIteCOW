FROM python:3.7.9 as builder
RUN python3.7 -m pip install --upgrade pip && \
    python3.7 -m pip install poetry && \
    poetry config virtualenvs.in-project true
COPY ./litecow/pyproject.toml litecow/pyproject.toml
RUN (cd /litecow && poetry install -E server-cpu)
COPY ./litecow/litecow ./litecow/litecow
RUN (cd /litecow; poetry install -E server-cpu)

FROM python:3.7.9-slim
COPY --from=builder /litecow/ /litecow/
ENV PATH="/litecow/.venv/bin:$PATH"
