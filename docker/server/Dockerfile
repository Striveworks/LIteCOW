FROM python:3.7.9 as builder
RUN python3.7 -m pip install --upgrade pip && \
    python3.7 -m pip install poetry && \
    poetry config virtualenvs.in-project true
COPY ./src/litecow_common litecow_common
COPY ./src/litecow_server/pyproject.toml litecow_server/pyproject.toml
RUN (cd /litecow_server; poetry install)
COPY ./src/litecow_server/litecow_server litecow_server/litecow_server
RUN (cd /litecow_server; poetry install)

FROM python:3.7.9-slim
COPY --from=builder /litecow_server/ /litecow_server/
ENV PATH="/litecow_server/.venv/bin:$PATH"
