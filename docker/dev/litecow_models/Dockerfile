FROM python:3.7
RUN python3.7 -m pip install --upgrade pip && \
    python3.7 -m pip install poetry
WORKDIR /litecow_models
COPY ./src/litecow_models/pyproject.toml ./src/litecow_models/poetry.lock /litecow_models/
RUN poetry install
COPY ./src/litecow_models/ /litecow_models
