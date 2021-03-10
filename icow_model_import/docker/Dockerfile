FROM python:3.7
RUN python3.7 -m pip install --upgrade pip && \
    python3.7 -m pip install poetry
WORKDIR /icow_model_import
COPY pyproject.toml poetry.lock /icow_model_import/
RUN poetry install
COPY . /icow_model_import
