FROM python:3.7.9 as builder
RUN pip install poetry furo myst-parser Sphinx sphinx-copybutton mkdocs-material
COPY src/litecow/pyproject.toml src/litecow/
COPY src/litecow_models/pyproject.toml src/litecow_models/
RUN poetry config virtualenvs.create false && \
    mkdir /src/litecow/litecow && \
    touch /src/litecow/litecow/__init__.py && \
    mkdir /src/litecow_models/litecow_models && \
    touch /src/litecow_models/litecow_models/__init__.py && \
    (cd /src/litecow && poetry install -E server-cpu) && \
    (cd /src/litecow_models && poetry install)
COPY src/litecow/ src/litecow/
COPY src/litecow_models/ src/litecow_models/
COPY docs/ docs/
RUN cd docs/ && \
    sphinx-apidoc -f -o source/litecow ../src/litecow && \
    sphinx-apidoc -f -o source/models ../src/litecow_models ../src/litecow_models/tests && \
    make html && \
    mkdocs build 


FROM httpd:alpine
COPY --from=builder docs/build/html /usr/local/apache2/htdocs/
COPY --from=builder docs/site/index.html /usr/local/apache2/htdocs/
COPY --from=builder docs/site/assets /usr/local/apache2/htdocs/assets
COPY --from=builder docs/site/images /usr/local/apache2/htdocs/images
COPY --from=builder docs/site/extra.css /usr/local/apache2/htdocs/
