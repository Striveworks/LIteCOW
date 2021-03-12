FROM python:3.7.9 as builder
RUN python3.7 -m pip install --upgrade pip && \
    python3.7 -m pip install poetry && \
    poetry config virtualenvs.in-project true
COPY ./litecow_common/ litecow_common/
COPY ./litecow_server/pyproject.toml litecow_server/pyproject.toml
RUN (cd /litecow_server && poetry install -E gpu)
COPY ./litecow_server/litecow_server litecow_server/litecow_server
RUN (cd /litecow_server; poetry install -E gpu)

FROM nvidia/cuda:11.2.1-cudnn8-runtime
RUN apt update -y && \
	DEBIAN_FRONTEND=noninteractive apt install -y software-properties-common && \
	add-apt-repository ppa:deadsnakes/ppa && \
	apt install -y python3.7 && \
	apt remove -y software-properties-common && \
	apt autoremove -y && \
	ln -s /usr/bin/python3.7 /usr/local/bin

COPY --from=builder /litecow_server/ /litecow_server/
ENV PATH="/litecow_server/.venv/bin:$PATH"
