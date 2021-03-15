FROM python:3.7.9 as builder
RUN python3.7 -m pip install --upgrade pip && \
    python3.7 -m pip install poetry && \
    poetry config virtualenvs.in-project true
COPY ./litecow/pyproject.toml litecow/pyproject.toml
RUN (cd /litecow && poetry install -E server-gpu)
COPY ./litecow/litecow ./litecow/litecow
RUN (cd /litecow; poetry install -E server-gpu)

FROM nvidia/cuda:11.2.1-cudnn8-runtime
RUN apt update -y && \
	DEBIAN_FRONTEND=noninteractive apt install -y software-properties-common && \
	add-apt-repository ppa:deadsnakes/ppa && \
	apt install -y python3.7 && \
	apt remove -y software-properties-common && \
	apt autoremove -y && \
	ln -s /usr/bin/python3.7 /usr/local/bin

COPY --from=builder /litecow/ /litecow/
ENV PATH="/litecow/.venv/bin:$PATH"
