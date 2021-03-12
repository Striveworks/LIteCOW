FROM docker:dind

RUN apk add  --no-cache musl-dev linux-headers g++ curl bash python3 python3-dev py3-pip 	zlib-dev jpeg-dev\
    && curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.10.0/kind-linux-amd64 \
    && chmod +x ./kind \
    && mv ./kind /usr/local/bin/kind \
    && curl -LO https://dl.k8s.io/release/v1.20.0/bin/linux/amd64/kubectl \
    && chmod +x ./kubectl \
    && mv ./kubectl /usr/local/bin/kubectl \
    && curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3 \
    && chmod 700 get_helm.sh \
    && ./get_helm.sh \
    && curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 - \
    && pip install --upgrade pip \
    && pip install Pillow

COPY . /icow-lite

WORKDIR /icow-lite

CMD ["./sandbox/entrypoint.sh"]
