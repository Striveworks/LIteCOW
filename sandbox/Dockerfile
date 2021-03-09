FROM docker:dind

RUN apk add curl \
    && curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.10.0/kind-linux-amd64 \
    && chmod +x ./kind \
    && mv ./kind /usr/local/bin/kind \
    && curl -LO https://dl.k8s.io/release/v1.20.0/bin/linux/amd64/kubectl \
    && chmod +x ./kubectl \
    && mv ./kubectl /usr/local/bin/kubectl

COPY . /icow-lite

WORKDIR /icow-lite

CMD ["./sandbox/entrypoint.sh"]
