#! /bin/bash

# get directory where this is called from 
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null 2>&1 && pwd)"

# install the knative serving components and networking layer kourier
kubectl apply --filename https://github.com/knative/serving/releases/download/v0.21.0/serving-crds.yaml
kubectl apply --filename https://github.com/knative/serving/releases/download/v0.21.0/serving-core.yaml
kubectl apply --filename https://github.com/knative/net-kourier/releases/download/v0.21.0/kourier.yaml

# patch kourier to expose it via node port,
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: kourier
  namespace: kourier-system
  labels:
    networking.knative.dev/ingress-provider: kourier
    serving.knative.dev/release: "v0.21.0"
spec:
  ports:
  - name: http2
    port: 80
    protocol: TCP
    targetPort: 8080
    nodePort: 31080
  - name: https
    port: 443
    protocol: TCP
    targetPort: 8443
    nodePort: 31443
  selector:
    app: 3scale-kourier-gateway
  type: NodePort
EOF


# use kourier as default networking layer for knative services
kubectl patch configmap/config-network \
  --namespace knative-serving \
  --type merge \
  --patch '{"data":{"ingress.class":"kourier.ingress.networking.knative.dev"}}'

# use 127.0.0.1.nip.io as dns to reach knative services
kubectl patch configmap/config-domain \
  --namespace knative-serving \
  --type merge \
  --patch '{"data":{"127.0.0.1.nip.io":""}}'

echo ""
echo "Knative serving installed. Waiting for pods to be ready 🐢"
echo ""

kubectl wait --for=condition=ready pod --all -n knative-serving
kubectl wait --for=condition=ready pod --all -n kourier-system

helm install -n icow --create-namespace icow $DIR/../deployment/icow
