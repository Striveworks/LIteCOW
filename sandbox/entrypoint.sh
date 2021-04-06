#/bin/sh
set -e

unset DOCKER_HOST

#Build ICOW service container
docker build -t dev.local/icow_service -f docker/server/Dockerfile src

#Write Kind cluster config
cat > clusterconfig.yaml <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  extraPortMappings:
  - containerPort: 31080
    hostPort: 80
  - containerPort: 31443
    hostPort: 443
  - containerPort: 32000
    hostPort: 9000
EOF



#Create Kind cluster
kind create cluster --name knative --config clusterconfig.yaml --wait 5m

#Load ICOW service container into Kind cluster
kind load docker-image --name knative dev.local/icow_service

kubectl apply --filename https://github.com/knative/serving/releases/download/v0.21.0/serving-crds.yaml
kubectl apply --filename https://github.com/knative/serving/releases/download/v0.21.0/serving-core.yaml
kubectl apply --filename https://github.com/knative/net-kourier/releases/download/v0.21.0/kourier.yaml

  kubectl patch configmap/config-network \
  --namespace knative-serving \
  --type merge \
  --patch '{"data":{"ingress.class":"kourier.ingress.networking.knative.dev"}}'

  kubectl patch configmap/config-domain \
    --namespace knative-serving \
    --type merge \
    --patch '{"data":{"127.0.0.1.nip.io":""}}'

cat <<EOF | kubectl apply -f -
  apiVersion: v1
  kind: Service
  metadata:
    name: kourier
    namespace: kourier-system
    labels:
      networking.knative.dev/ingress-provider: kourier
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


echo ""
echo "Knative serving installed. Waiting for pods to be ready 🐢"
echo ""

kubectl wait --for=condition=ready pod --all -n knative-serving
kubectl wait --for=condition=ready pod --all -n kourier-system

echo ""
echo "Knative serving ready. Installing litecow 💡🐄"
echo ""

helm install --set minio.service.type=NodePort --create-namespace icow -n icow deployment/icow

kubectl wait --for=condition=ready ksvc --all -n icow
echo ""
echo "litecow ready 🎉 💡🐄 🎉"
echo ""

echo ""
echo ""
echo ""
echo "🎉Setup complete 🎉"
echo ""
echo ""
#Keep container alive indefinitely
tail -f /dev/null
