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
kind create cluster --name knative --config clusterconfig.yaml

#Load ICOW service container into Kind cluster
kind load docker-image --name knative dev.local/icow_service

kubectl apply --filename https://github.com/knative/serving/releases/download/v0.21.0/serving-crds.yaml
kubectl apply --filename https://github.com/knative/serving/releases/download/v0.21.0/serving-core.yaml


  kubectl patch configmap/config-network \
  --namespace knative-serving \
  --type merge \
  --patch '{"data":{"ingress.class":"kourier.ingress.networking.knative.dev"}}'

  kubectl patch configmap/config-domain \
    --namespace knative-serving \
    --type merge \
    --patch '{"data":{"127.0.0.1.nip.io":""}}'

helm install --set minio.service.type=NodePort --skip-crds --create-namespace icow -n icow deployment/icow

cat > kourier-patch.yaml <<EOF
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

kubectl apply -f kourier-patch.yaml

cat > service.yaml <<EOF
apiVersion: serving.knative.dev/v1 # Current version of Knative
kind: Service
metadata:
  name: helloworld-go # The name of the app
  namespace: default # The namespace the app will use
spec:
  template:
    spec:
      containers:
        - image: gcr.io/knative-samples/helloworld-go # The URL to the image of the app
          env:
            - name: TARGET # The environment variable printed out by the sample app
              value: "Hello Knative Serving is up and running with Kourier!!"
EOF


echo "Setup complete 🎉"
#Keep container alive indefinitely
tail -f /dev/null
