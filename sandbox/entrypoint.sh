#/bin/sh
set -e

unset DOCKER_HOST

#Build ICOW service container
docker build -t dev.local/icow_service:0.1 -f docker/server/Dockerfile .

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
EOF

#Create Kind cluster
kind create cluster --name knative --config clusterconfig.yaml

#Apply knative manifests
kubectl apply --filename https://github.com/knative/serving/releases/download/v0.21.0/serving-crds.yaml
kubectl apply --filename https://github.com/knative/serving/releases/download/v0.21.0/serving-core.yaml

curl -Lo kourier.yaml https://github.com/knative-sandbox/net-kourier/releases/download/v0.21.0/kourier.yaml
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

#Install kourier
kubectl apply -f kourier.yaml
kubectl apply -f kourier-patch.yaml

#Patch network
kubectl patch configmap/config-network \
  --namespace knative-serving \
  --type merge \
  --patch '{"data":{"ingress.class":"kourier.ingress.networking.knative.dev"}}'

#Patch DNS
kubectl patch configmap/config-domain \
  --namespace knative-serving \
  --type merge \
  --patch '{"data":{"127.0.0.1.nip.io":""}}'

#Start minio container
docker run -d -p 9000:9000 minio/minio server /data

#Load ICOW service container into Kind cluster
kind load docker-image --name knative dev.local/icow_service:0.1

#Write Knative manifest for ICOW
cat > inference_cattle.yaml <<EOF
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: inference-service
  namespace: default
spec:
  template:
    spec:
      containers:
      - image: dev.local/icow_service:0.1
        imagePullPolicy: IfNotPresent
        command: ["icow-service"]
        args: ["10", "8080"]
        ports:
        - containerPort: 8080
          name: h2c
        env:
        - name: AWS_ACCESS_KEY
          value: minioadmin
        - name: AWS_SECRET_KEY
          value: minioadmin
        - name: S3ENDPOINT_URL
          value: http://localhost:9000
EOF

#Install ICOW Knative service
kubectl apply -f inference_cattle.yaml

echo "Setup complete 🎉"
#Keep container alive indefinitely
tail -f /dev/null
