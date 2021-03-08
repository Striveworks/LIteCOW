# ICOW Service Install

To install ICOW you need to have a k8s or similar cluster setup to deploy knative and ultimately icow to. If you don't please follow [this guide for bootstrapping a kind cluster](https://kind.sigs.k8s.io/docs/user/quick-start/#installation). For a regular k8s cluster you should find somewhat parallel instructions at the [following knative guide](https://knative.dev/docs/install/any-kubernetes-cluster/).

1. Create a cluster config based on [Step 1 of the knative tutorial](https://knative.dev/blog/1/01/01/how-to-set-up-a-local-knative-environment-with-kind-and-without-dns-headaches/#step-1-setting-up-your-kubernetes-deployment-using-kind) with the filename of `clusterconfig.yaml`. The only change should be from `kind.sigs.k8s.io/v1alpha3` to `4` as seen below.

    ```
    kind: Cluster
    apiVersion: kind.sigs.k8s.io/v1alpha4
    nodes:
    - role: control-plane
    extraPortMappings:
        ## expose port 31380 of the node to port 80 on the host
    - containerPort: 31080
        hostPort: 80
        ## expose port 31443 of the node to port 443 on the host
    - containerPort: 31443
        hostPort: 443
    ```

2. Create the knative cluster with your cluster's command. With kind this should look like something like the following.

    Creating a new cluster:
    ```bash
    kind create cluster --name knative --config clusterconfig.yaml
    ```

    Applying to an existing cluster:
    ```bash
    kubectl apply -f clusterconfig.yaml
    ```

3. Apply the knative service components.

    ```
    kubectl apply --filename https://github.com/knative/serving/releases/download/v0.21.0/serving-crds.yaml
  
    kubectl apply --filename https://github.com/knative/serving/releases/download/v0.21.0/serving-core.yaml
    ```

4. Follow step 3 of the [knative kind tutorial](https://knative.dev/blog/1/01/01/how-to-set-up-a-local-knative-environment-with-kind-and-without-dns-headaches/#step-3-set-up-networking-using-kourier) for downloading, adjusting, and installing Kourier.

    ```
    curl -Lo kourier.yaml https://github.com/knative-sandbox/net-kourier/releases/download/v0.21.0/kourier.yaml
    ```

    You may have to follow https://knative.dev/docs/install/any-kubernetes-cluster/ if you have a regular k8s deployment rather than kind.

5. At this point the cluster and knative is configured. Now we have to deploy icow. If you are using a kind cluster we will need to build the icow service image and push it into the cluster, or use an icow image in a repo.

    ```
    kind load dev.local/icow_service:0.1 --name knative
    ```
    if you're using a production repository you can use it's name here instead. However, if you're using a repository that does not provide the nescessary metadata, you can add it to knative's skip tag resolving by executing the following command and adding your repo to the list of other repo tags that are skipped as seen [here in their documentation](https://knative.dev/docs/serving/tag-resolution/#skipping-tag-resolution).

    `kubectl edit configmap/config-deployment -n knative-serving`
    ```
    registriesSkippingTagResolving: "other_names,your.repo.url"
    ```

6. Deploy the icow service. First create a icow service  deployment yaml for the cluster with the following format making sure that the defined s3 config leads to a valid minio or s3 instance.

    ```
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
                      value: <AWS ACCESS KEY>
                    - name: AWS_SECRET_KEY
                      value: <AWS SEC KEY>
                    - name: S3ENDPOINT_URL
                      value:  <S3 ENDPOINT>
    ```

    run kubectl apply to the inference cattle yaml you defined.

    ```
    kubectl apply -f inference_cattle.yaml
    ```

    you should now be able see the service when running `kubectl get ksvc` which will also give you the URL to the service.

    When using this for icow use the url retrieved from the above command but ignore the `http://` portion of the URL and add the port (typically port 80) that was defined in the kourier setup. To run against the service you can now start an icow_client docker and start making requests using the above service URL!
