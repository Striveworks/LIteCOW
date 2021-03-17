# Sandbox

This sandbox tutorial is setup to make getting started with ICOW quickly

## Install
```
# curl -s https://nacho.striveworks.us/chariot/icow-light/-/raw/sandbox-setup/sandbox/setup.sh | bash
git clone git@nacho.striveworks.us:chariot/icow-light.git
bash sandbox/setup.sh

```
or
```
git clone git@nacho.striveworks.us:chariot/icow-light.git
make sandbox
```

This will setup a kind cluster with knative and the ICOW service installed as well as a minio server.

From here, you can use the ICOW client to hit ICOW at [http://localhost:31080](http://localhost:31080)
and access Minio at [http://localhost:9000](http://localhost:9000)


```{admonition} Minio Credentials

| Access Key | Secret Key |
|------------|------------|    
|`minioadmin`|`minioadmin`|

```

## Import Model
Import your first model from the [ONNX Model Zoo](https://github.com/onnx/models)

```
litecow import-model --sourcehttps://github.com/onnx/models/blob/master/vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.onnx tinyyolov3
```
