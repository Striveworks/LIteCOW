# Sandbox

This sandbox tutorial is setup to make getting started with ICOW quickly

## Install
```
curl -s https://raw.githubusercontent.com/Striveworks/LIteCOW/main/sandbox/setup.sh | bash

```
or
```
git clone git@github.com:Striveworks/LIteCOW.git
make sandbox
```

This will setup a kind cluster with knative and the ICOW service installed as well as a minio server.

From here, you can use the ICOW client to hit ICOW at [icow-service.icow.127.0.0.1.nip.io:80](icow-service.icow.127.0.0.1.nip.io:80)
and access Minio at [http://localhost:9000](http://localhost:9000)


```{admonition} Minio Credentials

| Access Key | Secret Key |
|------------|------------|    
|`minioadmin`|`minioadmin`|

```

## Import Model
Import your first model from the [ONNX Model Zoo](https://github.com/onnx/models)

```
litecow import-model --source https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.onnx tinyyolov3
```

## Object Detection
Run the sandbox python file to use the tinyyolov3 model that you just imported
```
python sandbox/sandbox.py
```
