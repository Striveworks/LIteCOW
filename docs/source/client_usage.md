# Client Usage

Once you have an ICOW sandbox or ICOW deployed to your kubernetes cluster you can quickly start using it to run inference.

Get the address of service with kubectl

```bash
kubectl get ksvc -n icow
```

```{hint}
When calling the service with grpc don't include the `http` scheme.
```

Get the python client

```bash
pip install litecow
```

Run an inference

```python
import grpc

from litecow.client import ICOWClient

client = ICOWClient.create_with_channel_options("icow-service.icow.127.0.0.1.nip.io:80")
result = client.get_inference("s3://models/tinyyolov2", np.float32(np.random.random(1, 3, 20, 20)), model_version="v1")

print(result)
```
