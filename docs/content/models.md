# ICOW Models


## Preparing an S3 Instance

ICOW requires an S3 Instance with versioning enabled. We have a cli command that comes installed with the `litecow_models` package .For development purposes, use the [sandbox](/sandox)

To use the CLI please see the below steps.

1. Install `litecow_models`.

2. Set S3 configuration parameters in the environment including the following.

    ```
    AWS_ACCESS_KEY=<Access key for your s3>
    AWS_SECRET_KEY=<Secret key for your s3>
    S3ENDPOINT_URL=<Endpoint for your s3 instance>
    ```

2. Use the `litecow enable-versioning` cli command to initialize a new bucket in the configured S3 instance, passing in the name for an existing bucket that will have versioning enabled, or for a new bucket to be created.

    ```bash
    litecow enable-versioning --model-bucket models
    ```


## How to serialize a Pytorch model for use with ICOW

This guide assumes you already have an S3 bucket setup (AWS, Minio, etc.) with versioning enabled. For developmental purposes you can launch the [sandbox](/sandbox)

1. Install `litecow_models`

2. Set S3 configuration parameters in the environment including the following.

    ```
    S3_ACCESS_KEY=<Access key for your s3>
    S3_SECRET_KEY=<Secret key for your s3>
    S3_URL=<Endpoint for your s3 instance>
    ```

3. Import `pytorch_to_onnx_file` and `onnx_file_to_s3` from `litecow_models` wherever your model can be loaded or is defined and call it with the desired parameters. An example of this can be seen below.

    ```python
    from tempfile import NamedTemporaryFile

    import torch

    from litecow_models.model_import import onnx_file_to_s3, pytorch_to_onnx_file


    class SimpleMLP(torch.nn.Module):
    """Simple neural network for testing purposes.

    Parameters
    ----------
    layer_sizes : list
        A list of the layer sizes inclusive of the input
        and output layers
    classification : bool
        If True, softmax is applied to the output. Otherwise,
        L2 normalization is performed.
    """

    def __init__(
        self, layer_sizes=[256, 128, 8], classification=True,
    ):
        super().__init__()

        self.classification = classification
        modules = []
        for i, s in enumerate(layer_sizes[1:]):
            modules += [torch.nn.Linear(layer_sizes[i], s)]
            if i + 1 != len(layer_sizes) - 1:
                modules += [torch.nn.ReLU()]
        self.net = torch.nn.Sequential(*modules)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Function for einputecuting the forward pass of a torch nn model.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input to the model.

        Returns
        -------
        torch.tensor
            Result of the forward pass of the network.
        """
        input_tensor = input_tensor.reshape(
            input_tensor.shape[0], input_tensor.shape[-1]
        )
        input_tensor = self.net(input_tensor)
        if not self.classification:
            input_tensor = torch.nn.functional.normalize(input_tensor)
        return input_tensor

    model = SimpleMLP()
    dummy_forward_input = torch.randn(1, 1, 256).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    with NamedTemporaryFile() as tempfile:
        pytorch_to_onnx_file(
            model,
            tempfile.name,
            model_input_height,
            model_input_width,
            dynamic_shape=False,
            dummy_forward_input=dummy_forward_input,
        )
        onnx_file_to_s3(
            tempfile.name,
            "models",
            "my model object name",
            "my model version",
        )
    ```


## Using litecow CLI to Upload Serialized ONNX Models

This guide assumes you already have serialized ONNX models that you would like to use with ICOW.

1. Install `litecow_models`.

2. Upload a model
  ```
  litecow import-model --source ./tinyyolov2-7.onnx tinyyolov2
  ```
