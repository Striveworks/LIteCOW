# ICOW Model Import


## How to use to serialize a Pytorch model for use with ICOW

1. Install `icow_model_import`

2. Set S3 configuration parameters in the environment including the following.

    ```
    AWS_ACCESS_KEY=<Access key for your s3>
    AWS_SECRET_KEY=<Secret key for your s3>
    S3ENDPOINT_URL=<Endpoint for your s3 instance>
    ```

3. Import `pytorch_to_onnx_file` and `onnx_file_to_s3` from `icow_model_import` wherever your model can be loaded or is defined and call it with the desired parameters. An example of this can be seen below.

    ```python
    from tempfile import NamedTemporaryFile

    import torch

    from icow_model_import import onnx_file_to_s3, pytorch_to_bucket


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


## Testing with Docker Compose

1. Start the docker-compose stack. This will create the nescessary containers and kick off appropriate processes for testing.

    ```bash
    docker-compose up --build
    ```

2. Stop the docker-compose stack after testing finishes. You can use `CTRL + C` or if you started the compose stack detached you can use the following command to stop the containers.

    ```bash
    docker-compose down
    ```

## Development environment with Docker-Compose

For dev purposes, feel free to edit the startup.sh script and add `tail -f /dev/null` to the end of it so that the docker stays alive and exec into the container like so.

    ```bash
    docker exec -it icow_model_import_icow_model_import_1 bash
    ```

Thanks to docker volume mounts, all changes outside and inside of the docker will persist and allow editing and development.