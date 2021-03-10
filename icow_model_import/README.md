# ICOW Model Import


## How to use to serialize a Pytorch model for use with ICOW

1. Install `icow_model_import`

2. Set S3 configuration parameters in the environment including the following.

    ```
    AWS_ACCESS_KEY=<Access key for your s3>
    AWS_SECRET_KEY=<Secret key for your s3>
    S3ENDPOINT_URL=<Endpoint for your s3 instance>
    ```

3. Import `pytorch_to_bucket` from `icow_model_import` wherever your model can be loaded or is defined and call it with the desired parameters. An example of this can be seen below.

    ```python
    import torch

    from icow_model_import import pytorch_to_bucket


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
    model_name = "simple_model"
    pytorch_to_bucket(
        model,
        1,
        256,
        "models",
        model_name,
        1,
        dynamic_shape=False,
        dummy_forward_input=dummy_forward_input,
    )
    ```

    Please see our code docs for a more in-depth look but here is a quick look at the parameter list for `pytorch_to_bucket`.

    ```
    Parameters
    ----------
    net : torch.nn.Module
        Network to serialize.
    model_input_height : int
        Model input dimension height.
    model_input_width : int
        Model input dimension width.
    model_bucket : str
        S3 bucket to use for storing the serialized models.
    model_object_name : str
        Name for the uploaded model object in s3.
    model_version : int
        Version for the uploaded model object.
    dynamic_shape : Optional[bool]
        Whether or not the spatial dimensions of the input are dynamic. Defaults to
        False, meaning the shape is static.
    output_names : Optional[List[str]]
        List of output headnames for mutli-head classification networks. Defaults to
        None which will be assigned to ["output"].
    dummy_forward_input : Optional[torch.Tensor]
        Dummy forward pass input that will be run through the model for tracing export
        purposes. Defaults to None meaning a dummy input will be generated.
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