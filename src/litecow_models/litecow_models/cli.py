import click

from litecow_models.model import ModelLoader, initialize_s3


@click.group()
@click.pass_context
def cli(ctx):
    """
    litecow is a CLI to interact with lIteCOW server-side component.
    Easily import and export models
    """
    ctx.ensure_object(dict)


@cli.command()
@click.option("--source", help="Model source URL")
@click.option("--model-bucket", default="models", help="Model registry S3 bucket")
@click.option("--version", default="1", help="Version of model")
@click.argument("model")
@click.pass_context
def import_model(ctx, source: str, model_bucket: str, model: str, version: str):
    ModelLoader.import_model(source, model_bucket, model, version)
    click.echo("Model imported 🚀")


@cli.command()
@click.option("--model-bucket", default="models", help="Model registry S3 bucket")
@click.argument("model")
@click.pass_context
def export_model(ctx, model_bucket: str, model: str):
    ModelLoader.export_model(model_bucket, model)
    click.echo("Model exported 🚀")


@cli.command()
@click.pass_context
@click.option("--model-bucket", default="models", help="Model registry S3 bucket")
def enable_versioning(ctx, model_bucket: str):
    initialize_s3(model_bucket)


if __name__ == "__main__":
    cli()
