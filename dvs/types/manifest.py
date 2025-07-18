import typing

import pydantic

import dvs


class Manifest(pydantic.BaseModel):
    name: typing.Text = pydantic.Field(
        default="default",
        description="The name of the database.",
    )
    description: typing.Text = pydantic.Field(
        default="",
        description="The description of the database.",
    )
    version: typing.Text = pydantic.Field(
        default=dvs.__version__,
        description="The version of the manifest.",
    )
    manifest_table_name: typing.Text = pydantic.Field(
        default="manifest",
        description="The name of the table in DuckDB that stores the manifest data.",
    )
    points_table_name: typing.Text = pydantic.Field(
        default="points",
        description="The name of the table in DuckDB that stores the vector embeddings and related point data.",  # noqa: E501
    )
    documents_table_name: typing.Text = pydantic.Field(
        default="documents",
        description="The name of the table in DuckDB that stores the document metadata.",  # noqa: E501
    )
    embedding_model: typing.Text = pydantic.Field(
        ...,
        description="The name of the OpenAI embedding model to use for generating vector embeddings.",  # noqa: E501
    )
    embedding_dimensions: int = pydantic.Field(
        ...,
        description="The number of dimensions in the vector embeddings generated by the chosen model.",  # noqa: E501
    )
