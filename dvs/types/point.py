# dvs/types/point.py
import logging
import typing

import openai_embeddings_model as oai_emb_model
import pydantic

import dvs.utils.chunk
import dvs.utils.ids

if typing.TYPE_CHECKING:
    from dvs.types.document import Document

logger = logging.getLogger(__name__)


class Point(pydantic.BaseModel):
    """
    Represents a point in the vector space, associated with a document.

    This class encapsulates the essential information about a point in the vector space,
    including its unique identifier, the document it belongs to, a content hash, and its
    vector embedding.

    Attributes:
        point_id (Text): A unique identifier for the point in the vector space.
        document_id (Text): The identifier of the document this point is associated with.
        content_md5 (Text): An MD5 hash of the content, used for quick comparisons and integrity checks.
        embedding (List[float]): The vector embedding representation of the point in the vector space.

    The Point class is crucial for vector similarity search operations, as it contains
    the embedding that is used for comparison with query vectors.
    """  # noqa: E501

    model_config = pydantic.ConfigDict(validate_assignment=True)

    # Fields
    point_id: typing.Text = pydantic.Field(
        default_factory=lambda: dvs.utils.ids.get_id("pt"),
        description="Unique identifier for the point in the vector space.",
    )
    document_id: typing.Text = pydantic.Field(
        ...,
        description="Identifier of the associated document.",
    )
    content_md5: typing.Text = pydantic.Field(
        ...,
        description="MD5 hash of the content for quick comparison and integrity checks.",  # noqa: E501
    )
    embedding: typing.Text | None = pydantic.Field(
        default=None,
        description=(
            "Vector embedding representation of the point. "
            + "Python list of floats (float64) or "
            + "base64 string of numpy array in float32."
        ),
    )

    metadata: typing.Dict[typing.Text, typing.Any] = pydantic.Field(
        default_factory=dict,
        description="Additional metadata associated with the point.",
    )

    @pydantic.field_validator("embedding", mode="before")
    @classmethod
    def embedding_as_base64_string(cls, v: typing.Any) -> typing.Text:
        """
        Convert the embedding to a base64 string.
        """
        from dvs.utils.to import vector_to_base64

        if isinstance(v, typing.Sequence):
            if all(isinstance(x, (int, float)) for x in v):
                return vector_to_base64(list(v))

        if isinstance(v, str):
            return v

        logger.warning(f"Unknown embedding type: {type(v)}, let pydantic handle it")
        return v  # type: ignore

    @classmethod
    def set_embeddings_from_contents(
        cls,
        points: typing.Sequence["Point"],
        contents: typing.Sequence[typing.Text] | typing.Sequence["Document"],
        *,
        model: oai_emb_model.OpenAIEmbeddingsModel,
        model_settings: oai_emb_model.ModelSettings,
    ) -> typing.List["Point"]:
        """
        Set the embeddings for the points from the contents.
        """
        if len(points) != len(contents):
            raise ValueError("Points and contents must be the same length")

        output: typing.List["Point"] = []

        _contents: typing.List[typing.Text] = [
            c if isinstance(c, str) else c.content for c in contents
        ]
        _embeddings_resp = model.get_embeddings(
            input=_contents, model_settings=model_settings
        )
        for point, embedding in zip(points, _embeddings_resp.output):
            point.embedding = embedding
            output.append(point)

        logger.debug(f"Created {len(output)} embeddings")
        return output

    @property
    def is_embedded(self) -> bool:
        return self.embedding is not None

    def to_python(self) -> typing.List[float]:
        import base64

        import numpy as np

        if self.embedding is None:
            raise ValueError("Embedding is not set")

        return np.frombuffer(
            base64.b64decode(self.embedding), dtype=np.float32
        ).tolist()
