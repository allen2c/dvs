import json
import time
import typing

import openai_embeddings_model as oai_emb_model
import pydantic
from str_or_none import str_or_none

import dvs.utils.ids

if typing.TYPE_CHECKING:
    import tiktoken

    from dvs.types.point import Point


class Document(pydantic.BaseModel):
    """
    Represents a document in the system, containing metadata and content information.

    This class encapsulates all the relevant information about a document, including
    its unique identifier, name, content, and various metadata fields. It is designed
    to work in conjunction with the Point class for vector similarity search operations.

    Attributes:
        document_id (Text): A unique identifier for the document.
        name (Text): The name or title of the document.
        content (Text): The full text content of the document.
        content_md5 (Text): An MD5 hash of the content for integrity checks.
        metadata (Optional[Dict[Text, Any]]): Additional metadata associated with the document.
        created_at (Optional[int]): Unix timestamp of when the document was created.
        updated_at (Optional[int]): Unix timestamp of when the document was last updated.

    The Document class is essential for storing and retrieving document information
    in the vector similarity search system. It provides a structured way to manage
    document data and metadata, which can be used in conjunction with vector embeddings
    for advanced search and retrieval operations.
    """  # noqa: E501

    # Fields
    # Core Identity
    document_id: typing.Text = pydantic.Field(
        default_factory=lambda: dvs.utils.ids.get_id("doc"),
        description="Unique identifier for the document.",
    )
    name: typing.Text = pydantic.Field(
        ...,
        description="Name or title of the document.",
    )

    # Content
    content: typing.Text = pydantic.Field(
        ...,
        description="Full text content of the document.",
    )
    content_md5: typing.Text = pydantic.Field(
        ...,
        description="MD5 hash of the content for integrity checks.",
    )

    # Source & Structure
    source_id: typing.Text = pydantic.Field(
        default="",
        description="Source ID of the document.",
    )
    chunk_index: int = pydantic.Field(
        default=0,
        description="Chunk index of the document from source.",
    )
    is_chunk: bool = pydantic.Field(
        default=False,
        description="Whether this document is part of a parent document.",
    )

    # Metrics
    total_tokens: int = pydantic.Field(
        default=0,
        description="Total number of tokens in the document.",
    )

    # Metadata
    metadata: typing.Dict[typing.Text, typing.Any] = pydantic.Field(
        default_factory=dict,
        description="Additional metadata associated with the document.",
    )

    # Timestamps
    created_at: int = pydantic.Field(
        default_factory=lambda: int(time.time()),
        description="Unix timestamp of document creation.",
    )
    updated_at: int = pydantic.Field(
        default_factory=lambda: int(time.time()),
        description="Unix timestamp of last document update.",
    )

    @classmethod
    def hash_content(cls, content: typing.Text) -> typing.Text:
        import dvs.utils.hash

        return dvs.utils.hash.hash_content(content)

    @classmethod
    def from_content(
        cls,
        content: typing.Text,
        *,
        name: typing.Optional[typing.Text] = None,
        metadata: typing.Optional[typing.Dict[typing.Text, typing.Any]] = None,
    ) -> "Document":
        _content = str_or_none(content)
        if _content is None:
            raise ValueError("Content is required")
        sanitized_content = _content
        content_md5 = cls.hash_content(sanitized_content)
        metadata = metadata or {}
        name = name or sanitized_content.strip().split("\n\n")[0][:36]
        doc = cls(
            name=name,
            content=sanitized_content,
            content_md5=content_md5,
            metadata=metadata,
        )
        return doc

    @classmethod
    def from_contents(
        cls,
        documents: typing.Union[
            "Document",
            typing.Iterable["Document"],
            str,
            typing.Iterable[str],
            typing.Iterable[typing.Union["Document", str]],
        ],
    ) -> typing.List["Document"]:
        """
        Create documents from the contents.
        """
        # Validate documents
        if isinstance(documents, str) or isinstance(documents, Document):
            documents = [documents]

        docs: list["Document"] = []
        for idx, doc in enumerate(documents):
            if isinstance(doc, str):
                doc = str_or_none(doc)
                if not doc:
                    raise ValueError(f"Document [{idx}] content cannot be empty: {doc}")
                doc = Document.model_validate(
                    {
                        "name": doc.split("\n")[0][:28],
                        "content": doc,
                        "content_md5": Document.hash_content(doc),
                        "metadata": {
                            "content_length": len(doc),
                        },
                        "created_at": int(time.time()),
                        "updated_at": int(time.time()),
                    }
                )

            docs.append(doc.sanitize())

        return docs

    def to_point_with_content(
        self,
        *,
        model: oai_emb_model.OpenAIEmbeddingsModel | None = None,
        model_settings: oai_emb_model.ModelSettings | None = None,
        with_embeddings: bool | None = None,
        metadata: typing.Optional[typing.Dict[typing.Text, typing.Any]] = None,
        verbose: bool | None = None,
    ) -> typing.Tuple["Point", typing.Text]:
        """
        Create the point from the document.
        """

        from dvs.types.point import Point

        if with_embeddings is True and (model is None or model_settings is None):
            raise ValueError(
                "Model and model_settings are required when with_embeddings is True"
            )

        _content = self.content
        _meta = json.loads(json.dumps(metadata or {}, default=str))
        _pt_data = {
            "point_id": dvs.utils.ids.get_id("pt"),
            "document_id": self.document_id,
            "content_md5": Document.hash_content(_content),
            "metadata": _meta,
        }
        _pt = Point.model_validate(_pt_data)

        if with_embeddings:
            if model is None or model_settings is None:
                raise ValueError(
                    "Model and model_settings are required when "
                    + "with_embeddings is True"
                )
            emb_resp = model.get_embeddings(
                input=_content, model_settings=model_settings
            )
            _pt.embedding = emb_resp.output[0]

        return (_pt, _content)

    def to_chunked_documents(
        self,
        *,
        lines_per_chunk: int = 20,
        tokens_per_chunk: int = 500,
        encoding: typing.Optional["tiktoken.Encoding"] = None,
    ) -> typing.List["Document"]:
        import chunkle

        children_docs: list["Document"] = []

        for chunk_idx, chunk in enumerate(
            chunkle.chunk(
                self.content,
                lines_per_chunk=lines_per_chunk,
                tokens_per_chunk=tokens_per_chunk,
            )
        ):
            child_doc = self.model_copy(
                update={"content": chunk, "chunk_index": chunk_idx}, deep=True
            )
            child_doc.sanitize(refresh=True, encoding=encoding)
            children_docs.append(child_doc)

        if len(children_docs) != 1:
            for child_doc in children_docs:
                child_doc.is_chunk = True

        return children_docs

    def sanitize(
        self,
        refresh: bool = False,
        *,
        encoding: typing.Optional["tiktoken.Encoding"] = None,
    ) -> typing.Self:
        """
        Sanitize the document.
        """
        # Validate content
        sanitized_content = str_or_none(self.content)
        if sanitized_content is None:
            raise ValueError("Content is required")
        else:
            self.content = sanitized_content

        # Hash content
        self.content_md5 = self.hash_content(self.content)

        # Validate name
        sanitized_name = str_or_none(self.name)
        if sanitized_name is None:
            self.name = sanitized_content.strip().split("\n\n")[0][:36]
        else:
            self.name = sanitized_name

        if refresh:
            self.updated_at = int(time.time())

        if encoding is not None:
            self.total_tokens = len(encoding.encode(self.content))

        return self

    @pydantic.model_validator(mode="after")
    def validate_string_fields(self) -> typing.Self:
        """
        Validate the string fields of the document.
        """
        return self.sanitize()
