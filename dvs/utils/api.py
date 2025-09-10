import dvs


class Utils:
    def __init__(self, dvs: dvs.DVS):
        """Utils API"""
        self.dvs = dvs

    def embed_text(self, text: str) -> list[float]:
        """Embed a text and return a single vector."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts and return vectors."""
        return (
            self.dvs.model.get_embeddings(texts, model_settings=self.dvs.model_settings)
        ).to_python()
