class ThumbnailPipelineError(Exception):
    """Base for all pipeline errors."""

class ClutterCheckError(ThumbnailPipelineError):
    def __init__(self, edge_density: float, threshold: float):
        self.edge_density = edge_density
        self.threshold = threshold
        super().__init__(
            f"Zone too cluttered: edge_density={edge_density:.4f} > threshold={threshold:.4f}"
        )

class PromptTooLongError(ThumbnailPipelineError):
    def __init__(self, word_count: int, limit: int = 15):
        self.word_count = word_count
        self.limit = limit
        super().__init__(f"Prompt too long: {word_count} words exceeds {limit}-word limit")

class NonEnglishPromptError(ThumbnailPipelineError):
    def __init__(self, prompt: str):
        self.prompt = prompt
        super().__init__(f"Prompt is not in English: {prompt!r}")