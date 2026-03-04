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

class LowTextImageContrastRatioError(ThumbnailPipelineError):
    def __init__(self, bg_color: tuple, primary_text_color: tuple, secondary_text_color: tuple, primary_contrast_ratio: float, secondary_contrast_ratio: float, role: str, threshold: float):
        self.bg_color = bg_color
        self.primary_text_color = primary_text_color
        self.secondary_text_color = secondary_text_color
        self.primary_contrast_ratio = primary_contrast_ratio
        self.secondary_contrast_ratio = secondary_contrast_ratio
        self.role = role        # "primary" or "secondary"
        self.threshold = threshold
        super().__init__(
            f"{role} text contrast too low: ratio={getattr(self, f'{role}_contrast_ratio'):.2f} < threshold={threshold:.2f} "
            f"(bg={bg_color}, text={getattr(self, f'{role}_text_color')})"
        )

class ArtifactDetectedError(ThumbnailPipelineError):
    def __init__(self, pos_score: float, neg_score: float):
        self.pos_score = pos_score
        self.neg_score = neg_score
        super().__init__(
            f"Background contains unwanted content (faces/hands/text/symbols): "
            f"clean={pos_score:.4f} content={neg_score:.4f}"
        )

class PromptImageSimilarityError(ThumbnailPipelineError):
    def __init__(self, score: float, threshold: float):
        self.score = score
        self.threshold = threshold
        super().__init__(
            f"CLIP score too low: score={score:.4f} < threshold={threshold:.4f}"
        )