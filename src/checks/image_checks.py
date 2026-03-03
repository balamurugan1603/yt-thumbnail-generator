from exceptions.pipeline_exceptions import ClutterCheckError
from settings.config import RetryConfig
from utils.image_utils import calculate_zone_edge_density

def check_zone_clutter(bg, text_design, cfg: RetryConfig = RetryConfig()):
    clutterness = calculate_zone_edge_density(bg, text_design)
    if clutterness["edge_density"] > cfg.clutter_threshold:
        raise ClutterCheckError(clutterness["edge_density"], cfg.clutter_threshold)
    return clutterness