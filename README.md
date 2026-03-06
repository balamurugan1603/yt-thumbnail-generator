# yt-thumbnail-generator

## Architecture:

![YT Thumbnail generator - HLD](<YT Thumbnail Generator - HLD.png>)

## Project Structure:

- src
  - batch.py
  - main.py
  - checks/
    - image_checks.py
    - prompt_checks.py
  - core/
    - models.py
    - prompt_builder.py
  - exceptions/
    - pipeline_exceptions.py
  - logs/
    - app.log
    - batch.log
  - pipeline/
    - batch_pipeline.py
    - solo_pipeline.py
  - services/
    - clip_service.py
    - diffusion_service.py
    - llm_service.py
  - settings/
    - config.py
  - ui/
    - gradio_app.py
  - utils/
    - image_utils.py
    - lang_utils.py

## Inference

For real time inference UI,

- Create virtual environment and install dependencies from `requirements.txt`
- Run `python src/main.py`

For batch inference,

- Create virtual environment and install dependencies from `requirements.txt`
- Update the `PROMPTS` list in `batch.py` script with the input prompts
- Update the `ARTIFACTS_DIR` and `REPORTS_DIR` in `batch.py` with the expected output directories path
- Run `python src/batch.py`
- Outputs and metrics report will be available in the given paths
