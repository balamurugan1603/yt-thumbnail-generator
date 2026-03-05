# yt-thumbnail-generator

## Architecture:

![YT Thumbnail generator - HLD](<YT Thumbnail Generator - HLD.png>)

## Project Structure:

| src
     | batch.py
     | main.py
     | checks/
         | image_checks.py
         | prompt_checks.py
     | core/
         | models.py
         | prompt_builder.py
     | exceptions/
         | pipeline_exceptions.py
     | logs/
         | app.log
         | batch.log
     | pipeline/
         | batch_pipeline.py
         | solo_pipeline.py
     | reports/
     | services/
         | clip_service.py
         | diffusion_service.py
         | llm_service.py
     | settings/
         | config.py
     | ui/
         | gradio_app.py
     | utils/
         | image_utils.py
         | lang_utils.py