import gradio as gr
from pipeline.thumbnail_generator import generate_thumbnail_pipeline

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AI YouTube Thumbnail Generator") as app:
        gr.Markdown("## AI YouTube Thumbnail Generator")
        gr.Markdown("Generate high-CTR 1280×720 thumbnails with smart text placement.")

        prompt_input = gr.Textbox(
            label="Thumbnail Title (≤ 15 words)",
            placeholder="Only 1% Can Solve This Riddle",
        )
        generate_btn  = gr.Button("Generate Thumbnail", variant="primary")
        output_image  = gr.Image(type="pil", label="Generated Thumbnail")

        generate_btn.click(
            fn=generate_thumbnail_pipeline,
            inputs=prompt_input,
            outputs=output_image,
        )

    return app