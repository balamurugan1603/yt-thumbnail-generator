from ui.gradio_app import build_ui
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),                        # console
        logging.FileHandler("logs/app.log"),            # file
    ],
)

if __name__ == "__main__":
    app = build_ui()
    app.launch()