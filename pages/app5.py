import gradio as gr
import os

def create_app5_page():
    """
    Gradio page for App 5: AI Grocery Scanner Video Demonstration.
    """
    # Resolve the video path relative to the current script directory
    video_path = os.path.join(os.path.dirname(__file__), "ai_scanner_demo.mp4")

    # Check if the video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"The video file '{video_path}' was not found.")

    with gr.Blocks() as app5_page:
        gr.Markdown("# AI Grocery Scanner")
        gr.Markdown(
            """
            This system uses a fine-tuned YOLO detector with Deep SORT to track grocery items.
            The system monitors motion from left to right and dynamically adds items to the basket accordingly.
            Below is a video demonstration of the AI Grocery Scanner in action.
            """
        )

        # Embed the video demonstration
        gr.Video(label="AI Grocery Scanner Video Demo", value=video_path)

    return app5_page
