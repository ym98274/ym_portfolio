import gradio as gr
import os

def create_app4_page():
    """
    Gradio page for App 4: Time Recording System Video Demonstration.
    """
    # Resolve the video path relative to the current working directory
    video_path = os.path.join(os.path.dirname(__file__), "time_recorder_demo.mp4")

    # Check if the video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"The video file '{video_path}' was not found.")

    with gr.Blocks() as app4_page:
        gr.Markdown("# AI Time Recording App")
        gr.Markdown(
            """
            This system captures screenshots of the user's screen in the background
            and uses an AI classifier to categorize work activities into one of the following classes:
            - Emails
            - Research and Analysis
            - Preparing Documents
            """
        )

        # Embed the video demonstration
        gr.Video(label="Time Recording System Video Demo", value=video_path)

    return app4_page
