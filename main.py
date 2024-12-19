import gradio as gr
import os
from pages.home import create_home_page
from pages.app1 import create_app1_page
from pages.app2 import create_app2_page
from pages.app3 import create_app3_page
from pages.app4 import create_app4_page
from pages.app5 import create_app5_page
from pages.contact import create_contact_page

def create_demo():
    with gr.Blocks() as demo:
        with gr.Tab("Home"):
            create_home_page()
        with gr.Tab("Gaze-AI Search"):
            create_app1_page()
        with gr.Tab("AI Motion control"):
            create_app2_page()
        with gr.Tab("Smart video search"):
            create_app3_page()
        with gr.Tab("TimeTracker AI"):
            create_app4_page()
        with gr.Tab("GroceryVisionApp"):
            create_app5_page()
        with gr.Tab("Contact"):
            create_contact_page()

    return demo

# Gradio app callable for Gunicorn
app = create_demo()

if __name__ == "__main__":
    # For local testing
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 8000)))
