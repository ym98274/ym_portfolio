import gradio as gr

def create_contact_page():
    with gr.Blocks() as contact_page:
        gr.Markdown("# Contact Me")
        gr.Markdown("Feel free to reach out to me at **y.mageit@gmail.com**.")
    return contact_page
