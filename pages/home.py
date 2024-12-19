import gradio as gr
from PIL import Image
import numpy as np
import os


def create_home_page():
    # Function to display the profile image
    def display_profile_image():
        image_path = "profile.jpg"  # Replace with your image path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {os.path.abspath(image_path)}")

        # Load and return the image
        image = Image.open(image_path)
        return np.array(image)

    with gr.Blocks(css="""
        #main-title { text-align: left; font-size: 2em; font-weight: bold; margin-bottom: 20px; }
        #profile-image { border-radius: 10px; max-width: 80%; display: block; margin: 0 auto; }
        #work-text, #focus-areas, #projects-text, #footer-text { line-height: 1.6; text-align: justify; margin-bottom: 20px; }
        hr { margin: 20px 0; border: none; border-top: 2px solid #ccc; }
    """) as home_page:
        # Title Section with Image Embedded
        with gr.Row():
            with gr.Column(scale=7, min_width=500):
                gr.Markdown(
                    """
                    # Welcome to My Portfolio  
                    I am **Younis Mageit**, a Machine Learning Engineer determined to design, build, and prototype intelligent autonomous systems 
                    that address complex challenges and redefine technological possibilities.
                    """,
                    elem_id="main-title",
                )
            with gr.Column(scale=3, min_width=200):
                gr.Image(
                    value=display_profile_image(),
                    label=None,
                    elem_id="profile-image",
                    show_label=False,
                )

        # Work Section
        with gr.Row():
            gr.Markdown(
                """
                ## My Work  
              I hold a Master's degree in Computer Science from the University of Nottingham 
              where I specialized in applying cutting-edge AI/ML techniques to solve a range of challenges and business problems. 
              My work included developing machine learning models for predictive maintenance in electric vehicles, creating AI-powered NLP systems to streamline booking processes,
              and leveraging AI with LiDAR sensor data to enhance autonomous planning and decision-making capabilities.
              My thesis explored how existing LLMs can be effectively adapted and modified to generate useful control signals for lateral and longitudinal vehicle movement. 
              Besides learning more about self-driving systems, I gained exposure in successfully configuring LLMs to support just about any downstream autonomous task.
                """,
                elem_id="work-text",
            )

        # Key Focus Areas Section
        with gr.Row():
            gr.Markdown(
                """
                ## Key Focus Areas and Interests  
                - Combining **AI technologies** to automate work and routine processes.  
                - Developing **AI solutions** from scratch, including data generation and model deployment.  
                - Fine-tuning and adapting **LLMs** for a range of tasks including search and reccommendation.  
                - Building **Agentic AI systems** capable of reliably performing complex tasks.  
                - Implementing **generative models** to address data shortages and enable new functionality.  
                """,
                elem_id="focus-areas",
            )

        # Projects Section
        with gr.Row():
            gr.Markdown(
                """
                ## Projects  
                I am passionate about prototyping and creating novel **AI solutions** to tackle complex challenges.  
                Through this portfolio, you can:  

                - **Interact** with cutting-edge AI systems to test their functionality firsthand.  
                - **Explore** how these solutions address real-world challenges, from process automation to advanced analytics.  
                - **Discover** unique AI approaches, including multimodal AI, gesture-based controls, and autonomous assistance tools.  

                Users are encouraged to stretch these systems and share any insights for how they can be improved.
                """,
                elem_id="projects-text",
            )

        # Footer Section
        gr.Markdown(
            """
            ---
            **Thank you for visiting my portfolio!**  
            Use the navigation tabs above to explore my projects and learn more about my work.
            """,
            elem_id="footer-text",
        )

    return home_page


# If you want to test the application separately
if __name__ == "__main__":
    home_page = create_home_page()
    home_page.launch()
