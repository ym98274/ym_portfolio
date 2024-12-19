import gradio as gr
from vehicle_assistant.gesture_detection import GestureDetector


# Global variable to manage the running state of the detection system
gesture_detector = GestureDetector()
is_running = False  # Tracks whether the gesture detection system is running


def start_detection():
    """
    Stream the gesture detection system's output for volume and menu states dynamically.
    """
    global is_running
    is_running = True  # Set the running state to True
    try:
        for state in gesture_detector.detect_gestures():
            if not is_running:  # Check if the stop button was pressed
                gesture_detector.stop()  # Stop the gesture detection safely
                break
            yield state["Volume"], state["Menu"]
    except RuntimeError as e:
        yield 0, "Error: Unable to access webcam"
    finally:
        gesture_detector.stop()  # Ensure the camera is released


def stop_detection():
    """
    Stop the gesture detection system.
    """
    global is_running
    is_running = False  # Set the running state to False
    gesture_detector.stop()  # Ensure webcam resources are released


def create_app2_page():
    """
    Gradio app for gesture-based volume and menu control.
    """
    with gr.Blocks() as app2_page:
        gr.Markdown("# Gesture-Based Dashboard Control")
        gr.Markdown(
            """
            This system detects hand gestures to control volume and navigate menus dynamically.
            """
        )

        # Interface for volume and menu control
        volume_display = gr.Textbox(label="Current Volume", interactive=False)
        menu_display = gr.Textbox(label="Current Menu", interactive=False)
        start_button = gr.Button("Start Gesture Detection")
        stop_button = gr.Button("Stop Gesture Detection")

        # Start button triggers gesture detection
        start_button.click(
            fn=start_detection,
            inputs=[],
            outputs=[volume_display, menu_display],
        )

        # Stop button stops gesture detection
        stop_button.click(
            fn=stop_detection,
            inputs=[],
            outputs=[],
        )

        # Add usage instructions below the main demo
        gr.Markdown("## Usage Instructions")
        gr.Markdown(
            """
            **How to use this system:**

            - **Volume Control:**
                - Moving your right index finger **clockwise** increases the volume.
                - Moving your left index finger **counterclockwise** decreases the volume.

            - **Menu Navigation:**
                - **Swipe left** to navigate to the next menu.
                - **Swipe right** to return to the previous menu.
                - Similar hand gestures can also achieve the same result.

            **Note:**
            - Ensure your webcam is active and positioned to detect your gestures.
            - Perform gestures in a well-lit environment for optimal accuracy.
            """
        )

    return app2_page
