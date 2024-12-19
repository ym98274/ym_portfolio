import gradio as gr
import threading
import time
import cv2
from pose_assistant.pose_check import process_frame_for_prediction
from pose_assistant.audio_transcription import record_and_transcribe
from pose_assistant.ai_agent import web_agent_openai


class RealTimePredictor:
    def __init__(self):
        self.prediction = "Waiting for input..."
        self.transcriptions = []
        self.current_transcription = ""
        self.search_result = ""
        self.running = False
        self.recording = False  # Tracks whether audio recording is active
        self.processing_query = False  # Tracks whether the agent is retrieving results

    def start_predictions(self):
        """Start the webcam and update predictions."""
        self.running = True
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            self.prediction = "Error: Could not access the webcam."
            self.running = False
            return

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    self.prediction = "Error: Unable to read frame."
                    break

                # Update the gaze prediction
                current_prediction = process_frame_for_prediction(frame)

                # Check if the user is still looking at the screen
                if current_prediction == "Looking at screen":
                    self.prediction = current_prediction
                    if not self.recording and not self.processing_query:
                        self.start_transcription()
                else:
                    self.prediction = current_prediction
                    if self.recording:
                        self.stop_transcription()  # Stop transcription if the user looks away
                    self.transcriptions.clear()  # Clear transcriptions when the user looks away
                    self.current_transcription = ""
                    self.reset_query()  # Clear search results

                time.sleep(0.1)  # Small delay to avoid overloading
        finally:
            cap.release()

    def stop_predictions(self):
        """Stop the webcam prediction loop."""
        self.running = False
        self.stop_transcription()  # Ensure transcription stops when predictions stop
        self.reset_query()  # Clear the current search result

    def start_transcription(self):
        """Start transcription if the user is looking at the screen."""
        self.recording = True
        threading.Thread(target=self._record_audio, daemon=True).start()

    def stop_transcription(self):
        """Stop transcription."""
        self.recording = False

    def _record_audio(self):
        """Record audio in a separate thread."""
        for transcription in record_and_transcribe():
            if not self.recording:
                break  # Stop the loop if transcription is stopped
            self.current_transcription = transcription

            # Display transcription in the UI
            self.transcriptions.append(transcription)

            # Check for valid query to pass to the agent
            if self._is_valid_query(transcription) and not self.processing_query:
                self.process_query(transcription)

    def _is_valid_query(self, query):
        """Validate the query before passing it to the AI agent."""
        # Ensure the transcription is not "No audio detected"
        if query.lower() == "no audio detected":
            return False
        # Only pass queries with more than 3 characters and containing letters
        return len(query.strip()) > 3 and any(char.isalpha() for char in query)

    def process_query(self, query):
        """Pass the query to the AI agent and retrieve results."""
        self.processing_query = True
        self.search_result = "Retrieving results..."
        threading.Thread(target=self._fetch_results, args=(query,), daemon=True).start()

    def _fetch_results(self, query):
        """Fetch search results from the AI agent."""
        try:
            response = web_agent_openai.run(query)  # Use the web agent to get the response
            self.search_result = response.content  # Extract the content of the response
        except Exception as e:
            self.search_result = f"Error retrieving results: {e}"
        finally:
            self.processing_query = False

    def reset_query(self):
        """Reset the current search result."""
        self.search_result = ""

    def get_data(self):
        """Return the current gaze prediction, transcription, and search results."""
        return self.prediction, self.current_transcription, self.search_result


# Create an instance of the predictor
predictor = RealTimePredictor()


def start_system():
    """Start the webcam and prediction system."""
    threading.Thread(target=predictor.start_predictions, daemon=True).start()
    return "System started!"


def stop_system():
    """Stop the webcam and reset the system."""
    predictor.stop_predictions()
    return "System stopped!"


def fetch_data():
    """Fetch the current gaze prediction, transcription, and search results."""
    return predictor.get_data()


def create_app1_page():
    """
    Gradio interface for gaze detection, audio transcription, and websearch integration.
    """
    with gr.Blocks() as app1_page:
        gr.Markdown("# Context-aware AI Assistant")
        gr.Markdown(
            "The system detects if you're looking at the screen, transcribes your speech, and uses an AI agent to retrieve search results.")

        # Textboxes for predictions, transcription, and search results
        prediction_display = gr.Textbox(label="Gaze Prediction", interactive=False)
        transcription_display = gr.Textbox(label="Current Transcription", interactive=False)
        search_results_display = gr.Textbox(label="Search Results", interactive=False)

        # Buttons to start and stop the system
        start_button = gr.Button("Start System")
        stop_button = gr.Button("Stop System")

        # Connect buttons to start/stop the system
        start_button.click(fn=start_system, outputs=prediction_display)
        stop_button.click(fn=stop_system, outputs=prediction_display)

        # Automatically update the displays
        app1_page.load(
            fn=fetch_data,
            inputs=[],
            outputs=[prediction_display, transcription_display, search_results_display],
            every=1.0,
        )

        # Add usage instructions below the demo
        gr.Markdown("## Usage Instructions")
        gr.Markdown(
            """
            **How to use this system:**
            1. **Start the system**: Click the "Start System" button to activate the webcam and the AI assistant.
            2. **Gaze Detection**: Ensure you're looking at the screen. The system will display "Looking at screen" when it detects your gaze.
            3. **Speech Transcription**: Speak clearly while looking at the screen. The system will transcribe your speech in real-time.
            4. **Web Search**: The transcription is sent to an AI agent for web search. Results are displayed in the "Search Results" box.
            5. **Stop the system**: Click the "Stop System" button to deactivate the webcam and reset the system.

            **Note:**
            - Ensure your webcam and microphone are enabled.
            - Speak loudly and clearly for accurate transcription.
            """
        )

    return app1_page
