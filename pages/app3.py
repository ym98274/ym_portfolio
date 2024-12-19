import gradio as gr
from video_assistant.video_processing import extract_frames
from video_assistant.frame_captioning import generate_frame_captions
from video_assistant.vector_database import (
    initialize_in_memory_client,
    create_database,
    add_frames_to_database,
    query_database,
)


def process_videos_and_store(video1, video2):
    """
    Process uploaded videos by extracting frames, generating captions, and storing them in ChromaDB.

    Args:
        video1 (file): Path to the first video file.
        video2 (file): Path to the second video file.

    Returns:
        str: Success message indicating the databases were created.
    """
    try:
        client = initialize_in_memory_client()

        if video1:
            # Process Video 1
            frames_video1 = extract_frames(video1.name, interval=1)
            captions_video1 = generate_frame_captions(frames_video1)
            collection_video1 = create_database(client, "video1_collection")
            add_frames_to_database(
                collection_video1,
                captions=[item.get("caption", [{"generated_text": "Error"}])[0]["generated_text"] for item in captions_video1],
                frame_indexes=[item["frame_index"] for item in captions_video1],
            )

        if video2:
            # Process Video 2
            frames_video2 = extract_frames(video2.name, interval=1)
            captions_video2 = generate_frame_captions(frames_video2)
            collection_video2 = create_database(client, "video2_collection")
            add_frames_to_database(
                collection_video2,
                captions=[item.get("caption", [{"generated_text": "Error"}])[0]["generated_text"] for item in captions_video2],
                frame_indexes=[item["frame_index"] for item in captions_video2],
            )

        return "Videos successfully processed and stored in ChromaDB!"

    except Exception as e:
        return f"Error: {str(e)}"


def query_videos(query_text):
    """
    Query the ChromaDB collections for the relevant video and timestamp/index.

    Args:
        query_text (str): The user-provided query.

    Returns:
        str: Relevant video and frame information.
    """
    try:
        client = initialize_in_memory_client()

        # Query Video 1
        video1_result = query_database(client, "video1_collection", query_text)
        video1_distance = video1_result["distance"] or float("inf")

        # Query Video 2
        video2_result = query_database(client, "video2_collection", query_text)
        video2_distance = video2_result["distance"] or float("inf")

        # Determine the best match
        if video1_distance < video2_distance:
            return f"Relevant Video: Video 1\nFrame Index: {video1_result['id']}\nDistance: {video1_distance}"
        elif video2_distance < video1_distance:
            return f"Relevant Video: Video 2\nFrame Index: {video2_result['id']}\nDistance: {video2_distance}"
        else:
            return "No relevant match found."

    except Exception as e:
        return f"Error querying videos: {str(e)}"


def create_app3_page():
    """
    Gradio app for video upload, query processing, and ChromaDB integration.
    """
    with gr.Blocks() as app3_page:
        gr.Markdown("# Multimodal AI Video Search System")
        gr.Markdown(
            "Upload two videos (max 15 seconds each), and the system will process frames, generate captions, "
            "store them in a vector database, and allow you to query for relevant video segments."
        )

        # File upload components
        video1_input = gr.File(label="Upload Video 1", file_types=[".mp4", ".avi", ".mov"])
        video2_input = gr.File(label="Upload Video 2", file_types=[".mp4", ".avi", ".mov"])

        # Button to process videos
        process_button = gr.Button("Process Videos")
        output_message = gr.Textbox(label="Status", interactive=False)

        # Query input and button
        query_input = gr.Textbox(label="Enter your query")
        query_button = gr.Button("Search")
        query_result = gr.Textbox(label="Query Result", interactive=False)

        # Link the button to the processing function
        process_button.click(
            fn=process_videos_and_store,
            inputs=[video1_input, video2_input],
            outputs=[output_message],
        )

        # Link the query button to the query function
        query_button.click(
            fn=query_videos,
            inputs=[query_input],
            outputs=[query_result],
        )

        # Add usage instructions below the main demo
        gr.Markdown("## Usage Instructions")
        gr.Markdown(
            """
            **How to use this system:**

            1. **Upload Videos**:
                - Click "Upload Video 1" and "Upload Video 2" to upload two videos.
                - Each video should have **distinct scenes** to test the system effectively.
                - Ensure the videos are **no longer than 15 seconds** to reduce processing time.

            2. **Process Videos**:
                - Click the "Process Videos" button to start processing.
                - Once the status displays **"Videos successfully processed and stored in ChromaDB!"**, the videos are ready for querying.

            3. **Search for Relevant Scenes**:
                - Enter a **textual query** describing a scene or object in one of the videos.
                - For example: *"a person holding a book"* or *"a car in motion"*.
                - Click the "Search" button to retrieve the **relevant video** and its corresponding **timestamp or frame index**.

            4. **Review Results**:
                - The system will display:
                    - The relevant video (Video 1 or Video 2).
                    - The specific frame index.
                    - The distance score indicating relevance.

            **Note:**
            - Use videos with clear and distinct scenes to achieve better results.
            - Ensure the videos are in supported formats: `.mp4`, `.avi`, `.mov`.
            """
        )

    return app3_page


# Entry point for testing
if __name__ == "__main__":
    app3 = create_app3_page()
    app3.launch()
