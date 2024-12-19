import requests
import base64
import io
import time
from PIL import Image

# Hugging Face API Configuration
API_TOKEN = "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"
MODEL_ID = "microsoft/git-base-coco"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}


def encode_image(image):
    """
    Encode a PIL image to a Base64 string.

    Args:
        image (PIL.Image): The image to encode.

    Returns:
        str: Base64-encoded string.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def query_huggingface_api_with_retry(image_base64, max_retries=5, wait_time=30):
    """
    Query the Hugging Face API with a retry mechanism.

    Args:
        image_base64 (str): Base64-encoded image string.
        max_retries (int): Maximum number of retries.
        wait_time (int): Time to wait between retries (in seconds).

    Returns:
        dict: The response from the API.
    """
    retries = 0
    while retries < max_retries:
        response = requests.post(API_URL, headers=HEADERS, json={"inputs": {"image": image_base64}})

        if response.status_code == 503:  # Model is loading
            print(f"Model is loading. Retrying in {wait_time} seconds... (Attempt {retries + 1}/{max_retries})")
            time.sleep(wait_time)
            retries += 1
        elif response.status_code == 200:
            return response.json()  # Successful response
        else:
            raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

    raise Exception("Max retries reached. Model did not load in time.")


def generate_frame_captions(frames):
    """
    Generate captions for a list of frames.

    Args:
        frames (list of PIL.Image): List of video frames as PIL images.

    Returns:
        list of dict: Captions for each frame.
    """
    results = []
    for idx, frame in enumerate(frames):
        try:
            print(f"Processing frame {idx}...")
            image_base64 = encode_image(frame)
            response = query_huggingface_api_with_retry(image_base64)
            results.append({"frame_index": idx, "caption": response})
        except Exception as e:
            print(f"Error processing frame {idx}: {e}")
            results.append({"frame_index": idx, "error": str(e)})
    return results
