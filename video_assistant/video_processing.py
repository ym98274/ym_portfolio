import cv2
from PIL import Image

def extract_frames(video_path, interval=1):
    """
    Extract frames from the video at specified intervals.
    """
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frames = []
    count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break
        if count % (fps * interval) == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
        count += 1

    video.release()
    return frames
