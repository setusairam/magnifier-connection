from flask import Flask, Response, render_template_string, request, session
import cv2
import threading
import queue
from flask_session import Session
import uuid

app = Flask(__name__)

# Configure session storage
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Shared queue to hold frames for streaming
frame_queue = queue.Queue(maxsize=2)  # Small queue to minimize latency
stop_event = threading.Event()

# Store user-specific settings in a global dictionary
user_settings = {}

def apply_filter_and_zoom(frame, settings):
    """Apply zoom and color filters to the frame."""
    zoom_level = settings.get("zoom", 1.0)
    color_filter = settings.get("filter", "none")
    
    # Apply zoom
    h, w, _ = frame.shape
    center_x, center_y = w // 2, h // 2
    crop_w, crop_h = int(w // zoom_level), int(h // zoom_level)

    left = max(center_x - crop_w // 2, 0)
    right = min(center_x + crop_w // 2, w)
    top = max(center_y - crop_h // 2, 0)
    bottom = min(center_y + crop_h // 2, h)

    frame = frame[top:bottom, left:right]
    frame = cv2.resize(frame, (w, h))

    # Apply color filter
    if color_filter == "grayscale":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif color_filter == "invert":
        frame = cv2.bitwise_not(frame)
    elif color_filter == "red":
        frame[:, :, 1] = 0  # Remove green
        frame[:, :, 0] = 0  # Remove blue
    elif color_filter == "blue":
        frame[:, :, 1] = 0  # Remove green
        frame[:, :, 2] = 0  # Remove red
    elif color_filter == "green":
        frame[:, :, 0] = 0  # Remove blue
        frame[:, :, 2] = 0  # Remove red

    return frame

def capture_frames():
    """Capture frames continuously in a separate thread."""
    cap = cv2.VideoCapture(0)

    # Set optimized resolution and frame rate
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    while not stop_event.is_set():
        success, frame = cap.read()
        if not success:
            print("Error: Failed to capture frame.")
            continue

        # Add the frame to the queue (overwrite old frames if the queue is full)
        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            try:
                frame_queue.get_nowait()  # Remove the oldest frame to make space
                frame_queue.put(frame)
            except queue.Empty:
                pass

    cap.release()

@app.route('/stream')
def stream():
    """Stream optimized video frames."""
    def generate_frames(user_id):
        """Fetch frames from the queue and encode them."""
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=1)  # Wait for a frame
            except queue.Empty:
                print("Warning: No frames in the queue.")
                continue

            # Get settings for the current user
            settings = user_settings.get(user_id, {"zoom": 1.0, "filter": "none"})

            # Apply user-specific filter and zoom
            processed_frame = apply_filter_and_zoom(frame, settings)

            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n\r\n')
    
    user_id = session.get("user_id", "default")
    return Response(generate_frames(user_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.before_request
def assign_user_id():
    """Assign a unique user ID to each session."""
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())  # Generate a unique user ID

@app.route('/')
def home():
    """Full-Screen Video Streaming Page with Controls."""
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Accessibility Webcam Stream</title>
        <style>
            body {
                margin: 0;
                overflow: hidden;
                display: flex;
                flex-direction: column;
                align-items: center;
                background: black;
                color: white;
                font-family: Arial, sans-serif;
            }
            video, img {
                width: 80%;
                height: auto;
                margin-bottom: 10px;
            }
            .controls {
                display: flex;
                justify-content: center;
                gap: 10px;
                margin-bottom: 20px;
            }
            button {
                padding: 10px 20px;
                font-size: 16px;
                cursor: pointer;
            }
        </style>
    </head>
    <body>
        <div class="controls">
            <button onclick="applyFilter('none')">None</button>
            <button onclick="applyFilter('grayscale')">Grayscale</button>
            <button onclick="applyFilter('invert')">Invert</button>
            <button onclick="applyFilter('red')">Red</button>
            <button onclick="applyFilter('green')">Green</button>
            <button onclick="applyFilter('blue')">Blue</button>
            <button onclick="zoom('in')">Zoom In</button>
            <button onclick="zoom('out')">Zoom Out</button>
        </div>
        <img src="/stream" alt="Webcam Stream">
        <script>
            function applyFilter(filter) {
                fetch(`/filter?type=${filter}`);
            }

            function zoom(direction) {
                fetch(`/zoom?direction=${direction}`);
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/filter', methods=['GET'])
def set_filter():
    """Set the desired color filter for the current user."""
    filter_type = request.args.get('type', 'none')
    user_id = session.get("user_id", "default")
    if user_id:
        user_settings.setdefault(user_id, {})["filter"] = filter_type
    return f"Filter applied: {filter_type}"

@app.route('/zoom', methods=['GET'])
def set_zoom():
    """Adjust the zoom level for the current user."""
    direction = request.args.get('direction', 'in')
    user_id = session.get("user_id", "default")
    if user_id:
        settings = user_settings.setdefault(user_id, {"zoom": 1.0})
        zoom_level = settings.get("zoom", 1.0)
        if direction == "in":
            zoom_level = min(zoom_level + 0.1, 3.0)  # Cap zoom at 3x
        elif direction == "out":
            zoom_level = max(zoom_level - 0.1, 1.0)  # Minimum zoom is 1x
        settings["zoom"] = zoom_level
    return f"Zoom level: {zoom_level:.1f}"

if __name__ == "__main__":
    # Start the frame capture thread
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()

    try:
        # Run the Flask app
        app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
    except KeyboardInterrupt:
        # Stop the frame capture thread on exit
        stop_event.set()
        capture_thread.join()
