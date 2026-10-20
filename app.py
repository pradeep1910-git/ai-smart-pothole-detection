from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
import os
import uuid
import glob
import subprocess

app = Flask(__name__)

# Load trained pothole model
model = YOLO("model/best.pt")

# Folders
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():

    # Check uploaded file
    if "video" not in request.files:
        return "No video uploaded"

    video = request.files["video"]

    if video.filename == "":
        return "No video selected"

    # Save uploaded video
    input_filename = str(uuid.uuid4()) + ".mp4"
    input_path = os.path.join(UPLOAD_FOLDER, input_filename)
    video.save(input_path)

    print("===================================")
    print("Video uploaded successfully")
    print("Starting YOLO pothole detection...")
    print("===================================")

    # Run YOLO
    results = model.predict(
        source=input_path,
        conf=0.25,
        save=True
    )

    # Count potholes
    total_potholes = 0
    detections = []

    for result in results:

        if result.boxes is not None:

            total_potholes += len(result.boxes)

            for box in result.boxes:
                confidence = round(float(box.conf[0]) * 100, 2)

                detections.append({
                    "confidence": confidence
                })

    print("Total potholes:", total_potholes)

    # Find newest YOLO output
    possible_files = []

    possible_files += glob.glob(
        os.path.join("runs", "detect", "**", "*.avi"),
        recursive=True
    )

    possible_files += glob.glob(
        os.path.join("runs", "detect", "**", "*.mp4"),
        recursive=True
    )

    if not possible_files:
        return "YOLO finished, but output video was not found."

    result_video_path = max(
        possible_files,
        key=os.path.getmtime
    )

    print("YOLO output:")
    print(result_video_path)

    # Output filename
    output_filename = "detected_" + str(uuid.uuid4()) + ".mp4"
    output_path = os.path.join(
        RESULT_FOLDER,
        output_filename
    )

    print("Converting video to MP4...")

    command = [
        "ffmpeg",
        "-y",
        "-i",
        result_video_path,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        output_path
    ]

    try:
        subprocess.run(
            command,
            check=True
        )

    except subprocess.CalledProcessError:
        return "FFmpeg conversion failed."

    print("===================================")
    print("Detection completed successfully")
    print(output_path)
    print("===================================")

    return render_template(
        "result.html",
        video=output_filename,
        count=total_potholes,
        detections=detections
    )


@app.route("/results/<filename>")
def results(filename):
    return send_from_directory(
        RESULT_FOLDER,
        filename
    )


if __name__ == "__main__":
    app.run(debug=True)