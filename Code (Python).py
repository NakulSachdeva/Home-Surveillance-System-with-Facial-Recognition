import cv2  # OpenCV library for computer vision tasks
import time  # Library for time-related functions
from datetime import datetime  # Library for date and time manipulation
import argparse  # Library for parsing command-line arguments
import os  # Library for interacting with the operating system

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Open the video capture
video = cv2.VideoCapture(0)

while True:
    # Read a frame from the video
    check, frame = video.read()

    if frame is not None:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)

        # Draw rectangles around the detected faces and save the images
        for x, y, w, h in faces:
            img = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            exact_time = datetime.now().strftime('%Y-%b-%d-%H-%S-%f')
            cv2.imwrite("face detected" + str(exact_time) + ".jpg", img)

        # Display the frame with rectangles
        cv2.imshow("home surv", frame)

        # Wait for a key press
        key = cv2.waitKey(1)

        if key == ord('q'):
            # Parse command-line arguments
            ap = argparse.ArgumentParser()
            ap.add_argument("-ext", "--extension", required=False, default='jpg')
            ap.add_argument("-o", "--output", required=False, default='output.mp4')
            args = vars(ap.parse_args())

            # Get the current directory path, extension, and output file name
            dir_path = '.'
            ext = args['extension']
            output = args['output']

            images = []

            # Get all files in the directory with the specified extension
            for f in os.listdir(dir_path):
                if f.endswith(ext):
                    images.append(f)

            # Get the path of the first image
            image_path = os.path.join(dir_path, images[0])
            frame = cv2.imread(image_path)
            height, width, channels = frame.shape

            # Create a video writer object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output, fourcc, 5.0, (width, height))

            # Write each image to the video
            for image in images:
                image_path = os.path.join(dir_path, image)
                frame = cv2.imread(image_path)
                out.write(frame)

            # Release the video capture and video writer
            break

# Release the video capture and close all windows
video.release()
cv2.destroyAllWindows()
