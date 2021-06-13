#########################################################################################
# Description: Opens a web-cam and detects facial expression from pre-defined
# category
#
# The emotion recognition model will return the emotion predicted real time.
# The model is trained on the fer2013 Kaggle dataset.
# Emotion classification test accuracy: 66%
#########################################################################################

# Import Modules
import imutils
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import time

# Constants
DETECTION_MODEL = 'haarcascade_frontalface_default.xml'
EMOTION_MODEL = '_mini_XCEPTION.102-0.66.hdf5'
EMOTION_CATEGORY = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]


def main():
    face_detection = cv2.CascadeClassifier(DETECTION_MODEL)

    # loading models
    emotion_classifier = load_model(EMOTION_MODEL, compile=False)
    cv2.namedWindow('Web Camera')

    camera = cv2.VideoCapture(0)
    while True:
        frame = camera.read()[1]
        # reading the frame
        frame = imutils.resize(frame, width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)

        canvas = np.zeros((250, 300, 3), dtype="uint8")
        frame_clone = frame.copy()
        if len(faces) > 0:
            faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            # X,Y Co-ordinate of face detected rectangle
            (fX, fY, fW, fH) = faces
            # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
            # the ROI (Region of Interest) for classification via the CNN
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            m_prediction = emotion_classifier.predict(roi)[0]

            # Find maximum of each prediction
            label = EMOTION_CATEGORY[m_prediction.argmax()]

            # Put it back to live
            cv2.putText(frame_clone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frame_clone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

        cv2.imshow('Web Camera', frame_clone)

        # Take Snap Shot if User presses 'y'
        if cv2.waitKey(1) & 0xFF == ord('z'):
            imageName = str(time.strftime("%Y_%m_%d_%H_%M_%S")) + '.jpg'
            cv2.imwrite(imageName, frame_clone)

        # Run live until User presses 'x' key
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    camera.release()
    cv2.destroyAllWindows()

#Call to main function - Entry Point
main()
