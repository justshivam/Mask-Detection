import os
import cv2
import json
import logging
import numpy as np
import mediapipe as mp
import tensorflow as tf


def detect_mask(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_p = cv2.resize(image, (224, 224))
    image_p = image_p.reshape(
        (1, image_p.shape[0], image_p.shape[1], image_p.shape[2]))
    image_p = tf.keras.applications.mobilenet_v2.preprocess_input(image_p)
    pred = model.predict(image_p)
    confidence = max(pred[0])
    tag = np.where(pred[0] == confidence)
    return tags[tag[0][0]], confidence


def run_loop():
    DELAY = 15
    M_SCORE = tags[2], 1.0

    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face.process(imgRGB)

        if results.detections:
            for id, detection in enumerate(results.detections):
                h, w, c = img.shape
                box = detection.location_data.relative_bounding_box
                fbox = int(box.xmin*w), int(box.ymin * h), \
                    int(box.width*w), int(box.height*h)
                crop_img = img[fbox[1]:fbox[1] +
                               fbox[2], fbox[0]:fbox[0]+fbox[3]]
                if DELAY == 20:
                    M_SCORE = detect_mask(img)
                    print(M_SCORE)
                    DELAY = 0
                cv2.rectangle(img, fbox, (0, 255, 0), 2)
                cv2.putText(img, f'{M_SCORE[0]}: {int(M_SCORE[1]*100)}%',
                            (fbox[0], fbox[1]-20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

            DELAY += 1
        cv2.imshow("Mask Detector", img)
        cv2.waitKey(1)


if __name__ == '__main__':

    logging.getLogger('tensorflow').setLevel(logging.FATAL)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    tags = ['Mask_Weared_Incorrectly', 'With_Mask', 'Without_Mask']

    model = tf.keras.models.load_model('MaskDetection.h5')

    cap = cv2.VideoCapture(0)
    mpFace = mp.solutions.face_detection
    face = mpFace.FaceDetection()
    mpDraw = mp.solutions.drawing_utils
    run_loop()
