import tensorflow as tf
import logging
import os
import json
import sys
import numpy as np


def predict_img(model, img_path):
    image = tf.keras.preprocessing.image.load_img(
        img_path, target_size=(224, 224))
    image_p = tf.keras.preprocessing.image.img_to_array(image)
    image_p = image_p.reshape(
        (1, image_p.shape[0], image_p.shape[1], image_p.shape[2]))
    image_p = tf.keras.applications.mobilenet_v2.preprocess_input(image_p)
    pred = model.predict(image_p)
    confidence = max(pred[0])
    tag = np.where(pred[0] == confidence)
    print(tag, confidence)


if __name__ == '__main__':
    paths = sys.argv[1:]
    if len(paths) == 0:
        print('Please pass the paths as program argument.')
        sys.exit()
    logging.getLogger('tensorflow').setLevel(logging.FATAL)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    model = tf.keras.models.load_model('MaskDetection.h5')
    for path in paths:
        print(path)
        predict_img(model, path)
