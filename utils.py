import base64
import numpy as np
from PIL import ImageOps, Image
import streamlit as st

def classify(image, model, class_names):

    # convert image to (224, 224)
    image = ImageOps.fit(image, (224,224), Image.Resampling.LANCZOS)
    image = ImageOps.grayscale(image)

    # convert image to numpy array
    image = np.asarray(image)

    # normalize the data
    image = image.reshape((1,224,224,1))
    image = image.astype(np.float32)/225

    # make prediction
    confidant_score = model.predict(image)
    index = confidant_score >= 0.5
    class_name = class_names[int(index)]

    return confidant_score[0][0], class_name