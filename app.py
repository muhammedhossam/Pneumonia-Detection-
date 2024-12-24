import streamlit as st
import tensorflow as tf
from PIL import Image
from utils import classify

# Set Title
st.title('Pneumonia Classification')
# upload the file
st.header('please upload an of a chest X-Ray')
file = st.file_uploader('', type=['jpg', 'png', 'jpeg'])
# load the classfier
model = tf.keras.models.load_model('./my_model.h5')
# load the class names
class_names = {0:'Pneumonia', 1:'Normal'}
# display the image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_container_width =True)
    # classify the image
    score, classification = classify(image, model, class_names)
    # write the classification
    st.write("### class is: {}".format(classification))
    st.write("### score: {}".format(score))
