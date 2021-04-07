import streamlit as st
import cv2
import numpy as np

st.title("Sketches with Python")

st.write(
    """This is a web-app that will create a pencil sketch of the image that you provide, start by uploading an image 👇
    """
)

uploaded_file = st.file_uploader("Upload Files", type=["jpeg", "jpg", "png", "jiff"])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)


def dodge(x, y):
    return cv2.divide(x, 255 - y, scale=256)


def sketch(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_invert = cv2.bitwise_not(img_gray)
    img_smoothing = cv2.GaussianBlur(img_invert, (21, 21), sigmaX=0, sigmaY=0)
    final_img = dodge(img_gray, img_smoothing)
    return final_img


try:
    col1, col2 = st.beta_columns(2)

    col1.image(opencv_image, use_column_width=True, channels="BGR", caption="Orignal")

    col2.image(sketch(opencv_image), use_column_width=True, caption="Sketch")
except:
    pass
