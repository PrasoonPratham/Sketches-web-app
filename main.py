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
    # final_img = cv2.add(final_img,np.array([-10.0]))
    return final_img

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

try:
    # opencv_image = image_resize(opencv_image, width=768)

    col1, col2 = st.beta_columns(2)

    col1.image(opencv_image, use_column_width=True, channels="BGR", caption="Orignal")

    col2.image(adjust_gamma(sketch(opencv_image), gamma= 0.1), use_column_width=True, caption="Sketch")
except:
    pass