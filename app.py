# This code redue the size of Image and colorize it in order to reduce computaional time.

import numpy as np
import cv2
import os
import argparse

DIR = r"C:\Users\Adi computer\Desktop\AI Project"
PROTOTXT = os.path.join(DIR, r"model/colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, r"model/pts_in_hull.npy")
MODEL = os.path.join(DIR, r"model/colorization_release_v2.caffemodel")

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
                help="path to input black and white image")
args = vars(ap.parse_args())

print("Load Modal")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

# Load centers for ab channel quantization used for rebalancing.
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Load the input image
image = cv2.imread(args["image"])
scaled = image.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50

print("Colorizing the image")
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)

colorized = (255 * colorized).astype("uint8")

cv2.imshow("Original", image)
cv2.imshow("Colorized", colorized)
cv2.waitKey(0)




# This Image Colorize Full Sized Orignal Image ... I did not executed it because I don't have required computational resources

# import numpy as np
# import cv2
# import os
# import argparse

# DIR = r"C:\Users\Adi computer\Desktop\AI Project"
# PROTOTXT = os.path.join(DIR, r"model/colorization_deploy_v2.prototxt")
# POINTS = os.path.join(DIR, r"model/pts_in_hull.npy")
# MODEL = os.path.join(DIR, r"model/colorization_release_v2.caffemodel")

# # Construct the argument parser
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", type=str, required=True,
#                 help="path to input black and white image")
# args = vars(ap.parse_args())

# print("Load Modal")
# net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
# pts = np.load(POINTS)

# # Load centers for ab channel quantization used for rebalancing.
# class8 = net.getLayerId("class8_ab")
# conv8 = net.getLayerId("conv8_313_rh")
# pts = pts.transpose().reshape(2, 313, 1, 1)
# net.getLayer(class8).blobs = [pts.astype("float32")]
# net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# # Load the input image
# image = cv2.imread(args["image"])
# scaled = image.astype("float32") / 255.0
# lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

# L = cv2.split(lab)[0]
# L -= 50

# print("Colorizing the image")
# net.setInput(cv2.dnn.blobFromImage(L))
# ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

# ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

# colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

# colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
# colorized = np.clip(colorized, 0, 1)

# colorized = (255 * colorized).astype("uint8")

# cv2.imshow("Original", image)
# cv2.imshow("Colorized", colorized)
# cv2.waitKey(0)



# This code is for my future plan about this project curently I failed to exeute it with User Interface

# from flask import Flask, request, render_template
# import base64
# from io import BytesIO

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/colorize', methods=['POST'])
# def colorize():
#     image_data = request.files['image'].read()
#     image_np = np.frombuffer(image_data, np.uint8)
#     image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

#     # Your colorization code here
#     import numpy as np
#     import cv2
#     import os
#     import argparse

#     DIR = r"C:\Users\Adi computer\Desktop\AI Project"
#     PROTOTXT = os.path.join(DIR, r"model/colorization_deploy_v2.prototxt")
#     POINTS = os.path.join(DIR, r"model/pts_in_hull.npy")
#     MODEL = os.path.join(DIR, r"model/colorization_release_v2.caffemodel")

#     # Construct the argument parser
#     ap = argparse.ArgumentParser()
#     ap.add_argument("-i", "--image", type=str, required=True,
#                     help="path to input black and white image")
#     args = vars(ap.parse_args())

#     print("Load Modal")
#     net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
#     pts = np.load(POINTS)

#     # Load centers for ab channel quantization used for rebalancing.
#     class8 = net.getLayerId("class8_ab")
#     conv8 = net.getLayerId("conv8_313_rh")
#     pts = pts.transpose().reshape(2, 313, 1, 1)
#     net.getLayer(class8).blobs = [pts.astype("float32")]
#     net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

#     # Load the input image
#     image = cv2.imread(args["image"])
#     scaled = image.astype("float32") / 255.0
#     lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

#     resized = cv2.resize(lab, (224, 224))
#     L = cv2.split(resized)[0]
#     L -= 50

#     print("Colorizing the image")
#     net.setInput(cv2.dnn.blobFromImage(L))
#     ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

#     ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

#     L = cv2.split(lab)[0]
#     colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

#     colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
#     colorized = np.clip(colorized, 0, 1)

#     colorized = (255 * colorized).astype("uint8")

#     cv2.imshow("Original", image)
#     cv2.imshow("Colorized", colorized)
#     cv2.waitKey(0)

#     # Convert the colorized image to base64 for sending to the frontend
#     colorized_image_base64 = base64.b64encode(cv2.imencode('.png', colorized)[1]).decode('utf-8')

#     return {'colorized': colorized_image_base64}

# if __name__ == '__main__':
#     app.run(debug=True)