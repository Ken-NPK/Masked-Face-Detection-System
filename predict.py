# import required packages
import config
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2 as cv


# Load face detection model
def load_face_detect(base_path_face_detect_model):

    # Obtain the face detection model file path
    model_file = os.path.join(base_path_face_detect_model, "res10_300x300_ssd_iter_140000.caffemodel")

    # Obtain the face detection model configuration file path
    config_file = os.path.join(base_path_face_detect_model, "deploy.prototxt.txt")

    # Load the DNN
    net = cv.dnn.readNetFromCaffe(config_file, model_file)

    return net


# Detect and locate the face
def face_detect(dnn, image_path):

    # Load the input image from disk, clone it, and grab the image spatial dimensions
    image = cv.imread(image_path)

    # Construct a blob from the Image
    blob = cv.dnn.blobFromImage(cv.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))

    # Pass the blob through the network and obtain the face detections
    dnn.setInput(blob)
    detections = dnn.forward()

    return detections


# Perform classification
def face_mask_classification(image_path, detections, classification_model):

    # Loop the image using openCV library
    image = cv.imread(image_path)
    (h, w) = image.shape[:2]

    # To loop through the detections
    for i in range(detections.shape[2]):
        # extract the probability that is related with the detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the probability is greater than the 0.5
        if confidence > 0.5:

            # Getting the coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (xmin, ymin, xmax, ymax) = box.astype("int")

            # Obtain the face image from the whole image
            face = image[ymin:ymax, xmin:xmax]

            # Convert from BGR to RGB (OpenCV read image in BGR format)
            face = cv.cvtColor(face, cv.COLOR_BGR2RGB)

            # Resize the face image to 224 x 224
            face = cv.resize(face, (224, 224))

            # Preprocess the face image
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # Pass the face through the model to determine whether the face is masked or unmasked
            (masked, unmasked) = classification_model.predict(face)[0]

            # Determine the class label and color we'll use to draw the bounding box and text
            if masked > unmasked:
                label = "Masked"

                # Green
                color = (0, 255, 0)

            else:
                label = "Unmasked"

                # Red
                color = (0, 0, 255)

            # Include the probability in the label
            probability = "Probability: {:.2f}%".format(max(masked, unmasked) * 100)

            # Display the label and bounding box rectangle on the output image
            cv.putText(image, label, (xmin, ymin - 40), cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv.putText(image, probability, (xmin, ymin - 20), cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

    return image


# Whole process of the prediction
def predict(base_input_path, base_output_path):

    # Load the face detection
    dnn_face_detect = load_face_detect(config.BASE_FACE_DETECTION_MODEL)

    # Load the classification_model
    face_mask_model = load_model(os.path.join(config.BASE_MODEL_PATH, "detect_mask.h5"))

    # Check whether any files exists in input directory
    if len(os.listdir(base_input_path)) == 0:
        print("No input images found!")
        return

    # Loop through the image in input directory
    for image in os.listdir(base_input_path):

        # Obtain the image_path
        input_image_path = os.path.join(base_input_path, image)

        # Detect the face
        face = face_detect(dnn_face_detect, input_image_path)

        # Perform classification on the image
        output = face_mask_classification(input_image_path, face, face_mask_model)

        # Save the image to the output directory
        output_image_name = "(Predicted) " + str(image)
        output_image_path = os.path.join(base_output_path, output_image_name)
        cv.imwrite(output_image_path, output)

        print("Image {} done masked face detection!".format(image))


# Beginning of the program
if __name__ == "__main__":

    # Perform classification
    predict(config.BASE_INPUT, config.BASE_OUTPUT)
