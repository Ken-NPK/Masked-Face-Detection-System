# Import required packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import config
import matplotlib.pyplot as plt
import numpy as np
import os
import random


# Load the image from dataset directory
def load_image(base_dataset_path):

    # Initialize empty list for data and label
    data_list = []
    label_list = []

    print("Loading the dataset...")

    # Load dataset with label through looping
    for label in os.listdir(base_dataset_path):

        # Obtain the image directory path
        image_directory_path = os.path.join(base_dataset_path, label)

        # Random sampling the image based on the data size initialize
        image_path_list = random.sample(os.listdir(image_directory_path), config.DATA_SIZE)

        # Loop through the list that contain image path
        for index, image in enumerate(image_path_list):

            # Obtain the image path
            image_path = os.path.join(image_directory_path, image)

            # Load the input image (224 x 224) and preprocess it.
            image = load_img(image_path, target_size=(224, 224))
            image = img_to_array(image)

            # Load the data and labels to the lists respectively.
            data_list.append(image)
            label_list.append(label)

            print("Image {} - ({}) completed pre-process.".format(index + 1, label))

    return data_list, label_list


# Pre-processing the images
def pre_processing_data(data_list, label_list):

    # Convert the data to Numpy arrays and normalize the input pixel intensities from the range [0, 255] to [0, 1]
    data_np_array = np.array(data_list, dtype="float32") / 255.0

    # Convert the class labels to Numpy arrays.
    label_np_array = np.array(label_list)

    # Perform one hot encoding on the labels
    lb = LabelBinarizer()
    bin_label = lb.fit_transform(label_np_array)
    label_np_array = to_categorical(bin_label)

    return data_np_array, label_np_array, lb


# Building the deep neural network
def build_model():

    # Load the MobileNetV2 network, ensuring the head FC layer sets are left off
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    # Construct the head of the model that will be placed on top of the base model
    head_model = base_model.output
    head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(128, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(2, activation="softmax")(head_model)

    # Place the head FC model on top of the base model (this will become the actual model we will train)
    model = Model(inputs=base_model.input, outputs=head_model)

    # Load over all layers in the base model and freeze them so they will not be updated during the first
    # training process
    for layer in base_model.layers:
        layer.trainable = False

    print("Compiling model")

    # Compile our model
    opt = Adam(lr=config.INIT_LR)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Print the structure of the neural network model
    print(model.summary())

    return model


# Train the deep neural network model that is built
def train_model(model, image_aug, train_x, train_y, test_x, test_y):

    print("Training the head of the network.")

    # Train the head of the network
    h = model.fit(
        image_aug.flow(train_x, train_y),
        validation_data=(test_x, test_y),
        batch_size=config.BATCH_SIZE,
        epochs=config.NUM_EPOCHS,
        verbose=1
    )

    print("Saving face mask detector model...")

    # Save the model
    model.save(os.path.join(config.BASE_MODEL_PATH, "detect_mask.h5"))

    return model, h


# Evaluate the classification model with test data
def eval_model(model, labeling, test_x, test_y):

    # Make predictions on the testing set
    print("Evaluating neural network")
    pred_idxs = model.predict(test_x, batch_size=config.BATCH_SIZE)

    # Find the index of the label with corresponding largest predicted probability
    pred_idxs = np.argmax(pred_idxs, axis=1)

    # Obtain a nicely formatted classification report
    report = classification_report(test_y.argmax(axis=1), pred_idxs, target_names=labeling.classes_)

    return report


# Plot and save a training accuracy graph
def plot_train_acc(h):

    # File path for saving purpose
    plot_path = os.path.sep.join([config.BASE_PLOT_PATH, "train_acc_plot.png"])
    n = config.NUM_EPOCHS

    # Plotting
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, n), h.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, n), h.history["val_accuracy"], label="val_acc")
    plt.title("Training Accuracy")
    plt.xlabel("Number of Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower left")

    # Saving plot
    plt.savefig(plot_path)


# Plot and save a training loss graph
def plot_train_loss(h):

    # File path for saving purpose
    plot_path = os.path.sep.join([config.BASE_PLOT_PATH, "train_loss_plot.png"])
    n = config.NUM_EPOCHS

    # Plotting
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, n), h.history["loss"], label="train_loss")
    plt.plot(np.arange(0, n), h.history["val_loss"], label="val_loss")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Number of Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")

    # Saving plot
    plt.savefig(plot_path)


# Start of the train.py
if __name__ == "__main__":

    # Load the image from the dataset
    data, label = load_image(config.BASE_DATASET_PATH)

    # Image pre-processing
    data_numpy_array, label_numpy_array, lb = pre_processing_data(data, label)

    # Build the model
    network_model = build_model()

    # Partition the data into training and testing splits using 80% of the data for training and
    # the remaining 20% for testing
    (trainX, testX, trainY, testY) = train_test_split(data_numpy_array, label_numpy_array,
                                                      test_size=0.20, stratify=label_numpy_array, random_state=42)

    # Construct the training image generator for data augmentation
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")

    # Train the model
    network_model, H = train_model(network_model, aug, trainX, trainY, testX, testY)

    # Obtain the formatted classification report
    class_report = eval_model(network_model, lb, testX, testY)

    print("Classification report")
    print(class_report)

    # Plot graph relate to metrics
    plot_train_acc(H)
    plot_train_acc(H)
