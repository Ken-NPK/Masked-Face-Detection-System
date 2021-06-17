# Path for input, output file from the program

# Define the base path for the input dataset
BASE_DATASET_PATH = "dataset"

# Define the base path for the input directory
BASE_INPUT = "input"

# Define the base path for the output directory
BASE_OUTPUT = "output"

# Define the base path for the output model
BASE_MODEL_PATH = "model"

# Define the base path for the output plot
BASE_PLOT_PATH = "plot"

# Define the base path for the face detection model
BASE_FACE_DETECTION_MODEL = "OpenCV DNN"

# Parameter for the classification model

# Initialize the number of images needed for each class for the model
# Noted: The value should not be more than the number of images available in each class.
# With_mask (5883), Without_mask (5909)
DATA_SIZE = 2000

# Initialize the initial learning rate for the neural network
INIT_LR = 1e-4

# Initialize the number of epoch to train the model
NUM_EPOCHS = 30

# Initialize the batch size
BATCH_SIZE = 32
