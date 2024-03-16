```markdown
# Hand Gesture Recognition Project

This project utilizes computer vision techniques to recognize hand gestures using a webcam. It includes
functionalities for using an existing dataset, creating a new dataset, and training a model for gesture classification.

## Requirements

Before running the project, ensure you have the necessary libraries installed:

- **OpenCV**: A library for computer vision tasks. Install it using `pip install opencv-python`.
- **cvzone**: A library that provides additional computer vision functionalities. Install it using `pip install cvzone`.
- **TensorFlow**: An open-source machine learning framework. Install it using `pip install tensorflow`.
- **NumPy**: A library for numerical computations. Install it using `pip install numpy`.

Alternatively, you can install the required libraries using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Using Existing Dataset

To run the project using the provided dataset, follow these steps:

1. **Clone the Repository**: Clone this repository to your local machine.
2. **Navigate to the Directory**: Open a terminal or command prompt and navigate to the project directory.
3. **Run the Main File**: Execute the `main.py` file to start the application.

```bash
python main.py
```

The `main.py` file initializes the webcam and performs real-time hand gesture recognition using the pre-trained model.

### 2. Creating a New Dataset

To create your own dataset for training the model, follow these steps:

1. **Run the Dataset Creation Script**: Execute the `create_dataset.py` script.
2. **Capture Gesture Images**: Place your hand within the webcam frame and perform various gestures.
3. **Save Images**: Press the 's' key to save each gesture image to the dataset folder.

```bash
python create_dataset.py
```

The dataset creation script captures images of hand gestures and saves them with appropriate labels for training the model.

3. Training the Model

To train the model using the created dataset, follow these steps:

1. **Update Dataset Path**: Update the dataset path in the `train_model.py` script to point to your created dataset.
2. **Run the Training Script**: Execute the `train_model.py` script to train the model using the specified dataset.

```bash
python train_model.py
```

The training script loads the dataset, defines the model architecture, and trains the model using TensorFlow.

**Additional Information**

- **Model Configuration**: You can adjust the model architecture and hyperparameters in the `train_model.py` script to optimize performance.
- **Model Persistence**: The trained model will be saved as `My_model.h5` in the specified directory for later use.
- **Real-Time Recognition**: Modify the `main.py` file to integrate real-time gesture recognition with the trained model.

**Acknowledgements**

**YouTube:** We thank YouTube for hosting valuable tutorials and educational content related to computer 
vision and machine learning, which greatly contributed to the development of this project.

**GitHub:** We acknowledge the open-source community on GitHub for providing access to libraries, frameworks, 
and resources that were instrumental in building this project.
