# XRayNet: Pneumonia Detection Using Deep Learning

This repository contains a Jupyter Notebook implementing a deep learning pipeline for classifying chest X-ray images to detect pneumonia. It uses TensorFlow and Keras for model development and evaluation.

## Features

- **Data Preprocessing**: Loads and preprocesses chest X-ray images, converting them to grayscale and resizing them.
- **Data Augmentation**: Supports augmentation for better model generalization.
- **Deep Learning Model**: Builds a convolutional neural network (CNN) for binary classification (Pneumonia vs. Normal).
- **Performance Metrics**: Evaluates the model using classification reports and confusion matrices.

## Dataset

The notebook uses the [Chest X-ray Pneumonia dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia). Ensure you download and extract the dataset into the correct directory (`/kaggle/input/chest-xray-pneumonia/chest_xray`).

## Requirements

Install the following Python libraries to run the notebook:

- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV
- Scikit-learn
- tqdm

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/XRayNet.git
   cd XRayNet
   ```
2. Ensure the dataset is in the correct directory.
3. Open the notebook in Jupyter or any compatible environment:
   ```bash
   jupyter notebook xraynet-test-87.ipynb
   ```
4. Run the notebook cells sequentially to train and evaluate the model.

## Notebook Overview

### 1. **Imports and Setup**

- Necessary libraries for data loading, preprocessing, model creation, and evaluation.

### 2. **Data Preparation**

- Loads X-ray images from the dataset, preprocesses them by resizing and normalizing pixel values, and splits them into training and testing datasets, Note is Imbalance Data.

### 3. **Model Architecture**

- the main problem in the data was imbalanced, i try to solve is with many operation :

  - Data Augmentation
  - Class Weighting ( Most Accurate )
  - Oversampling and Undersampling
  - Transfer Learning with ResNet50

- Defines a CNN using TensorFlow/Keras with layers like Conv2D, MaxPooling, and BatchNormalization.

### 4. **Training**

- Trains the model using the training dataset with a specified batch size and learning rate.

### 5. **Evaluation**

- Generates predictions and evaluates the model using a confusion matrix and classification report.

## Results

Include a summary of model performance (accuracy, precision, recall, etc.) based on the output of the evaluation.

## Contributions

Feel free to submit pull requests or suggest new features. Feedback is always welcome!
