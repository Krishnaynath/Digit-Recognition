# Assignment 1 – Digit Recognition using Machine Learning

## Overview
This project is part of **Assignment 1** for the Machine Learning course.  
The goal is to create a simple **digit recognition system** using the **MNIST dataset** and **TensorFlow/Keras**.

The model learns to recognize handwritten digits (0–9) from 28x28 grayscale images and achieves high accuracy on unseen test data.

---

## Dataset
- **Name**: MNIST Handwritten Digits
- **Source**: Built into TensorFlow (`tensorflow.keras.datasets`)
- **Details**:
  - 60,000 training images
  - 10,000 test images
  - Each image is 28x28 pixels in grayscale
  - Labels: Digits from 0 to 9

---

## Steps Performed

### 1. **Importing Libraries**
We use:
- **TensorFlow/Keras** → Build and train the neural network
- **Matplotlib** → Visualize predictions
- **NumPy** → Numerical operations

### 2. **Loading the Dataset**
Using `mnist.load_data()` to load train and test sets.

### 3. **Preprocessing**
- Normalize pixel values to range [0, 1] for faster convergence.
- Convert labels to **one-hot encoded** vectors.

### 4. **Model Building**
Created a simple **Sequential** neural network:
1. **Flatten Layer** → Converts 28x28 images into a 1D array.
2. **Dense Layer (128 neurons, ReLU)** → Learns patterns in the data.
3. **Dense Layer (10 neurons, Softmax)** → Outputs probabilities for each digit.

### 5. **Model Compilation**
- Optimizer: `adam`
- Loss: `categorical_crossentropy`
- Metric: `accuracy`

### 6. **Training**
- Epochs: 5
- Batch size: default (32)
- Validation: Test set used for validation accuracy

### 7. **Evaluation**
Achieved:
- **Test Accuracy**: ~97.7%
- **Test Loss**: ~0.08

### 8. **Predictions & Visualization**
Plotted the first 10 test images with **True** and **Predicted** labels.

### 9. **Saving the Model**
Saved as `.keras` format for future use:
```python
model.save("digit_recognition_model.keras")
