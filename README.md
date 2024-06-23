# Prodigy_05

# FOOD RECOGNITION MODEL

## Introduction

This project aims to train a deep learning model to classify images of Indian food dishes. The dataset used consists of 16 different classes of Indian food dishes, with a total of 1,000 images. The trained model can predict the class of a new image of an Indian food dish.

## Project Steps

### 1. Data Preprocessing

**Objective:** Prepare the dataset for training the model.

- **Image Resizing:** All images were resized to a standard size of 224x224 pixels.
- **Normalization:** Pixel values were normalized to be between 0 and 1.
- **Data Splitting:** The dataset was split into:
  - **Training Set:** 70% of the data
  - **Validation Set:** 15% of the data
  - **Test Set:** 15% of the data

### 2. Model Training

**Objective:** Train a MobileNetV2 model for image classification.

- **Model Used:** MobileNetV2, a pre-trained model known for its performance in image classification tasks.
- **Training Duration:** 5 epochs
- **Batch Size:** 32
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Performance Metric:** Accuracy

### 3. Model Evaluation

**Objective:** Evaluate the model's performance.

- The model achieved an accuracy of 85% on the validation set after training.

### 4. Predictions

**Objective:** Use the trained model to predict the class of new images.

- The model successfully predicted the class of a new image of an Indian food dish.

## Conclusion

The project demonstrates that a deep learning model can classify images of Indian food dishes with high accuracy. This model could be integrated into a mobile app to help users identify various Indian food dishes.

## Detailed Explanation of Steps

### Data Preprocessing
- **Function Used:** `image_processing`
  - **Resizing:** 224x224 pixels
  - **Normalization:** Pixel values between 0 and 1
- **Data Split:**
  - Training Set: 70%
  - Validation Set: 15%
  - Test Set: 15%

### Model Training
- **Model:** MobileNetV2
- **Epochs:** 5
- **Batch Size:** 32
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Metric:** Accuracy

### Model Evaluation
- **Validation Set Accuracy:** 85%

### Predictions
- **New Image Classification:** The model correctly identified the class of a new image of an Indian food dish.

## Technologies Used

- **Programming Language:** Python
- **Deep Learning Framework:** TensorFlow with Keras
- **Model:** MobileNetV2
- **Libraries:**
  - NumPy
  - Pandas
  - Matplotlib
  - TensorFlow
  - Keras
  - Scikit-learn
- **Tools:**
  - Jupyter Notebook

## Future Work

To further improve the model:
- Train on a larger dataset.
- Develop a mobile app using the model to help users identify Indian food dishes.
