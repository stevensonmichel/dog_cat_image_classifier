# Dog-Cat-Image-Classifier  

## Overview  
This project is a deep learning-based image classifier designed to distinguish between dogs and cats with high accuracy. Using **Convolutional Neural Networks (CNNs)**, the model is trained on a large dataset of labeled images and optimized for precise classification.  

## Dataset  
The dataset used for training and validation is sourced from the **FreeCodeCamp Dataset**, which contains a diverse collection of dog and cat images. It is preprocessed to ensure optimal performance, including resizing, normalization, and augmentation techniques.  

## Technologies Used  
- **Python**  
- **TensorFlow/Keras** (for deep learning and CNN implementation)  
- **OpenCV** (for image processing)  
- **Matplotlib & Seaborn** (for visualization)  
- **NumPy & Pandas** (for data handling)  

## Model Architecture  
The CNN model is designed with multiple convolutional layers, batch normalization, pooling layers, and fully connected layers to effectively extract and learn image features. The architecture includes:  
- **Convolutional Layers** with ReLU activation  
- **Max-Pooling Layers** to reduce dimensionality  
- **Dropout Layers** to prevent overfitting  
- **Dense Layers** with a Softmax output for binary classification  

## Training & Evaluation  
- The dataset is split into **training** and **validation** sets.  
- The model is trained using the **Adam optimizer** and **categorical cross-entropy** loss function.  
- Performance is evaluated using **accuracy, precision, recall, and F1-score**.  
- The trained model achieves high accuracy on test images.  

## Installation & Usage  
### Prerequisites  
Make sure you have the following installed:  
- Python 3.x  
- TensorFlow/Keras  
- OpenCV  
- NumPy, Pandas, Matplotlib  

### Installation  
Clone this repository:  
```bash
git clone https://github.com/yourusername/Dog-Cat-Image-Classifier.git
cd Dog-Cat-Image-Classifier
