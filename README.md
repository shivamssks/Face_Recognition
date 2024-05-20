# Face Identification System

This project implements a face identification system capable of detecting faces in images, recognizing them using a pre-trained model, and classifying them using a Support Vector Machine (SVM).

## Features

- Face Detection using Multi-task Cascaded Convolutional Networks (MTCNN).
- Face Recognition leveraging a pre-trained ResNet-Inception model.
- Classification of recognized faces using an SVM model.

## Prerequisites

- Python 3.x
- OpenCV
- Keras
- scikit-learn
- tqdm
- joblib
- pandas
- numpy
- mtcnn

## Installation

1. Clone the repository:

2. Navigate to the project directory:

3. Install the required Python packages:

## Usage

1. Run the script with the following arguments:
   Replace `<path-to-dataset>`, `<path-to-weight-file>`, and `<path-for-cropped-images>` with the appropriate paths for your setup.

2. The script performs the following steps:
   - Detects faces in the provided dataset.
   - Recognizes the detected faces using the pre-trained model.
   - Saves cropped images of recognized faces.
   - Trains an SVM model on the extracted features.
   - Prints the accuracy of the trained SVM model.

## Contributing

Contributions are welcome. Please feel free to submit a Pull Request with your enhancements or bug fixes.
