import argparse
import cv2
import glob
import joblib
import json
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from keras.models import load_model
from mtcnn import MTCNN
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.svm import SVC
import sys
# sys.path.append("C:\\Users\\shiva\\OneDrive\\Desktop\\Project_resume\\face_recognition\\models")
from models import resnet_inception
class FaceIdentification:
    def __init__(self, img_path, weight_path, output_path):
        """
        Initializes the FaceIdentification class.

        Args:
            img_path (str): Path to the image directory.
            weight_path (str): Path to the weight file.
            output_path (str): Path to the output directory.
        """
        self.img_path = img_path
        self.weight_path = weight_path
        self.output_path = output_path

    def face_detection(self):
        """
        Performs face detection on images in the specified image path.

        Returns:
            list: A list of bounding boxes of the detected faces.
        """
        bboxes = []
        face_detector = MTCNN()

        for img_path in tqdm(glob.glob(os.path.join(self.img_path, "*/*.jpg")),desc="Face detection is in progress"):
            name = os.path.basename(os.path.dirname(img_path))
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_copy = img.copy()
            detections = face_detector.detect_faces(img)

            for detection in detections:
                confidence = detection["confidence"]
                if confidence > 0.90:
                    bbox = detection["box"]
                    x, y, w, h = bbox
                    top_left = (x, y)
                    bottom_right = (x + w, y + h)
                    img_crop = img_copy[y:y + h, x:x + w]
                    img_crop = cv2.resize(img_crop, (160, 160), interpolation=cv2.INTER_AREA)
                    crop_img_dir = os.path.join(self.output_path, "cropped_img", name)
                    os.makedirs(crop_img_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%H%M%S")
                    cv2.imwrite(os.path.join(crop_img_dir, f"{timestamp}.jpg"), img_crop)
                    annotated_img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
                    cv2.imwrite(os.path.join(self.output_path, f"{name}_{timestamp}.jpg"), annotated_img)
                    bboxes.append(bbox)
        return bboxes

    def face_recognition(self):
        """
        Performs face recognition using the FaceNet model and saves the embeddings and labels.
        """
        face_model = resnet_inception.InceptionResNetV1(
        input_shape=(None, None, 3),
        classes=128)
        face_model.load_weights('models\\facenet_keras_weights.h5')

        # face_model = load_model(os.path.join(self.weight_path, "facenet_keras.h5"))
        embeddings = []
        labels = []
        label_map = {}

        for crop_img_path in tqdm(glob.glob(os.path.join(self.output_path, "cropped_img/*/*.jpg")),desc = "Embedding extraction is in progress"):
            label = os.path.basename(os.path.dirname(crop_img_path))
            crop_img = cv2.imread(crop_img_path)
            crop_img = crop_img.reshape(1, 160, 160, 3)
            mean, std = crop_img.mean(), crop_img.std()
            crop_img = (crop_img - mean) / std
            embedding = face_model.predict(crop_img)[0].reshape(1, -1)
            normalized_embedding = normalize(embedding, norm="l2")
            embeddings.append(normalized_embedding)
            labels.append(label)

        labels = np.array(labels)
        encoder = LabelEncoder()
        encoded_labels = encoder.fit_transform(labels)

        for i, label in enumerate(encoder.classes_):
            label_map[i] = label

        mapper = json.dumps(label_map)
        with open(os.path.join("utils", "mapping.json"), "w") as file:
            file.write(mapper)

        all_embeddings = np.reshape(np.array(embeddings), (len(embeddings), 128))
        dataset = np.column_stack((all_embeddings, encoded_labels))
        np.save(os.path.join("utils", "feature_metric_train.npy"), dataset)

    def classification(self):
        """
        Classifies the dataset using Support Vector Machine (SVM) algorithm.

        Returns:
            float: The accuracy of the SVM model on the testing data, multiplied by 100.
        """
        dataset = np.load(os.path.join("utils", "feature_metric_train.npy"))
        df = pd.DataFrame(dataset).sample(frac=1).reset_index(drop=True)
        df.to_csv("utils\\temporary_data.csv", index=False)
        X = df.drop(128, axis=1)
        Y = df[128]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        svm = SVC(kernel="linear", gamma=0.5, C=1.0)
        print("Fitting the SVM model...")
        model = svm.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        joblib.dump(model, os.path.join("models", "svm.pkl"))
        accuracy = accuracy_score(Y_test, y_pred)
        return accuracy * 100

def main(args):
    face_identification = FaceIdentification(args.input_path, args.weight_file, args.output_data)

    print("Performing face detection...")
    bbox = face_identification.face_detection()
    print("Face detection completed.\n")

    print("Performing face recognition...")
    face_identification.face_recognition()
    print("Face recognition completed.\n")

    print("Training the SVM model...")
    accuracy = face_identification.classification()
    print(f"The model accuracy is: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="./dataset/train", help="Path for dataset")
    parser.add_argument("--weight_file", default="./models/", help="Path for the weight file")
    parser.add_argument("--output_data", default="./output", help="Cropped image storage path")
    args = parser.parse_args()
    main(args)