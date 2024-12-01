#!/usr/bin/env python
# coding: utf-8

# Install required packages
get_ipython().system('pip install ipython-autotime')
get_ipython().system('pip install bing-image-downloader')
get_ipython().system('mkdir i2mages')

# Enables tracking execution time for each cell
get_ipython().run_line_magic('load_ext', 'autotime')

# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from bing_image_downloader import downloader
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import requests
from io import BytesIO

# -----------------------------------------
# 1. IMAGE COLLECTION
# Automates dataset collection using the bing-image-downloader library.
# Saves images into specific folders by category.
# -----------------------------------------

downloader.download("sunflower", limit=30, output_dir='i2mages', adult_filter_off=True)
downloader.download("rugby ball", limit=30, output_dir='i2mages', adult_filter_off=True)
downloader.download("Ice cream cone", limit=30, output_dir='i2mages', adult_filter_off=True)

# -----------------------------------------
# 2. DATA PREPROCESSING
# Reads and resizes images to 150x150 pixels with 3 color channels.
# Converts resized images into flattened vectors for ML model compatibility.
# Labels images by category using integer encoding.
# -----------------------------------------

DATADIR = 'i2mages'
CATEGORIES = ['Ice cream cone', 'rugby ball', 'sunflower']

flat_data = []  # Flattened image data for ML
images = []     # Resized image data for visualization
target = []     # Labels corresponding to image categories

for category in CATEGORIES:
    class_num = CATEGORIES.index(category)  # Label encoding
    path = os.path.join(DATADIR, category)  # Create path to category folder
    for img in os.listdir(path):
        if img.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.gif')):
            img_array = imread(os.path.join(path, img))
            img_resized = resize(img_array, (150, 150, 3))  # Resize to uniform dimensions
            flat_data.append(img_resized.flatten())
            images.append(img_resized)
            target.append(class_num)

flat_data = np.array(flat_data)
target = np.array(target)
images = np.array(images)

# -----------------------------------------
# 3. DATASET PREPARATION
# Organizes data into flat_data, images, and target arrays.
# Splits the data into training (70%) and testing (30%) sets.
# -----------------------------------------

x_train, x_test, y_train, y_test = train_test_split(flat_data, target, test_size=0.3, random_state=109)

# -----------------------------------------
# 4. MODEL TRAINING
# Uses a Support Vector Machine (SVM) classifier.
# Performs hyperparameter tuning with GridSearchCV.
# Trains the model on x_train and y_train.
# -----------------------------------------

param_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]
svc = svm.SVC(probability=True)
clf = GridSearchCV(svc, param_grid)  # Perform hyperparameter tuning
clf.fit(x_train, y_train)

# -----------------------------------------
# 5. EVALUATION
# Predicts categories for the test set (x_test).
# Computes accuracy and confusion matrix for evaluation.
# -----------------------------------------

y_pred = clf.predict(x_test)
print("Accuracy:", accuracy_score(y_pred, y_test))
print("Confusion Matrix:\n", confusion_matrix(y_pred, y_test))

# -----------------------------------------
# 6. MODEL SERIALIZATION
# Saves the trained model using pickle for later use.
# -----------------------------------------

pickle.dump(clf, open('img_model.p', 'wb'))

# -----------------------------------------
# 7. INFERENCE ON NEW IMAGES
# Accepts a URL for a new image, resizes it, and predicts the category.
# -----------------------------------------

flat_data = []
url = input('Enter your URL: ')

# Fetch the image from the URL
response = requests.get(url)
if response.status_code == 200:
    image_data = BytesIO(response.content)  # Convert response content to bytes stream
    img = imread(image_data)
    img_resized = resize(img, (150, 150, 3))  # Resize the image
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)

    # Display the resized image
    plt.imshow(img_resized)
    plt.show()

    # Predict the category
    y_out = clf.predict(flat_data)
    print(f'Predicted Output: {CATEGORIES[y_out[0]]}')
else:
    print("Failed to fetch the image. Please check the URL.")
