from flask import Flask, request, jsonify
import numpy as np
import pickle
import requests
from skimage.io import imread
from skimage.transform import resize
from io import BytesIO
import csv
from datetime import datetime
import os

#Initialize Flask app
app = Flask(__name__)

#Load the pre-trained model.
model = pickle.load(open('img_model.p', 'rb'))
CATEGORIES = ['Ice cream cone', 'rugby ball', 'sunflower']
RESULTS_FILE = "c:/Users/Admin/image1/Refined/project//processed_results.csv"  # Path to the CSV file for results logging


#Append results to the CSV file.
def save_result_to_csv(url, prediction, confidence, timestamp, error=None):
    file_exists = os.path.isfile(RESULTS_FILE)
    with open(RESULTS_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write headers if the file is new.
            writer.writerow(["Image URL", "Prediction", "Confidence", "Timestamp", "Error"])
        writer.writerow([url, prediction, confidence, timestamp, error])


@app.route('/')
def hello_world():
    return 'Welcome to the Image Classification API!'


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return jsonify({
            "instructions": "Submit a POST request with a JSON body containing {'image_urls': ['<url1>', '<url2>', ...]} to classify images and receive predictions.",
        })

    if request.method == 'POST':
        try:
            data = request.get_json()
            image_urls = data.get('image_urls', [])
            
            if not image_urls or not isinstance(image_urls, list):
                return jsonify({"error": "Invalid input. Provide a JSON with a list of 'image_urls'."}), 400

            results = []  #Store predictions for each image
            for url in image_urls:
                try:
                    #Fetch image from URL
                    response = requests.get(url)
                    if response.status_code != 200:
                        error_message = "Failed to fetch the image."
                        results.append({"url": url, "error": error_message})
                        save_result_to_csv(url, None, None, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), error_message)
                        continue

                    #Process the image
                    image_data = BytesIO(response.content)
                    img = imread(image_data)
                    img_resized = resize(img, (150, 150, 3))
                    flat_data = np.array([img_resized.flatten()])

                    #Make prediction.
                    prediction = model.predict(flat_data)
                    probabilities = model.predict_proba(flat_data)[0]
                    confidence = max(probabilities)
                    predicted_category = CATEGORIES[prediction[0]]
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    #Append results and log them to CSV
                    results.append({
                        "url": url,
                        "prediction": predicted_category,
                        "confidence": confidence,
                        "timestamp": timestamp,
                    })
                    save_result_to_csv(url, predicted_category, confidence, timestamp)

                except Exception as e:
                    error_message = str(e)
                    results.append({"url": url, "error": error_message})
                    save_result_to_csv(url, None, None, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), error_message)

            return jsonify(results)
        except Exception as e:
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5003, debug=True)


