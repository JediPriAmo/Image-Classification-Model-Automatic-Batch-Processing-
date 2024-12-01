import os
import csv
import time
import pickle
from datetime import datetime
from skimage.io import imread
from skimage.transform import resize
import numpy as np

# Load the pre-trained model
MODEL_FILE = "img_model.p"  # Ensure this matches your model file
model = pickle.load(open(MODEL_FILE, "rb"))
CATEGORIES = ['Ice cream cone', 'Rugby ball', 'Sunflower']

# Input and output paths
IMAGE_STORAGE_PATH = "c:/Users/Admin/image1/Refined/project/image_storage"
RESULTS_FILE = "c:/Users/Admin/image1/Refined/project/processed_results.csv"

#Load the list of already processed file names from results.csv.
def load_processed_files():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as file:
            reader = csv.DictReader(file)
            return {row["File Name"] for row in reader}
    return set()


#Load new image file paths from the storage folder, excluding already processed files.
def load_images_from_storage(processed_files):
    return [
        os.path.join(IMAGE_STORAGE_PATH, f)
        for f in os.listdir(IMAGE_STORAGE_PATH)
        if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.gif')) and f not in processed_files
    ]

#Process an image for prediction.
def process_image(image_path):
    try:
        img = imread(image_path)
        img_resized = resize(img, (150, 150, 3))
        flat_data = np.array([img_resized.flatten()])
        prediction = model.predict(flat_data)
        probabilities = model.predict_proba(flat_data)[0]
        confidence = max(probabilities)
        return {
            "file_name": os.path.basename(image_path),
            "prediction": CATEGORIES[prediction[0]],
            "confidence": confidence,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": "",
        }
    except Exception as e:
        return {
            "file_name": os.path.basename(image_path),
            "prediction": "",
            "confidence": "",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": str(e),
        }


#Save prediction results to a CSV file.
def save_results_to_csv(results):
    file_exists = os.path.isfile(RESULTS_FILE)
    with open(RESULTS_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write header if the file is new
            writer.writerow(["File Name", "Prediction", "Confidence", "Timestamp", "Error"])
        for result in results:
            writer.writerow([
                result.get("file_name"),
                result.get("prediction", ""),
                result.get("confidence", ""),
                result.get("timestamp"),
                result.get("error", ""),
            ])

#Run batch prediction on new images in storage.            
def batch_predict():
    print("Starting batch processing...")
    
    # Load already processed files
    processed_files = load_processed_files()
    print(f"Found {len(processed_files)} already processed files. Skipping them.")

    # Load new images to process
    images = load_images_from_storage(processed_files)
    if not images:
        print("No new images found in storage.")
        return

    results = []
    for image_path in images:
        print(f"Processing: {image_path}")
        result = process_image(image_path)
        results.append(result)
        print(f"Result: {result}")

    # Save new results to CSV
    save_results_to_csv(results)
    print("Batch processing complete. Results saved to", RESULTS_FILE)

if __name__ == "__main__":
    start_time = time.time()
    batch_predict()
    print("Execution time: {:.2f} seconds".format(time.time() - start_time))
