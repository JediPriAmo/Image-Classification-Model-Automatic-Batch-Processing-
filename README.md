# Refund Item Classification System

## Overview
This project is designed to automate the classification of returned items for an online shopping platform. The system uses a machine learning model to categorize items based on images and includes features for batch processing and a REST API for real-time predictions. The ultimate goal is to reduce manual labor and costs associated with sorting refund items.

## Features
- **Batch Prediction**: Automatically processes refund item images overnight.
- **REST API**: Provides real-time predictions with confidence scores for item categories.
- **Modularity**: Organized into clear modules for image storage, batch processing, REST API, and nightly scheduling.
- **Efficiency**: Reduces manual sorting, saving time and workforce resources.

---

## Setup Instructions

### 1. Platform: Jupyter Lab & Image Storage
1. Inside the Jupyter Lab environment, create a folder directory and store all the project files in it.  
   **Example**: `C:/Users/Admin/YourFolderName`
2. Create an `image_storage` folder inside this directory. This is where incoming refund product images will be stored.
3. Add images to the `image_storage` folder, such as pictures of sunflower, rugby ball, or ice cream cone. Ensure the image extensions are among the following: `.jpg`, `.png`, `.jpeg`, `.JPG`, `.gif`.  
   **Note**: You can modify and retrain the `image_model.py` file to include additional extensions and save it as a pickle file.

---

### 2. Batch Processing Module & Results CSV
1. Copy the path to the `image_storage` folder.
2. Open the `batch_predict.py` file and update the paths:
   - **Image Storage Path**: Replace `IMAGE_STORAGE_PATH` with your folder's path.
   - **Results File Path**: Update `RESULTS_FILE` accordingly.  
   **Example**:
   ```python
   IMAGE_STORAGE_PATH = "C:/Users/Admin/YourFolderName/image_storage"
   RESULTS_FILE = "C:/Users/Admin/YourFolderName/processed_results.csv"

3. Save the changes (CTRL + S).
4. Open the terminal in the same directory and run the following command:
```bash
python batch_predict.py
```

5. A new file, processed_results.csv, will be created in the directory to store the results from batch_predict.py and r2fineAPI.py.

---

### 3. Nightly Schedular & Logs
1. Run the nightly_schedular.py script in the terminal:
```bash
python nightly_schedular.py
```

2. A nightly_scheduler.log file will appear, logging the execution of the nightly batch prediction.

---

### 4. REST API: r2fineAPI.py & r2API_URL.ipynb
1. Open the r2fineAPI.py script and update the RESULTS_FILE path:
```python
RESULTS_FILE = "C:/Users/Admin/YourFolderName/processed_results.csv"
```

- **NOTE**: At the bottom of the script, the default port is set to 5003. If another service is using this port, change it to 5004 or another number of your choice.

2. Save the script (CTRL + S).
3. Close the terminal running nightly_schedular.py and open a new terminal.
4. Run the following command:
```bash
python r2fineAPI.py
```

5. You will see a message like:
```csharp
* Running on http://127.0.0.1:5003
```

6. Open the r2API_URL.ipynb notebook and:
- Paste the above URL into the corresponding cell.
- Replace or add new image URLs to test predictions.
7. Execute the notebook cell to get the output in the following structure:
- Confidence, Prediction, Timestamp, URL.
8. API results are logged to processed_results.csv.
Execute the notebook cell to get the output in the following structure:
Confidence, Prediction, Timestamp, URL.
API results are logged to processed_results.csv.

---

### Notes
- Ensure all scripts and directories are properly set up before running.
- Check the logs for troubleshooting (nightly_scheduler.log or terminal outputs).
- Modify scripts as needed to fit your environment or additional requirements.

---

### Contributing
Feel free to open issues or submit pull requests for improvements. Contributions are always welcome!

---

### License
This project is open-source and available under the MIT License.



