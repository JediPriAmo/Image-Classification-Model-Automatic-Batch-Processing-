import schedule
import time
import os
import logging
from datetime import datetime

#Logging setup
logging.basicConfig(filename='nightly_scheduler.log', level=logging.INFO)

def run_batch_prediction():
    logging.info(f"Nightly batch prediction started at {datetime.now()}")
    os.system("python batch_predict.py")  #run the batch prediction script.

# schedule the task to run every night at midnight
schedule.every().day.at("00:00").do(run_batch_prediction)

if __name__ == "__main__":
    logging.info("Nightly scheduler started.")
    while True:
        schedule.run_pending()
        time.sleep(1)
