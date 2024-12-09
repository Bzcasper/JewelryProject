import os
import subprocess
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from google.cloud import storage
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from dotenv import load_dotenv
import json
import logging
from rich.logging import RichHandler
import time

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder=None)
CORS(app)

# Configure logging with RichHandler for colored logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)

# Google Cloud Storage configuration
GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME')
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET_NAME)

# Azure Computer Vision configuration
AZURE_VISION_API_KEY = os.getenv('AZURE_VISION_API_KEY')
AZURE_VISION_ENDPOINT = os.getenv('AZURE_VISION_ENDPOINT')
azure_cv_client = ComputerVisionClient(AZURE_VISION_ENDPOINT, CognitiveServicesCredentials(AZURE_VISION_API_KEY))

# Retry decorator
def retry(ExceptionToCheck, tries=3, delay=2, backoff=2):
    def deco_retry(f):
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except ExceptionToCheck as e:
                    logger.warning(f"{str(e)}, Retrying in {mdelay} seconds...")
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)
        return f_retry
    return deco_retry

# Endpoint to handle file uploads and pipeline selection
@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        # Get pipeline selections
        pipelines = json.loads(request.form.get('pipelines'))
        logger.info(f"Selected pipelines: {pipelines}")

        # Save uploaded files
        for file in request.files.getlist('files'):
            rel_path = file.filename.replace('/', '\\')  # Ensure Windows-style paths
            raw_dir = os.path.join(os.getcwd(), 'data', 'raw')
            save_path = os.path.join(raw_dir, rel_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            file.save(save_path)
            logger.info(f"Saved file: {save_path}")

        # Run selected pipelines
        if any(pipelines.values()):
            # Image Processing
            if pipelines.get('imageProcessing'):
                logger.info("Starting Image Processing Pipeline...")
                subprocess.run([
                    'python',
                    os.path.join(os.getcwd(), 'scripts', 'image_processing.py'),
                    os.path.join(os.getcwd(), 'data', 'raw'),
                    os.path.join(os.getcwd(), 'data', 'cropped'),
                    os.path.join(os.getcwd(), 'data', 'augmented')
                ], check=True)
                logger.info("Image Processing Pipeline completed.")

            # Moondream Classification
            if pipelines.get('moondreamClassification'):
                logger.info("Starting Moondream Classification Pipeline...")
                subprocess.run([
                    'python',
                    os.path.join(os.getcwd(), 'scripts', 'moondream_classification.py'),
                    os.path.join(os.getcwd(), 'data', 'augmented')
                ], check=True)
                logger.info("Moondream Classification Pipeline completed.")

            # Vision Analysis with Fallback to Azure Vision
            if pipelines.get('visionAnalysis'):
                logger.info("Starting Vision Analysis Pipeline...")
                try:
                    subprocess.run([
                        'python',
                        os.path.join(os.getcwd(), 'scripts', 'vision_api.py'),
                        os.path.join(os.getcwd(), 'data', 'augmented')
                    ], check=True)
                    logger.info("Vision Analysis Pipeline completed using Google Vision API.")
                except subprocess.CalledProcessError:
                    logger.error("Google Vision API failed. Falling back to Azure Vision API.")
                    subprocess.run([
                        'python',
                        os.path.join(os.getcwd(), 'scripts', 'azure_vision_api.py'),
                        os.path.join(os.getcwd(), 'data', 'augmented')
                    ], check=True)
                    logger.info("Vision Analysis Pipeline completed using Azure Vision API.")

            # eBay Listing Generation
            if pipelines.get('ebayListing'):
                logger.info("Starting eBay Listing Generation Pipeline...")
                subprocess.run([
                    'python',
                    os.path.join(os.getcwd(), 'scripts', 'ebay_listing_generator.py'),
                    os.path.join(os.getcwd(), 'data', 'augmented'),
                    os.path.join(os.getcwd(), 'metadata', 'listings.csv')
                ], check=True)
                logger.info("eBay Listing Generation Pipeline completed.")

            # GPT-2 Text Generation
            if pipelines.get('gpt2TextGeneration'):
                logger.info("Starting GPT-2 Text Generation Pipeline...")
                subprocess.run([
                    'python',
                    os.path.join(os.getcwd(), 'scripts', 'gpt2_text_generation.py'),
                    os.path.join(os.getcwd(), 'metadata', 'listings.csv'),
                    os.path.join(os.getcwd(), 'metadata', 'generated_text.csv')
                ], check=True)
                logger.info("GPT-2 Text Generation Pipeline completed.")

            # Caption Generation
            if pipelines.get('captionGeneration'):
                logger.info("Starting Caption Generation Pipeline...")
                subprocess.run([
                    'python',
                    os.path.join(os.getcwd(), 'scripts', 'captions_generator.py'),
                    os.path.join(os.getcwd(), 'data', 'augmented'),
                    os.path.join(os.getcwd(), 'metadata', 'captions.csv')
                ], check=True)
                logger.info("Caption Generation Pipeline completed.")

            # Dataset Preparation
            if pipelines.get('datasetPreparation'):
                logger.info("Starting Dataset Preparation Pipeline...")
                subprocess.run([
                    'python',
                    os.path.join(os.getcwd(), 'scripts', 'dataset_preparation.py'),
                    os.path.join(os.getcwd(), 'data', 'augmented'),
                    os.path.join(os.getcwd(), 'metadata', 'captions.csv'),
                    os.path.join(os.getcwd(), 'data', 'resnet_dataset'),
                    os.path.join(os.getcwd(), 'data', 'llava_dataset.zip')
                ], check=True)
                logger.info("Dataset Preparation Pipeline completed.")

        # Upload processed data to Google Cloud Storage
        logger.info("Starting upload to Google Cloud Storage...")
        upload_to_gcs(os.path.join(os.getcwd(), 'data', 'augmented'), 'augmented/')
        upload_to_gcs(os.path.join(os.getcwd(), 'metadata'), 'metadata/')
        upload_to_gcs(os.path.join(os.getcwd(), 'data', 'resnet_dataset'), 'datasets/resnet_dataset/')
        upload_to_gcs(os.path.join(os.getcwd(), 'data', 'llava_dataset.zip'), 'datasets/llava_dataset.zip')
        logger.info("Upload to Google Cloud Storage completed.")

        return jsonify({"message": "Upload and processing complete."}), 200

    except subprocess.CalledProcessError as e:
        logger.error(f"Subprocess error: {e}")
        return jsonify({"message": "An error occurred during processing."}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({"message": "An unexpected error occurred."}), 500

@retry(Exception, tries=3, delay=2, backoff=2)
def upload_to_gcs(local_path, gcs_path):
    try:
        if os.path.isdir(local_path):
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, local_path)
                    blob = bucket.blob(os.path.join(gcs_path, relative_path).replace('\\', '/'))
                    blob.upload_from_filename(file_path)
                    logger.info(f"Uploaded {file_path} to gs://{GCS_BUCKET_NAME}/{gcs_path}{relative_path}")
        else:
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            logger.info(f"Uploaded {local_path} to gs://{GCS_BUCKET_NAME}/{gcs_path}")
    except Exception as e:
        logger.error(f"Failed to upload {local_path} to GCS: {e}")
        raise e

# Serve frontend build
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path != "" and os.path.exists(os.path.join(os.getcwd(), 'frontend', 'build', path)):
        return send_from_directory(os.path.join(os.getcwd(), 'frontend', 'build'), path)
    else:
        return send_from_directory(os.path.join(os.getcwd(), 'frontend', 'build'), 'index.html')

if __name__ == '__main__':
    # Run the server
    app.run(host='0.0.0.0', port=5000, debug=False)
