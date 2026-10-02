"""Environment-only credentials; never log credential values."""
import os
import boto3
from botocore.config import Config

PROJECT = 'q35ocr-20260921-p560'
BUCKET = os.getenv('PROJECT_BUCKET', 'qwen35-9b-ocr-finetuning')

def client():
    return boto3.client('s3', endpoint_url=os.environ['AWS_ENDPOINT_URL'],
        region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1'),
        config=Config(signature_version='s3v4', connect_timeout=15, read_timeout=120,
                      retries={'max_attempts': 5, 'mode': 'standard'}, s3={'addressing_style': 'path'}))
