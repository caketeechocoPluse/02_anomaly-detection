# s3_uploader.py
import boto3
import os
from dotenv import load_dotenv

load_dotenv()

class S3Uploader:
    def __init__(self) -> None:
        self.s3_client = boto3.client(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACESS_KEY'),
            region_name=os.getenv('AWS_REGION')
        )
    
    def upload_csv(self, file_path: str, bucket_name: str, s3_key: str) -> bool:
        """로컬 CSV 파일을 S3 버킷의 경로로 업로드"""
        try:
            self.s3_client.upload_file(file_path, bucket_name, s3_key)
            return True
        except Exception as e:
            print(f"Upload failed: {e}")
            return False