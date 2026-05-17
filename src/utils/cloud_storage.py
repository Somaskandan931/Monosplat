"""
cloud_storage.py
Cloud storage integration for MonoSplat pipeline.

Supports:
- AWS S3 (via boto3)
- Google Cloud Storage (via google-cloud-storage)
- Local filesystem fallback

This enables true cloud deployment where files are stored in object storage
rather than local disk, enabling horizontal scaling and multi-worker setups.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urlparse

# Try to import cloud storage libraries
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    from google.cloud import storage as gcs
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False


class CloudStorageBackend:
    """Abstract base class for cloud storage backends."""
    
    def upload_file(self, local_path: str, remote_key: str) -> str:
        """Upload a file and return the public URL."""
        raise NotImplementedError
    
    def download_file(self, remote_key: str, local_path: str) -> None:
        """Download a file from cloud storage."""
        raise NotImplementedError
    
    def get_public_url(self, remote_key: str) -> str:
        """Get a public URL for a file."""
        raise NotImplementedError
    
    def file_exists(self, remote_key: str) -> bool:
        """Check if a file exists in cloud storage."""
        raise NotImplementedError


class S3Backend(CloudStorageBackend):
    """AWS S3 storage backend."""
    
    def __init__(self, bucket_name: str, region: str = "us-east-1", 
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None):
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for S3 support. Install with: pip install boto3")
        
        self.bucket_name = bucket_name
        self.region = region
        
        # Initialize S3 client
        session_kwargs = {}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs['aws_access_key_id'] = aws_access_key_id
            session_kwargs['aws_secret_access_key'] = aws_secret_access_key
        
        self.s3_client = boto3.client('s3', region_name=region, **session_kwargs)
        self.s3_resource = boto3.resource('s3', region_name=region, **session_kwargs)
        self.bucket = self.s3_resource.Bucket(bucket_name)
    
    def upload_file(self, local_path: str, remote_key: str) -> str:
        """Upload a file to S3 and return the public URL."""
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        try:
            self.bucket.upload_file(str(local_path), remote_key)
            print(f"[cloud] Uploaded {local_path.name} → s3://{self.bucket_name}/{remote_key}")
            return self.get_public_url(remote_key)
        except ClientError as e:
            raise RuntimeError(f"S3 upload failed: {e}")
    
    def download_file(self, remote_key: str, local_path: str) -> None:
        """Download a file from S3."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.bucket.download_file(remote_key, str(local_path))
            print(f"[cloud] Downloaded s3://{self.bucket_name}/{remote_key} → {local_path}")
        except ClientError as e:
            raise RuntimeError(f"S3 download failed: {e}")
    
    def get_public_url(self, remote_key: str) -> str:
        """Get a public URL for an S3 object."""
        # For public buckets, use the standard S3 URL format
        return f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{remote_key}"
    
    def file_exists(self, remote_key: str) -> bool:
        """Check if a file exists in S3."""
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=remote_key)
            return True
        except ClientError:
            return False


class GCSBackend(CloudStorageBackend):
    """Google Cloud Storage backend."""
    
    def __init__(self, bucket_name: str, credentials_path: Optional[str] = None):
        if not GCS_AVAILABLE:
            raise ImportError("google-cloud-storage is required for GCS support. Install with: pip install google-cloud-storage")
        
        self.bucket_name = bucket_name
        
        # Initialize GCS client
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        self.client = gcs.Client()
        self.bucket = self.client.bucket(bucket_name)
    
    def upload_file(self, local_path: str, remote_key: str) -> str:
        """Upload a file to GCS and return the public URL."""
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        try:
            blob = self.bucket.blob(remote_key)
            blob.upload_from_filename(str(local_path))
            print(f"[cloud] Uploaded {local_path.name} → gs://{self.bucket_name}/{remote_key}")
            return self.get_public_url(remote_key)
        except Exception as e:
            raise RuntimeError(f"GCS upload failed: {e}")
    
    def download_file(self, remote_key: str, local_path: str) -> None:
        """Download a file from GCS."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            blob = self.bucket.blob(remote_key)
            blob.download_to_filename(str(local_path))
            print(f"[cloud] Downloaded gs://{self.bucket_name}/{remote_key} → {local_path}")
        except Exception as e:
            raise RuntimeError(f"GCS download failed: {e}")
    
    def get_public_url(self, remote_key: str) -> str:
        """Get a public URL for a GCS object."""
        blob = self.bucket.blob(remote_key)
        # Make blob public if not already
        if not blob.public_url:
            blob.make_public()
        return blob.public_url
    
    def file_exists(self, remote_key: str) -> bool:
        """Check if a file exists in GCS."""
        blob = self.bucket.blob(remote_key)
        return blob.exists()


class LocalBackend(CloudStorageBackend):
    """Local filesystem fallback backend (for development)."""
    
    def __init__(self, base_path: str = "cloud_storage"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def upload_file(self, local_path: str, remote_key: str) -> str:
        """Copy file to local storage directory."""
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        dest_path = self.base_path / remote_key
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        import shutil
        shutil.copy2(local_path, dest_path)
        print(f"[cloud] Copied {local_path.name} → {dest_path}")
        return str(dest_path)
    
    def download_file(self, remote_key: str, local_path: str) -> None:
        """Copy file from local storage directory."""
        src_path = self.base_path / remote_key
        if not src_path.exists():
            raise FileNotFoundError(f"File not found in local storage: {src_path}")
        
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        import shutil
        shutil.copy2(src_path, local_path)
        print(f"[cloud] Copied {src_path} → {local_path}")
    
    def get_public_url(self, remote_key: str) -> str:
        """Return local file path as URL (for development)."""
        return f"file://{self.base_path / remote_key}"
    
    def file_exists(self, remote_key: str) -> bool:
        """Check if file exists in local storage."""
        return (self.base_path / remote_key).exists()


def get_cloud_storage(config: Dict[str, Any]) -> CloudStorageBackend:
    """
    Factory function to get the appropriate cloud storage backend.
    
    Config format:
    {
        "type": "s3" | "gcs" | "local",
        "bucket": "bucket-name",
        "region": "us-east-1",  # for S3
        "credentials_path": "/path/to/creds.json",  # for GCS
        "aws_access_key_id": "...",  # optional for S3
        "aws_secret_access_key": "...",  # optional for S3
        "local_path": "cloud_storage"  # for local backend
    }
    """
    storage_type = config.get("type", "local").lower()
    
    if storage_type == "s3":
        return S3Backend(
            bucket_name=config["bucket"],
            region=config.get("region", "us-east-1"),
            aws_access_key_id=config.get("aws_access_key_id"),
            aws_secret_access_key=config.get("aws_secret_access_key")
        )
    elif storage_type == "gcs":
        return GCSBackend(
            bucket_name=config["bucket"],
            credentials_path=config.get("credentials_path")
        )
    elif storage_type == "local":
        return LocalBackend(
            base_path=config.get("local_path", "cloud_storage")
        )
    else:
        raise ValueError(f"Unknown storage type: {storage_type}. Must be 's3', 'gcs', or 'local'")


def upload_job_to_cloud(job_id: str, work_dir: str, storage: CloudStorageBackend) -> Dict[str, str]:
    """
    Upload all job outputs to cloud storage.
    
    Returns a dict mapping file types to their cloud URLs:
    {
        "splat": "https://...",
        "ply": "https://...",
        "spz": "https://...",
        "thumbnail": "https://..."
    }
    """
    work_path = Path(work_dir)
    urls = {}
    
    # Upload .splat file
    splat_path = work_path / "models" / "gaussian" / f"{job_id}.splat"
    if splat_path.exists():
        urls["splat"] = storage.upload_file(str(splat_path), f"jobs/{job_id}/{job_id}.splat")
    
    # Upload .ply file
    ply_path = work_path / "models" / "gaussian" / f"{job_id}.ply"
    if ply_path.exists():
        urls["ply"] = storage.upload_file(str(ply_path), f"jobs/{job_id}/{job_id}.ply")
    
    # Upload .spz file
    spz_path = work_path / "models" / "gaussian" / f"{job_id}.spz"
    if spz_path.exists():
        urls["spz"] = storage.upload_file(str(spz_path), f"jobs/{job_id}/{job_id}.spz")
    
    # Upload thumbnail
    thumb_path = work_path / "thumbnail.png"
    if thumb_path.exists():
        urls["thumbnail"] = storage.upload_file(str(thumb_path), f"jobs/{job_id}/thumbnail.png")
    
    # Upload chunks if they exist
    chunks_dir = work_path / "models" / "gaussian" / f"{job_id}_chunks"
    if chunks_dir.exists():
        for chunk_file in chunks_dir.glob("*.splat"):
            urls[f"chunk_{chunk_file.stem}"] = storage.upload_file(
                str(chunk_file), 
                f"jobs/{job_id}/chunks/{chunk_file.name}"
            )
        # Upload manifest
        manifest_path = chunks_dir / "manifest.json"
        if manifest_path.exists():
            urls["chunks_manifest"] = storage.upload_file(
                str(manifest_path),
                f"jobs/{job_id}/chunks/manifest.json"
            )
    
    print(f"[cloud] Uploaded {len(urls)} files for job {job_id}")
    return urls
