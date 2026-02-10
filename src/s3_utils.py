import logging
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import pandas as pd

logger = logging.getLogger(__name__)

def load_csv_from_s3(
    bucket_name: str, 
    key: str, 
    aws_access_key_id: str = None, 
    aws_secret_access_key: str = None
) -> pd.DataFrame:

    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        logger.info(f"Fetching s3://{bucket_name}/{key}")
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        df = pd.read_csv(obj['Body'])
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except s3.exceptions.NoSuchKey:
        raise FileNotFoundError(f"Object not found: s3://{bucket_name}/{key}")
    except NoCredentialsError:
        raise PermissionError("AWS credentials not found or invalid")
    except ClientError as e:
        logger.error(f"S3 error: {e}")
        raise ConnectionError(f"Failed to connect to S3: {e}")
