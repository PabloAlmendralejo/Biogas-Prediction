import boto3
import pandas as pd

def load_csv_from_s3(bucket_name, key, aws_access_key_id=None, aws_secret_access_key=None):
    """
    Load CSV data from S3 into pandas DataFrame.
    """
    s3 = boto3.client('s3', 
                      aws_access_key_id=aws_access_key_id, 
                      aws_secret_access_key=aws_secret_access_key)
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    df = pd.read_csv(obj['Body'])
    return df
