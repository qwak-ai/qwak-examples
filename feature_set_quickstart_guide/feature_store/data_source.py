from frogml.feature_store.data_sources import CsvSource

# The S3 anonymous config class is required for public S3 buckets
from frogml.feature_store.data_sources import AnonymousS3Configuration


# Create a CsvSource object to represent a CSV data source 
# This example uses a CSV file from a public S3 bucket
csv_source = CsvSource(
    name='credit_risk_data',                                    # Name of the data source
    description='A dataset of personal credit details',         # Description of the data source
    date_created_column='date_created',                         # Column name that represents the creation date
    path='s3://qwak-public/example_data/data_credit_risk.csv',  # S3 path to the CSV file 
    filesystem_configuration=AnonymousS3Configuration(),        # Configuration for anonymous access to S3
    quote_character='"',                                        # Character used for quoting in the CSV file
    escape_character='"'                                        # Character used for escaping in the CSV file
)
