from qwak.feature_store.sources.data_sources import CsvSource
from qwak.feature_store.entities import Entity, ValueType
from qwak.feature_store.features.feature_sets import BatchFeatureSet, Metadata, Backfill
from qwak.feature_store.features.functions import SqlFunction
from qwak.feature_store.features.read_policies import ReadPolicy
from datetime import datetime

# Add Environment specific credit risk file
S3_CREDIT_RISK_FILE = ""

batch_feature_set = BatchFeatureSet(
    name='user_credit_risk_features_v2',
    metadata=Metadata(
        display_name='[Batch] User Credit Risk Features',
        description='[Batch] User Credit Features',
        owner='[Batch] User Credit Risk Features'),
    entity='user',
    data_sources={
        'credit_risk_data': ReadPolicy.NewOnly
    },
    backfill=Backfill(start_date=datetime(2015, 1, 1)),
    scheduling_policy='@yearly',
    function=SqlFunction("""
        SELECT user_id,
               age,
               sex,
               job,
               housing,
               saving_account,
               checking_account,
               credit_amount,
               duration,
               purpose
        FROM credit_risk_data
        """)
)

entity = Entity(
    name='user',
    description='A User ID',
    key=['user_id'],
    value_type=ValueType.STRING)

csv_source = CsvSource(
    name='credit_risk_data',
    description='a csv source description',
    date_created_column='DATE_CREATED',
    path=S3_CREDIT_RISK_FILE,
    quote_character='"',
    escape_character='"'
)

print(csv_source.get_sample())
print(batch_feature_set.get_sample())
