from datetime import datetime
from qwak.feature_store.feature_sets import batch
from qwak.feature_store.feature_sets.read_policies import ReadPolicy
from qwak.feature_store.feature_sets.transformations import SparkSqlTransformation


@batch.feature_set(
    name="user-credit-risk-features",
    entity="user_id",
    data_sources={"credit_risk_data": ReadPolicy.NewOnly},
    timestamp_column_name="date_created"
)
@batch.scheduling(cron_expression="0 0 * * *")
@batch.backfill(start_date=datetime(2021, 1, 1, 2, 0, 0))
def transform():
    return SparkSqlTransformation(sql="""
SELECT user_id as user_id,
       date_created as date_created
FROM credit_risk_data
""")
