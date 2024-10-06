
from qwak.feature_store.feature_sets import streaming
from qwak.feature_store.feature_sets.execution_spec import ClusterTemplate
from qwak.feature_store.feature_sets.transformations import SparkSqlTransformation, QwakAggregation


@streaming.feature_set(
    name="credit-risk-streaming",
    data_sources=["transactions"],
    key="user_id",
    timestamp_column_name="timestamp",
    offline_scheduling_policy="0 * * * *",
    online_trigger_interval=30
)
@streaming.metadata(
        display_name="Credit Risk Streaming",
        description="streaming transaction aggregations over 1,15,30,60 minutes",
        owner="hudson@qwak.com"
)

@streaming.execution_specification(
    online_cluster_template=ClusterTemplate.SMALL,
    offline_cluster_template=ClusterTemplate.MEDIUM,
)
def transform():
    return SparkSqlTransformation(sql="""SELECT * FROM transactions""")\
            .aggregate(QwakAggregation.sum("transaction_amount"))\
            .aggregate(QwakAggregation.count("transaction_amount"))\
            .aggregate(QwakAggregation.max("transaction_amount"))\
            .aggregate(QwakAggregation.sample_stdev("transaction_amount"))\
            .aggregate(QwakAggregation.last_n("transaction_amount", 5))\
            .aggregate(QwakAggregation.last_distinct_n("transaction_amount", 5))\
            .aggregate(QwakAggregation.percentile("transaction_amount", 50)\
                    .alias("median_transaction_amount"))\
            .by_windows("1 minute", "15 minutes", "30 minutes", "1 hour")
