import os
from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from qwak.feature_store.entities import Entity, ValueType
from pyspark.sql.functions import col, from_json
from qwak.feature_store.sources.data_sources import KafkaSourceV1, CustomDeserializer

from qwak.feature_store.features import Metadata
from qwak.feature_store.features.aggregations import QwakAggregation
from qwak.feature_store.features.execution_spec import StreamingExecutionSpec, ClusterTemplate, ResourceConfiguration
from qwak.feature_store.features.streaming_feature_sets import StreamingFeatureSetV1
from qwak.feature_store.features.transform import SqlTransformation


def deser_function(df):
    schema = StructType(
        [
            StructField("timestamp", TimestampType()),
            StructField("user_id", StringType()),
            StructField("transaction_amount", IntegerType()),
        ]
    )
    deserialized = df.select(col("partition"), col("topic"), col("offset"),
                             from_json(col("value").cast(StringType()), schema).alias("data")
                             ).select(col("data.*"), col("partition"), col("topic"), col("offset")) \
        .select(col("partition"), col("topic"), col("offset"), col("timestamp"), col("user_id"),
                col("transaction_amount"))

    return deserialized


deserializer = CustomDeserializer(f=deser_function)
bootstrap_servers = 'b-2.qwak-cluster.za36zh.c6.kafka.us-east-1.amazonaws.com:9094,b-1.qwak-cluster.za36zh.c6.kafka.us-east-1.amazonaws.com:9094,b-3.qwak-cluster.za36zh.c6.kafka.us-east-1.amazonaws.com:9094'

passthrough_configs = {
    "qwak.online.maxOffsetsPerTrigger": "200000",
    "startingOffsets": "earliest",
}

kafka_source = KafkaSourceV1(
    name="transactions",
    description="Transaction Test Source",
    bootstrap_servers=bootstrap_servers,
    subscribe="transactions",
    deserialization=deserializer,
    passthrough_configs=passthrough_configs,
)

entity = Entity(
    name="user_id", description="A User ID", key="user_id", value_type=ValueType.STRING
)

streaming_fs = StreamingFeatureSetV1(
    name="transaction-aggregates-demo",
    metadata=Metadata(
        display_name="Transaction Aggregates",
        description="streaming transaction aggregations over 1,15,30,60 minutes",
        owner="hudson@qwak.com",
    ),
    data_sources=["transactions"],
    entity="user_id",
    transform=SqlTransformation(sql="""SELECT * FROM transactions""")
        .aggregate(QwakAggregation.sum("transaction_amount"))
        .aggregate(QwakAggregation.count("transaction_amount"))
        .aggregate(QwakAggregation.max("transaction_amount"))
        .aggregate(QwakAggregation.sample_stdev("transaction_amount"))
        .aggregate(QwakAggregation.last_n("transaction_amount", 5))
        .aggregate(QwakAggregation.last_distinct_n("transaction_amount", 5))
        .aggregate(QwakAggregation.percentile("transaction_amount", 50)
                   .alias("median_transaction_amount"))
        .by_windows("1 minute", "15 minutes", "30 minutes", "1 hour"),

    timestamp_col_name="timestamp")

# count_transaction_amount_1m             bigint
# last_distinct_5_transaction_amount_1m   array<int>
# sum_transaction_amount_1m               bigint
# sample_stdev_transaction_amount_1m      double
# median_transaction_amount_1m            int
# last_5_transaction_amount_1m            array<int>
# max_transaction_amount_1m               int
# count_transaction_amount_30m            bigint
# last_distinct_5_transaction_amount_30m  array<int>
# sum_transaction_amount_30m              bigint
# sample_stdev_transaction_amount_30m     double
# median_transaction_amount_30m           int
# last_5_transaction_amount_30m           array<int>
# max_transaction_amount_30m              int
# count_transaction_amount_15m            bigint
# last_distinct_5_transaction_amount_15m  array<int>
# sum_transaction_amount_15m              bigint
# sample_stdev_transaction_amount_15m     double
# median_transaction_amount_15m           int
# last_5_transaction_amount_15m           array<int>
# max_transaction_amount_15m              int
# count_transaction_amount_1h             bigint
# last_distinct_5_transaction_amount_1h   array<int>
# sum_transaction_amount_1h               bigint
# sample_stdev_transaction_amount_1h      double
# median_transaction_amount_1h            int
# last_5_transaction_amount_1h            array<int>
# max_transaction_amount_1h               int

if __name__ == "__main__":
    df = streaming_fs.get_sample()
    print(df)