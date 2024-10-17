import os
from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from pyspark.sql.functions import col, from_json
from qwak.feature_store.data_sources import KafkaSource, CustomDeserializer

from qwak.feature_store.feature_sets import streaming
from qwak.feature_store.feature_sets.transformations import SparkSqlTransformation, QwakAggregation


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


deserializer = CustomDeserializer(function=deser_function)
bootstrap_servers = 'b-2.qwak-cluster.za36zh.c6.kafka.us-east-1.amazonaws.com:9094,b-1.qwak-cluster.za36zh.c6.kafka.us-east-1.amazonaws.com:9094,b-3.qwak-cluster.za36zh.c6.kafka.us-east-1.amazonaws.com:9094'

passthrough_configs = {
    "qwak.online.maxOffsetsPerTrigger": "200000",
    "startingOffsets": "earliest",
}

kafka_source = KafkaSource(
    name="transactions",
    description="Transaction Test Source",
    bootstrap_servers=bootstrap_servers,
    subscribe="transactions",
    deserialization=deserializer,
    passthrough_configs=passthrough_configs,
)


@streaming.feature_set(
    name="credit-risk-streaming",
    data_sources=["transactions"],
    key="user_id",
    timestamp_column_name="timestamp")
@streaming.metadata(
        display_name="Credit Risk Streaming",
        description="streaming transaction aggregations over 1,15,30,60 minutes",
        owner="hudson@qwak.com"
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
