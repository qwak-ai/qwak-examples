
from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from qwak.feature_store.data_sources import KafkaSource, CustomDeserializer
from pyspark.sql.functions import col, from_json
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

# replace with your own Kafka brokers
bootstrap_servers = 'b-2.qwak-cluster.za36zh.c6.kafka.us-east-1.amazonaws.com:9094,b-1.qwak-cluster.za36zh.c6.kafka.us-east-1.amazonaws.com:9094,b-3.qwak-cluster.za36zh.c6.kafka.us-east-1.amazonaws.com:9094'

# replace with your own Kafka passthrough configurations
passthrough_configs = {
    "qwak.online.maxOffsetsPerTrigger": "200000",
    "startingOffsets": "earliest",
}


kafka_source = KafkaSource(
    # metadata name of the DataSource within Qwak
    name="transactions",
    # metadata description of the DataSource within Qwak
    description="Transaction Test Source",
    # Bootstrap servers of the Kafka client to be connected to
    bootstrap_servers=bootstrap_servers,
    # Kafka topic to be ingested
    subscribe="transactions",
    # Deserialization function described above
    deserialization=deserializer,
    # Passthrough Kafka configs for offset settings, defined above
    passthrough_configs=passthrough_configs,
)
