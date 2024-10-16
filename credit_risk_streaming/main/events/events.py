from pyspark.sql import SparkSession, DataFrame

import pyspark.sql.functions as F


def get_case_condition():
    entities = ["e41160de-0a56-47cf-8193-a0c97fe2e752",
                "b0ca3ac4-5432-4c21-8251-a6ae0d3ad874",
                "4b7af572-b249-4bae-9815-10ed3a2cd01d",
                "d3b985e5-2309-48c6-9453-865aa6b26108",
                "b3c16619-be7f-4deb-a9cc-b99e94de89fb",
                "ea4882fa-8af3-4aa3-a791-8b3c159c2cf0",
                "a3a2eaf9-5d8f-46dd-9cf4-9d186c973200",
                "a8cd57ca-0199-46e3-a44e-a9d5927edd7a",
                "8b011e21-bc59-4d8f-857d-2686dc91cb6a",
                "95ec0c53-4e27-4490-b85f-1448de70fc26",
                "a591a687-65f6-4c36-841a-2d965816f880",
                "162d2517-ce7a-46aa-8e51-3e38a1de5d22",
                "25a06f81-7ab0-4e3e-bb2a-99b2be9d29ad",
                "5e045817-d7b9-4c98-b01b-c992e79624a4",
                "5f97899e-de8a-4527-b48b-633ec77dc722",
                "66025b0f-6a7f-4f86-9666-6622be82d870",
                "742f09a6-b88b-4518-aa7b-431aa8ae8a16",
                "924f9868-af17-45af-831e-6c5f8b25b5ed",
                "a8af33ad-23b9-48f9-8fe2-8b3d1a0a6829",
                "b234d555-116b-4deb-b6f4-79a9753eb0a1",
                "1831d2b0-df5a-40aa-aa43-ee26eaa7568c",
                "a9350cff-8321-4f09-b3b4-4bb4df5951ac",
                "149ca9e4-0e4a-48e8-b9e2-263b23d371c0",
                "46e361c2-9c09-4bea-bfa6-178b80bc5aec",
                "46ad9e4b-1d0f-47b7-a73d-71cc66538b03",
                "52ef7f25-6db2-47d0-9476-88bb8e6fa605",
                "f61e0322-4d40-441d-8a18-cb98b3b4730c",
                "47da77d0-3151-4965-a1cf-7eb7234a01ad",
                "6871475c-eb71-4457-bd11-9aedea791ac8",
                "d5eef17c-b25b-4c47-b29c-f02a5f1bd0ef",
                "ca2da30a-ffde-4334-87e5-7dae3bebb5db",
                "02d7f29f-90b3-47f2-9952-126bdd18c378",
                "48d601e6-72b5-457c-b372-a1d73a80a02e",
                "5abb1d49-a1aa-4f2a-92a2-6f537ec7421d",
                "5d1d338c-4ac9-447b-b388-823951ede8f6",
                "72878824-25ec-4a5b-a4db-6ed18009691d",
                "89161639-e300-4789-afee-d675cfa383e1",
                "455d686b-d8f2-4e2c-aa6b-851e9b3de446",
                "9fbc0143-ad6a-4a5a-b52e-693f0b4f1add",
                "15b6edce-9b91-4c1f-aae9-2247d0d1a427",
                "181153df-581b-4cf9-9649-aee2eedb25d5",
                "73142300-493b-45f5-b355-3dadd54b0c13",
                "ee34b414-1632-4491-8c35-a93ca8fe07a0",
                "8410fd1f-4f76-4375-8125-df3fe383cc60",
                "aa4f0106-bea0-45a8-b50f-9b06e0dc629a",
                "35343bfd-f15d-48f7-9ebf-f0f724dbb2a9",
                "5a114060-1038-4dc6-a038-3b7423cf2c16",
                "2c39a950-04e0-43d5-bbba-bea628309b0c",
                "84fb62bb-8097-4c1f-b7dd-33e0c06ab3d3",
                "83d78248-9ee8-48a2-864a-2d8b134dda89"]
    pred = F
    for idx, user_id in enumerate(entities):
        pred = pred.when(F.expr(f"pmod(value, {len(entities)})").eqNullSafe(F.lit(idx)), F.lit(user_id))

    pred = pred.otherwise("UNKNOWN")

    return pred.alias("user_id")


if __name__ == "__main__":
    ## Create father Spark
    spark: SparkSession = SparkSession.builder \
        .appName("father") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.1") \
        .config("spark.sql.streaming.metricsEnabled", "true") \
        .getOrCreate()

    ## Use "rates" to fill the kafka topic
    rates = spark.readStream.format("rate").option("rowsPerSecond", 50).load()

    rates.createOrReplaceTempView("rates")


    with_key: DataFrame = rates.select(F.col("timestamp"),
                                       get_case_condition(),
                                       F.col("value").alias("transaction_amount"))

    with_key.createOrReplaceTempView("shlomke")
    after_func = spark.sql(
        "select cast(user_id as string) as key, to_json(struct(*)) as value from shlomke"
    )

    bootstrap_servers = 'b-2.qwak-cluster.za36zh.c6.kafka.us-east-1.amazonaws.com:9094,b-1.qwak-cluster.za36zh.c6.kafka.us-east-1.amazonaws.com:9094,b-3.qwak-cluster.za36zh.c6.kafka.us-east-1.amazonaws.com:9094'
    query = (
        after_func.writeStream.outputMode("append")
            .format("kafka")
            .option(
            "kafka.bootstrap.servers", bootstrap_servers
        )
            .option("topic", "transactions")
            .option("kafka.security.protocol", "SSL")
            .option("truncate", "false")
            .option("checkpointLocation", "/tmp/demo_checkpoint")
            .start()
    )
    query.awaitTermination()