from pandas import DataFrame
from qwak import QwakClient
import pandas as pd
import snowflake.connector as sf
from qwak.clients.batch_job_management import ExecutionConfig

from airflow.models.baseoperator import chain
from qwak.applications import QwakApplication

# QWAK_FS_DB_NAME = "qwak_feature_store_0000e982_76c8_4a95_851b_8f060d793242"
# SLEEP_BETWEEN_INGESTION_POLLS = 30
# MAX_TIMEOUT_TIME = SLEEP_BETWEEN_INGESTION_POLLS * 2 * 60 * 2 # 1 minute (30 seconds * 2) * 60 (1 hour) * 2 == 2 hours timeout per job

# def get_fs_max_processing_time(client: QwakClient, fs_name: str):
#     try:
#         return client.run_analytics_query(query=f"""
#             SELECT MAX("END_TIMESTAMP") as max_end_time
#             FROM "{QWAK_FS_DB_NAME}"."offline_feature_store_{fs_name.replace("-", "_")}"
#             """)['max_end_time'][0]
#     except Exception as e:
#         print(f"Failed to fetch max processing time for feature set '{fs_name}', error is {e}")
#         exit(1)


# def trigger_batch_ingestions(client: QwakClient, fss_names: List[str]):
#     fs_initial_update_times = dict()
#
#     # Trigger all batch ingestion in parallel - as `trigger_batch_feature_set` is non-blocking
#     for fs_name in fss_names:
#         fs_initial_update_times[fs_name] = get_fs_max_processing_time(client, fs_name)
#         client.trigger_batch_feature_set(feature_set_name=fs_name)
#
#     # Track
#     for fs_name in fss_names:
#         ingestion_polling_time = 0
#         while fs_initial_update_times[fs_name] == get_fs_max_processing_time(client, fs_name) \
#                 and ingestion_polling_time < MAX_TIMEOUT_TIME:
#             ingestion_polling_time += SLEEP_BETWEEN_INGESTION_POLLS
#             sleep(SLEEP_BETWEEN_INGESTION_POLLS)
#         print(f"{fs_name} has updated!")


def get_data_from_snowflake() -> DataFrame:
    # Fetch population from Snowflake
    conn = sf.connect(user='qwakro',
                      password="YallabalaganQ2022!",
                      account="ura77389.us-east-1",
                      warehouse="LILI_ANALYTICS_QWAK",
                      database="DSG_SANDBOX",
                      schema="OMERA",
                      role="DSG")

    pop = """SELECT distinct bank_account_id, 'batch' as request_type
                 FROM lili_analytics.ods.mysql_account limit 50"""

    df = pd.read_sql(pop, conn)
    df.columns = df.columns.str.lower()

    return df


def run_fs_model_application():

    client = QwakClient()
    with QwakApplication("batch-fs-to-batch-model", schedule="daily") as app:

        fs_jobs = []
        fss_names = [fs.name for fs in client.list_feature_sets()]
        for fs_name in fss_names:
            fs_jobs.append(
                BatchFeatureSetOperator(name=fs_name, app=app)
            )
        chain(*fs_jobs)

        execution_config = ExecutionConfig(
            execution=ExecutionConfig.Execution(
                model_id="hold_days_pilot_batch"
            )
        )


        model = BatchModelInferenceOperator(config=execution_config,
                                            app=app,
                                            batch_size=100,
                                            data=QueryOperator(data=get_data_from_snowflake)
                                            )

        # Run all the fs_jobs and then the model
        fs_jobs[-1] >> model


if __name__ == '__main__':
    run_fs_model_application()

