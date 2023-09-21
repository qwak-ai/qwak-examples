import pandas as pd
import snowflake.connector as sf
from airflow.models.baseoperator import chain
from pandas import DataFrame
from qwak import QwakClient
from qwak.applications import QwakApplication
from qwak.applications.operators import BatchFeatureSetOperator, BatchModelInferenceOperator, DataSourceOperator
from qwak.clients.batch_job_management import ExecutionConfig


def get_data_from_snowflake() -> DataFrame:
    conn = sf.connect(user='user',
                      password="password",
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


def run():

    client = QwakClient()
    with QwakApplication("batch-fs-to-batch-model", schedule="daily") as app:
        # Running all feature sets
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

        query_data = DataSourceOperator(data=get_data_from_snowflake)

        model = BatchModelInferenceOperator(config=execution_config,
                                            app=app,
                                            batch_size=100,
                                            data=query_data.data  # How to pass data from a query operator
                                            )

        # Run all the fs_jobs and then the model
        fs_jobs[-1] >> query_data >> model


if __name__ == '__main__':
    run()
