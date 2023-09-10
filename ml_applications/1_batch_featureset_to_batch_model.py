from qwak import QwakClient
from qwak.applications import QwakApplication, step
from qwak.clients.batch_job_management import BatchJobManagerClient
from qwak.clients.batch_job_management.executions_config import ExecutionConfig


class BatchFeaturesAndBatchInferenceApp(QwakApplication):
    """
    1. Run the batch feature set "user-credit-risk-features"
    2. Wait for the batch feature set to finish
    3. Run the batch model "batch_churn_model"

    Open questions:
    - How to point the model to fresh data from the feature store?

    """
    @step
    def start(self):
        self.next(self.run_batch_feature_ingestion)

    @step
    def run_batch_feature_ingestion(self):
        # 1. Use the QwakClient and wait for completion
        client = QwakClient()
        client.trigger_batch_feature_set("user-credit-risk-features")
        self.next(self.perform_batch_feature_ingestion)

        # 2. Have the application perform it natively
        self.execute_batch_feature_set("user-features",
                                       on_complete=self.run_batch_model_inference,
                                       execution_params={})

    @step
    def run_batch_model_inference(self, execution_params):
        execution_spec = ExecutionConfig.Execution(
            model_id="batch_churn_model",
            access_token_name="api-token",
            access_secret_name="api-secret",
            source_bucket="input_s3_bucket",
            source_folder="data_folder",
            input_file_type="csv",
            destination_bucket="output_s3_bucket",
            destination_folder="output_data_folder",
            output_file_type="csv",
        )
        execution_config = ExecutionConfig(execution=execution_spec)

        # 1. Use the BatchJobManagerClient and wait for completion
        batch_job_manager_client = BatchJobManagerClient()
        execution_result = batch_job_manager_client.start_execution(execution_config)
        self.next(self.end)

        # 2. Use a native command in the Application
        self.execute_batch_model(execution=execution_spec,
                                 on_complete=self.end,
                                 execution_params={})


if __name__ == '__main__':
    flow = BatchFeaturesAndBatchInferenceApp()
    flow.execute(schedule="0 * * * *")
    flow.execute_now()
