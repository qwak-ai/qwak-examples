from qwak.applications import QwakApplication, step
from qwak.clients.batch_job_management import BatchJobManagerClient
from qwak.clients.batch_job_management.executions_config import ExecutionConfig


class BatchModelsChainingApp(QwakApplication):
    """
    1. Run the batch model "batch_churn_model"
    2. Wait for the batch model to finish
    3. Run the batch model "batch_summary_model"

    Open questions
    - Support to file-based or df-based?
    - Pass the df as a param
    """

    @step
    def start(self, execution_params):
        self.next(self.run_initial_batch_model,
                  params=execution_params)

    @step
    def run_initial_batch_model(self, params):
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
        self.next(self.run_summarizing_batch_model,
                  execution_params={
                      "execution_spec": execution_spec,
                      "execution_result": execution_result
                  }
        )

        # 2. Use a native command in the Application
        self.execute_batch_model(execution=execution_spec,
                                 on_complete=self.run_summarizing_batch_model,
                                 execution_params={
                                     "execution_spec": execution_spec,
                                     "execution_result": execution_result
                                 })


    @step
    def run_summarizing_batch_model(self, execution_params):
        execution_spec = ExecutionConfig.Execution(
            model_id="batch_summary_model",
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
        self.next(self.end,
                  execution_params={
                      "execution_spec": execution_spec,
                      "execution_result": execution_result
                  }
        )

        # 2. Use a native command in the Application
        self.execute_batch_model(execution=execution_spec,
                                 on_complete=self.end,
                                 execution_params={
                                     "execution_spec": execution_spec,
                                     "execution_result": execution_result
                                 })


if __name__ == '__main__':
    flow = BatchModelsChainingApp()
    flow.execute(schedule="0 * * * *")
    flow.execute_now()

    # Open questions
    # 1. How to perform DataFrame based batch models - pass as params?

    with QwakApplication("fs-batch-model") as app:
        qwak  = BatchModelInference(name="user-credit-risk-features")

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
        model = BatchModelInference(config=execution_config)