from qwak import QwakClient
from qwak.clients.batch_job_management import ExecutionConfig
from qwak.applications import QwakApplication
from qwak.applications.operators import BatchFeatureSetOperator

def run_fs_model_application():

    client = QwakClient()
    with QwakApplication("batch-fs-to-batch-model", schedule="daily") as app:

        # Get all relevant Batch FS we want to run (in parallel)
        batch_fs = BatchFeatureSetOperator(name="user-conversion-features", app=app)

        execution_config = ExecutionConfig(
            execution=ExecutionConfig.Execution(
                model_id="conversion_to_paid_probability"
            )
        )

        model = BatchModelInferenceOperator(config=execution_config,
                                            app=app,
                                            batch_size=100,
                                            # Define BigQuery / File input and output
                                            )

        # Run all the fs_jobs and then the model
        batch_fs >> model


if __name__ == '__main__':
    run_fs_model_application()

