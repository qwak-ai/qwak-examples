from qwak.clients.batch_job_management import ExecutionConfig
from qwak.applications import QwakApplication
from qwak.applications.operators import BatchFeatureSetOperator, BatchModelInferenceOperator, ModelBuildOperator


def run():

    with QwakApplication("batch-fs-to-batch-model", schedule="daily") as app:
        model_id = "conversion_to_paid_probability"
        feature_set_id = "user-conversion-features"

        # Setup an operator to run a batch feature set
        batch_fs = BatchFeatureSetOperator(
            name=feature_set_id,
            app=app
        )

        # Building a new model
        model_build = ModelBuildOperator(
            model_id=model_id,
            instance="medium"
        )

        # Running batch model inference
        # We need to take the latest model build for the model inference
        execution_config = ExecutionConfig(
            execution=ExecutionConfig.Execution(
                model_id=model_id,
                build_id=model_build.build_id   # if we cannot use this, we'll use the QwakClient
            )
        )
        batch_model = BatchModelInferenceOperator(
            config=execution_config,
            app=app,
            batch_size=100,
            # Define BigQuery / File input and output
        )

        # Run all operators sequentially
        batch_fs >> model_build >> batch_model


if __name__ == '__main__':
    run()

