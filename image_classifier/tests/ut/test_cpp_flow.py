# test_cpp.py
import pytest
import subprocess
import os
import frogml
import torch

MODEL_ID = os.environ.get("QWAK_MODEL_ID")
#BUILD_ID = os.environ.get("QWAK_BUILD_ID")
BUILD_ID = '62f8e2b3-2c58-4f3a-a543-4648c792a7a3'
REPO = "cv-models"
MODEL_PATH = f"{MODEL_ID}-{BUILD_ID}.pth"  # Corrected model path

@pytest.fixture(scope="session")
def downloaded_model():
    """Downloads the model from Artifactory and returns its path."""
    if not MODEL_ID or not BUILD_ID:
        pytest.fail("QWAK_MODEL_ID and QWAK_BUILD_ID environment variables must be set.")

    #download_command = f"jfrog rt dl {REPO}/{MODEL_ID}/{BUILD_ID}/{MODEL_PATH} ."

    

    try:
        #subprocess.run(download_command, shell=True, check=True, capture_output=True, text=True)
        _model = frogml.pytorch.load_model(
                repository = REPO,
                model_name = MODEL_ID,
                version = BUILD_ID
            )
    
        torch.save(_model, MODEL_PATH)
        print(f"Model downloaded successfully.")
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Failed to download model: {e}")
    return MODEL_PATH  # Return the path to the downloaded model


def test_cpp_model_size(downloaded_model, qwak_tests_additional_dependencies):
    """Tests the size of the model using the C++ program."""
    try:
        # Attempt to change permissions.  Requires execute permissions on chmod itself
        result = subprocess.run(['chmod', '+x', f"{qwak_tests_additional_dependencies}embedded_code/test_model"], capture_output=True, text=True, check=True)
        print(f"chmod output: {result.stdout}")  # Print output to see if it worked
        print(f"chmod errors: {result.stderr}")

        result = subprocess.run([f"{qwak_tests_additional_dependencies}embedded_code/test_model", downloaded_model], capture_output=True, text=True, check=True)
        output = result.stdout
        print(f"C++ test output: {output}")

        assert "TEST_PASSED" in output, f"C++ test failed. Output: {output}"
        # Optionally, you can parse the output for the model size and assert against a specific value:
        # if "Model file size:" in output:
        #     size_line = [line for line in output.splitlines() if "Model file size:" in line][0]
        #     size_str = size_line.split(":")[1].strip().split(" ")[0] # Extract size
        #     size = int(size_str)
        #     assert size > 1024, "Model size is too small"


    except subprocess.CalledProcessError as e:
        pytest.fail(f"C++ test failed: {e.stderr}")
