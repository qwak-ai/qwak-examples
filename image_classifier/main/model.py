import numpy as np
import pandas as pd
from frogml import FrogMlModel
import frogml
from torchvision import models, transforms
import torch
from frogml_core.model.adapters import NumpyInputAdapter
import json
import os
from frogml_core.tools.logger import get_frogml_logger

logger = get_frogml_logger()

REPOSITORY = 'cv-models'

class ImageClassifier(FrogMlModel):

    def __init__(self):
        self._model = None
        self._preprocess = None
        self._categories = None

    def _log_model_to_jfrog(self):
        """Logs a model to the JFrog repository, with error handling and logging."""
        try:
            frogml.pytorch.log_model(
                model = self._model,
                model_name=os.environ['QWAK_MODEL_ID'],
                repository=REPOSITORY,
                version=os.environ["QWAK_BUILD_ID"]
            )
            logger.info(f"Successfully logged model '{os.environ['QWAK_MODEL_ID']}' to JFrog repository '{REPOSITORY}'.")

        except Exception as e:
            logger.error(f"An error occurred while logging the model to JFrog: {e}")
            raise

    def _load_model_from_jfrog(self, model_version):
        """Loads a model from a JFrog repository, with error handling and logging."""
        try:
            self._model = frogml.pytorch.load_model(
                repository=REPOSITORY,
                model_name=os.environ['QWAK_MODEL_ID'],
                version=model_version
            )
            logger.info(f"Successfully loaded FINETUNED model '{os.environ['QWAK_MODEL_ID']}' version '{model_version}' from JFrog repository '{REPOSITORY}'.")

        except Exception as e:
            logger.exception(f"An error occurred while loading model version '{model_version}' from JFrog: {e}")
            raise  # Re-raise the exception to signal a critical error

        return self._model

    def _quantize_model(self, model, backend="fbgemm"):
        """Quantizes a PyTorch model for embedded deployment."""
        model.eval()

        # Check if fbgemm is available
        if torch.backends.quantized.engine != 'fbgemm':
            logger.warning("fbgemm is NOT available.  Falling back to CPU.")
            logger.warning(f"Available backends: {torch.backends.quantized.supported_engines}")
            return model  # Return the original model

        # Fuse layers for better performance
        try:
            model = torch.quantization.fuse_modules(model, ['features.0', 'features.1'], inplace=True)
            logger.info("Successfully fused layers.")
        except Exception as e:
            logger.warning(f"Failed to fuse layers: {e}")

        # Specify quantization configuration
        model.qconfig = torch.quantization.get_default_qconfig(backend)  # Use fbgemm for x86
        logger.info(f"Using quantization configuration: {model.qconfig}")

        # Prepare the model for quantization
        try:
            torch.quantization.prepare(model, inplace=True)
            logger.info("Successfully prepared model for quantization.")
        except Exception as e:
            logger.error(f"Failed to prepare model for quantization: {e}")
            return model

        # Calibrate the model (run it on a representative dataset)
        try:
            sample_data = torch.randn(1, 3, 224, 224)
            model(sample_data)
            logger.info("Successfully calibrated model.")
        except Exception as e:
            logger.error(f"Failed to calibrate model: {e}")
            return model

        # Convert the model to a quantized version
        try:
            model_quantized = torch.quantization.convert(model, inplace=True)
            logger.info("Successfully converted model to quantized version.")
            return model_quantized
        except Exception as e:
            logger.error(f"Failed to convert model to quantized version: {e}")
            return model

    def build(self):
        # Load the pre-trained ResNet50 model from PyTorch
        #self._model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self._model = models.mobilenet_v2(pretrained=True)
        self._model = self._quantize_model(self._model)  # Use fbgemm for x86
        self._model.eval()  # Set the model to evaluation mode

        # Define preprocess pipeline using Torch's transforms
        self._preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load the ImageNet class labels
        with open('main/imagenet_classes.txt') as f:
            self._categories = [line.strip() for line in f.readlines()]

        self._log_model_to_jfrog()

    def initialize_model(self):
        pass

    def _preprocess_image(self, img_ndarray):
        # Check if img_ndarray is a string and convert to ndarray
        if isinstance(img_ndarray, str):
            img_list = json.loads(img_ndarray)  # Convert JSON string to Python list
            img_ndarray = np.array(img_list)    # Convert list to ndarray
        # Note: If img_ndarray is already an ndarray, this part will be skipped
        # Ensure correct format and type
        if img_ndarray.ndim == 2:
            img_ndarray = np.stack((img_ndarray,)*3, axis=-1)
        if img_ndarray.dtype != np.float32:
            img_ndarray = img_ndarray.astype('float32')
        # Preprocess the image using Torch's transforms
        img_tensor = self._preprocess(img_ndarray)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

        return img_tensor
    

    @frogml.api(input_adapter=NumpyInputAdapter())
    def predict(self, img_ndarray: np.ndarray) -> pd.DataFrame:
        # Preprocess the image and make predictions
        preprocessed_image = self._preprocess_image(img_ndarray[0])  # assuming the adapter converts it to an ndarray
        with torch.no_grad():
            predictions = self._model(preprocessed_image)
        
        # Decode predictions (The function to decode PyTorch predictions can differ)
        probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, 3)
        
        decoded_predictions = [(self._categories[top_catid[i]], top_prob[i].item()) for i in range(top_prob.size(0))]
        top_prediction = pd.DataFrame(decoded_predictions, columns=['name', 'probability'])
        
        # Return the top prediction in a DataFrame
        return top_prediction
    
    