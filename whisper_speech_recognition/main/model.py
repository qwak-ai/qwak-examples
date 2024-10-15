import qwak # The API decorator
from qwak.model.base import QwakModel  # Base class for Qwak models
from qwak.model.schema import ModelSchema, InferenceOutput  # Schema definitions
from qwak.model.adapters import FileInputAdapter, JsonOutputAdapter  # Input/Output adapters for handling data
from transformers import pipeline
import torch
import numpy as np
import wave
import os
import io

# Define the WhisperModel class, which inherits from QwakModel
class WhisperModel(QwakModel):

    # Initialize model attributes
    def __init__(self):
        self.whisper_model = os.environ.get('WHISPER_MODEL', 'large-v2')
        self.model_id = f"openai/whisper-{self.whisper_model}"  # Pre-trained Whisper model ID for speech recognition
        self.pipeline = None  # Placeholder for the ASR model pipeline

    # Build method (currently a placeholder)
    def build(self):
        pass  # No specific building logic needed

    # Define the schema for the model's input and output
    def schema(self):
        model_schema = ModelSchema(
            outputs=[
                InferenceOutput(name="text", type=str)  # Expected output type
            ])
        return model_schema  # Return the model schema

    # Initialize the ASR model and its tokenizer
    def initialize_model(self):

        # Check if CUDA (GPU) is available
        device = 0 if torch.cuda.is_available() else -1

        if torch.cuda.is_available():
            print("GPU is available, model will run on:", torch.cuda.get_device_name(0))
        else:
            print("GPU not available, running on CPU")

        # Create the automatic speech recognition pipeline using the specified model
        self.asr_pipeline = pipeline("automatic-speech-recognition", 
                                     model=self.model_id,
                                     device=device)

    # Define the prediction method
    @qwak.api(input_adapter=FileInputAdapter(),  # Adapter for converting input to NumPy arrays
              output_adapter=JsonOutputAdapter())  # Adapter for converting output to JSON format
    def predict(self, file_streams):
        transcriptions = []  # List to store transcription results

        # Process each audio input stream
        for _file_stream in file_streams:

            file_as_bytes = _file_stream.read()
            
            print(f"Received file size: {len(file_as_bytes)} bytes")

            with io.BytesIO(file_as_bytes) as wrapped_file:
                
                data = self.read_wav_data(wrapped_file)

                # Append transcription results to the list
                transcriptions.append(self.asr_pipeline(data))

        return transcriptions  # Return the list of transcriptions
    
    # Define a utilitary method to read the relevant audio data from a file like object and return it as Numpy.ndarray which is what Whisper expects.
    def read_wav_data(self, file_stream):
        # Open the file stream as a WAV file
        with wave.open(file_stream, 'rb') as wav_file:
            # Read all the frames
            frames = wav_file.readframes(wav_file.getnframes())
            
            # Convert frames to a NumPy array (assumes 16-bit PCM)
            data = np.frombuffer(frames, dtype=np.int16)
            
        return data