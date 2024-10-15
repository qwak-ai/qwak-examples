from qwak.model.tools import run_local
from main import *

if __name__ == '__main__':
    # Create a new instance of the model from __init__.py
    m = load_model()  # Load the pre-trained model

    # Specify the path to the WAV audio file to be processed
    wav_file_path = "harvard.wav"

    # Open the specified WAV file as a binary stream
    with open(wav_file_path, 'rb') as wav_file_stream:

        # call the model using the local testing toolkit
        result = run_local(m, [wav_file_stream])

    # Print the transcription result returned by the model
    print(result)



