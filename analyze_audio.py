import configparser
from helper_functions import load_model, process_audio_file

config = configparser.ConfigParser()
config.read('config.ini')

# Load the model
model, processor, forced_decoder_ids = load_model()
# Process the audio file
process_audio_file(model, processor, forced_decoder_ids)
