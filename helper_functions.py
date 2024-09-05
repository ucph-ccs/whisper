from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio
import pandas as pd
import configparser

config_data = configparser.ConfigParser()

config_data.read('config.ini')


def load_model():
    # Load model and processor
    processor = WhisperProcessor.from_pretrained(config_data.get('Model', 'model'))
    model = WhisperForConditionalGeneration.from_pretrained(config_data.get('Model', 'model'))
    # Set the language and task
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=(config_data.get('Model', 'language')),
                                                          task=config_data.get('Model', 'task'))

    return model, processor, forced_decoder_ids


def process_audio_file(model, processor, forced_decoder_ids):
    # Initialize an empty DataFrame
    audio_path = config_data.get('Model', 'audio_path')
    speech_array, sampling_rate = torchaudio.load(audio_path)

    # If necessary, resample the audio to 16000 Hz
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        speech_array = resampler(speech_array)
        sampling_rate = 16000

    # Prepare the input features
    input_features_ = processor(speech_array.squeeze().numpy(),
                                sampling_rate=sampling_rate, return_tensors="pt").input_features

    # Generate token ids
    with torch.no_grad():  # Disable gradient computation
        predicted_ids = model.generate(input_features_, forced_decoder_ids=forced_decoder_ids)

    # Decode token ids to text
    prediction = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    # Print the transcription
    print(prediction)
    return prediction
