# Model Predcition
def predict_on_entire_file(audio, sample_rate):

    # Duration of file
    file_duration = len(audio)/sample_rate

    # Number of segments
    segments = math.floor(file_duration) - 2

    # Store predictions in this list
    predictions = []

    # Loop over the file and work in small "segments"
    for position in range (0, segments):

        # Determine start of segment
        start_position = position

        # Determine end of segment
        end_position = start_position + 2

        print('start position:', start_position)
        print('end position:', end_position)

        # Extract a 2 second segment from the audio file
        audio_segment = audio[start_position*librosa_sample_rate:end_position*librosa_sample_rate]

        # Create the spectrogram
        S = librosa.feature.melspectrogram(y=audio_segment, sr=sample_rate, n_mels=128)

        # Input spectrogram into model
        softmax = model.predict(np.reshape(S, (1,128,188,1)))

        print ('model output:', softmax)

        # Binary output
        binary_prediction = np.argmax(softmax,-1)

        print ('binary output:', binary_prediction[0])

        # Append result
        predictions.append('absence' if np.argmax(softmax,-1)[0]== 0 else 'presence')

        print()

    return predictions