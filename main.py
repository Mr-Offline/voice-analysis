# Open the WAV file
import wave
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

# Open the WAV file
with wave.open('./Recording.wav', 'rb') as wav_file:
    # Get the number of frames in the WAV file
    num_frames = wav_file.getnframes()

    # Get the frame rate of the WAV file
    frame_rate = wav_file.getframerate()

    # Set the frame size (in seconds)
    frame_size = 0.02

    # Calculate the number of frames per frame size
    frames_per_frame_size = int(frame_rate * frame_size)

    # Calculate the number of frames in all but the last frame
    num_frames_all_but_last_frame = num_frames - frames_per_frame_size

    # Initialize an array to store the frames
    frames_array = []

    # Iterate over each frame size
    for i in range(0, num_frames_all_but_last_frame, frames_per_frame_size):
        # Read the frames for the current frame size
        frames = wav_file.readframes(frames_per_frame_size)
        wav_file.setpos(wav_file.tell() - (int(frames_per_frame_size * 0.01)))

        read_frames = np.frombuffer(frames, dtype=np.int16)
        hamming_frame = signal.windows.hamming(len(read_frames))
        read_frames = read_frames * hamming_frame

        # Append the frames to the frame array
        frames_array.append(read_frames)

fig, ax = plt.subplots(1, 4)

energy_array = []
for frames in frames_array:
    energy_array.append(np.sum([frame**2 for frame in frames]) / len(frames))

ax[0].set_title("Energy")
ax[0].plot([i * 0.02 for i in range(1, len(energy_array) + 1)], energy_array, linewidth=2.0)

zcr_array = []
for frames in frames_array:
    zcr_array.append((np.sum(np.abs(np.diff(np.sign(frames))))) / (2 * len(frames)))

ax[1].set_title("ZCR")
ax[1].plot([i * 0.02 for i in range(1, len(zcr_array) + 1)], zcr_array, linewidth=2.0)

fft_frames = []
for frames in frames_array:
    fft_frames.append(np.mean(np.abs(fft(frames)[:len(frames) // 2])))

ax[2].set_title("FFT")
ax[2].plot(fft_frames, fftfreq(len(fft_frames), d=0.02), linewidth=2.0)

autocorolation_array = []
for frames in frames_array:
    autocorolation_array.append(np.correlate(frames, frames))

ax[3].set_title("Autocorolation")
ax[3].plot(autocorolation_array, linewidth=2.0)


plt.show()
