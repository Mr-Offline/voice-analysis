# Open the WAV file
import wave
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq, ifft
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
        read_frames = read_frames.reshape((-1, 2))
        read_frames = np.mean(read_frames, axis=1).astype(np.int16)

        hamming_frame = signal.windows.hamming(len(read_frames))
        read_frames = read_frames * hamming_frame

        # Append the frames to the frame array
        frames_array.append(read_frames)

# fig, ax = plt.subplots(1, 4)

fig4, ax4 = plt.subplots(1, 2)

energy_array = []
for frames in frames_array:
    energy_array.append(np.sum([frame**2 for frame in frames]) / len(frames))

ax4[0].set_title("Energy")
ax4[0].plot([i * 0.02 for i in range(1, len(energy_array) + 1)], energy_array, linewidth=2.0)

zcr_array = []
for frames in frames_array:
    zcr_array.append((np.sum(np.abs(np.diff(np.sign(frames))))) / (2 * len(frames)))

ax4[1].set_title("ZCR")
ax4[1].plot([i * 0.02 for i in range(1, len(zcr_array) + 1)], zcr_array, linewidth=2.0)

# fft_frames = []
# for frames in frames_array:
#     fft_frames.append(np.abs(fft(frames)[:len(frames) // 2]))

# ax[2].set_title("FFT")
# ax[2].plot(fft_frames, fftfreq(len(fft_frames), d=0.02), linewidth=2.0)

# autocorolation_array = []
# for frames in frames_array:
#     autocorolation_array.append(np.correlate(frames, frames))

# ax[3].set_title("Autocorolation")
# ax[3].plot(autocorolation_array, linewidth=2.0)

fig, ax = plt.subplots(1, 2)

fig1, ax1 = plt.subplots()

fig2, ax2 = plt.subplots()

fig3, ax3 = plt.subplots()

selected_frame_number = 39 # 0.78s
selected_frame = frames_array[selected_frame_number]

ax[0].set_title("Signal")
ax[0].plot(selected_frame, linewidth=2.0)

print(f"Selected frame energy: {sum([data ** 2 for data in selected_frame]) / len(selected_frame)}")

print(f"Selected frame ZCR: {sum(abs(np.diff(np.sign(selected_frame))) / 2) / len(selected_frame)}")

ax[1].set_title("Autocorolation")
ax[1].plot(np.correlate(selected_frame, selected_frame, mode='full'), linewidth=2.0)

ax1.grid(True)
ax1.set_title("FFT")
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Magnitude (dB)")
ax1.plot(10 * np.log10(np.abs(fft(selected_frame)[:len(selected_frame // 2)])), linewidth=2.0)

# cepstral
cepstral = np.real(ifft(np.log10(np.abs(fft(selected_frame)))))
ax2.set_title("Cepstral")
ax2.plot(cepstral, linewidth=2.0)

# amdf
amdf = [np.sum(np.abs(selected_frame - np.roll(selected_frame, i))) / len(selected_frame) for i in range(len(selected_frame))]
ax3.set_title("AMDF")
ax3.plot(amdf, linewidth=2.0)

plt.show()
