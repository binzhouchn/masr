# 录音模块sounddevice或者pyaudio
# pyaudio版本
import pyaudio
import wave

framerate = 16000
NUM_SAMPLES = 2000
channels = 1
sampwidth = 2
TIME = 10


def save_wave_file(filename, data):
    wf = wave.open(filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(framerate)
    wf.writeframes(b"".join(data))
    wf.close()

def record(f, time=5):
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=framerate,
        input=True,
        frames_per_buffer=NUM_SAMPLES,
    )
    my_buf = []
    count = 0
    print("录音中({}s)".format(str(time)))
    while count < TIME * time:
        string_audio_data = stream.read(NUM_SAMPLES)
        my_buf.append(string_audio_data)
        count += 1
        print(".", end="", flush=True)

    save_wave_file(f, my_buf)
    stream.close()


if __name__ == '__main__':
    #直接跑该代码可以录自己的音保存成wav文件
    record('../data_aishell/output.wav', time=2)

# sounddevie版
# import sounddevice as sd
# from scipy.io.wavfile import write
# framerate = 16000 # Sample rate
# channels = 1
# seconds = 2 # Duration of recording
# print('录音中...')
# myrecording = sd.rec(int(seconds * framerate), samplerate=framerate, channels=channels)
# sd.wait() # Wait until recording is finished
# write('/Users/zhoubin/Downloads/wmm.wav', framerate, myrecording) # Save as WAV file
