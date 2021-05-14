from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS
audio_path = "../project_files/project_dataset/audio/soccer.wav"
[Fs, x] = aIO.read_audio_file(audio_path)
segments = aS.silence_removal(x, Fs, 0.020, 0.020, smooth_window = 0.1, weight = 0.6, plot = False)

for i in range(len(segments)):
    print("[" + str(segments[i][0]) + "," + str(segments[i][1]) + "]")
    segments[i][0] = segments[i][0] // 30
    segments[i][1] = segments[i][1] // 30
