  
from transnetv2 import TransNetV2
import oldvideoplayer as vp
video_name = 'soccer'

summary_video_path = '../project_files/project_dataset/'+video_name+'.mp4'
# vp.PlayVideo(summary_video_path)
# location of learned weights is automatically inferred
# add argument model_dir="/path/to/transnetv2-weights/" to TransNetV2() if it fails
model = TransNetV2()
video_frames, single_frame_predictions, all_frame_predictions = model.predict_video(summary_video_path)

scenes = model.predictions_to_scenes(single_frame_predictions)

# scenes = []

# text_file = open("soccer.mp4.scenes.txt", "r")

# for line in text_file.readlines():
#     scenes.append(line.split(' '))

for i in range(len(scenes)):
    print(scenes[i])
    # start_frame = int(scenes[i][0])
    # end_frame = int(scenes[i][1])

    # vframe = start_frame // 30
    # minutes = vframe % 60
    # min_str = str(minutes)
    # if (minutes < 10):
    #     min_str = "0" + min_str

    # print("[" + str(vframe // 60) + ":" + min_str + ", ", end='')

    # vframe = end_frame // 30
    # minutes = vframe % 60
    # min_str = str(minutes)
    # if (minutes < 10):
    #     min_str = "0" + min_str

    # print(str(vframe // 60) + ":" + min_str + "]")
