import cv2, numpy as np
import sys
import time
from time import sleep
from pydub import AudioSegment
import simpleaudio as sa
from simpleaudio import play_buffer
import os
from os.path import isfile, join

def flick(x):
    pass

def PlayVideo(summary_frame_path, summary_audio_path):

    # video = sys.argv[1]
    videobuffer = []
    files = [int(os.path.splitext(f)[0]) for f in os.listdir(summary_frame_path) if isfile(join(summary_frame_path,f))]
    # sort the files
    # see python reference https://docs.python.org/3/howto/sorting.html
    files.sort()
    for i in range(len(files)):
        filename=summary_frame_path+str(files[i])+".jpg"
        img = cv2.imread(filename)
        videobuffer.append(img)

    audiocap = AudioSegment.from_file(summary_audio_path, "wav")
    # audiocap = AudioSegment.from_wav(debug_audio)

    cv2.namedWindow('image')
    cv2.moveWindow('image',320,180)
    cv2.namedWindow('controls')
    cv2.moveWindow('controls',250,50)

    controls = np.zeros((50,750),np.uint8)
    cv2.putText(controls, "F: Resume/Play, P: Pause, R: Rewind, N: Fast Forward, Esc: Exit", (120,30), cv2.FONT_HERSHEY_PLAIN, 1, 200)

    framerate = audiocap.frame_rate

    wave_obj = sa.WaveObject(
                        audiocap.raw_data,
                        num_channels=audiocap.channels,
                        bytes_per_sample=audiocap.sample_width,
                        sample_rate=audiocap.frame_rate
                    )

    play_obj = None

    FPS = 1.0 / 30.0 * 1000.0

    tots = len(videobuffer)

    i = 0

    frame_rate = 30

    def process(im):
        return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    status = 'stay'

    last_audio_sync = 0

    while True:
        new_time = time.time()
        cv2.imshow("controls",controls)
        try:
            if i==tots-1:
                i=0
            # cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            im = videobuffer[i]

            # r = 750.0 / im.shape[1]
            # dim = (750, int(im.shape[0] * r))
            # im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
            # if im.shape[0]>600:
            #     im = cv2.resize(im, (500,500))
            #     controls = cv2.resize(controls, (im.shape[1],25))
            #cv2.putText(im, status, )
            cv2.imshow('image', im)
            status = { ord('p'):'stay', ord('P'):'stay',
                        ord('f'):'play', ord('F'):'play',
                        ord('r'):'prev_frame', ord('R'):'prev_frame',
                        ord('n'):'next_frame', ord('N'):'next_frame',
                        -1: status, 
                        27: 'exit'}[cv2.waitKey(10)]

            if status == 'play':
                frame_rate = cv2.getTrackbarPos('F','image')

                if play_obj is None:
                    play_obj = wave_obj.play()

                if not play_obj.is_playing() or last_audio_sync > 30:
                    if play_obj is not None:
                        play_obj.stop()
                    # must have changed position
                    audio_frame_index = (i * 1000.0) // 30
                    newaudiocap = audiocap[audio_frame_index:]
                    wave_obj = sa.WaveObject(
                        newaudiocap.raw_data,
                        num_channels=audiocap.channels,
                        bytes_per_sample=audiocap.sample_width,
                        sample_rate=audiocap.frame_rate
                    )
                    play_obj = wave_obj.play()
                    last_audio_sync = 0
                # audio_frame_index = i / 30.0 * 1000.0
                #print(str(i) + ", " + str(audio_frame_index))
                # asa = audiocap[audio_frame_index:audio_frame_index+msbetweenframes]
                # play_buffer(asa.raw_data, 2, 2, 48000)
                last_audio_sync+=1
                i+=1
                while time.time() - new_time < 1.0/30.0:
                    pass
                cv2.setTrackbarPos('S','image',i)
                continue
            if status == 'stay':
                # i = cv2.getTrackbarPos('S','image')
                if play_obj is not None:
                    play_obj.stop()
            if status == 'exit':
                break
            if status=='prev_frame':
                i-=1
                status='stay'
            if status=='next_frame':
                i+=1 
                status='stay'

            while time.time() - new_time < 1.0/30.0:
                pass

        except KeyError:
            print("Invalid Key was pressed")
    cv2.destroyWindow('image')

if __name__=="__main__":
    
    video_names = ['test_video']

    for i in range(len(video_names)):
        # name of the video to process
        video_name = video_names[i]
        summary_frame_path = '../../project_files/summary/'+video_name+'/frames/'
        summary_audio_path = '../../project_files/summary/'+video_name+'/sound.wav'
        PlayVideo(summary_frame_path, summary_audio_path)
