#!/usr/bin/env python

# python -m pip install -U scikit-image
# (also Cython, numpy, scipi, matplotlib, networkx, pillow, imageio, tifffile, PyWavelets)
# 
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import mean_squared_error
# import numpy as np 
import cv2
# import glob
import numpy as np
import struct
import os
from os.path import isfile, join
from matplotlib import pyplot as plt
import moviepy.editor as mpe
import natsort
import wave
import videoplayer as vp
import sys
import struct

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures, MidTermFeatures
import matplotlib.pyplot as plt

im_shape = (320, 180)
total_frames = -1

# Creates an array of shot-change frames
def ShotChange(full_frame_path):
    print ('determining shot changes ...this might take a minute')
    # create array of file images and sort them
    # files = [f for f in os.listdir(full_frame_path) if isfile(join(full_frame_path,f))]
    # files.sort()
    files = os.listdir(full_frame_path)
    # number of frames in folder
    num = len(files)
    total_frames = num
    # initialize the shot change array variable
    # keep the very first frame as a scene change
    shot_change = [0]
    # initialize the first 3 frames
    last_frame = 0
    # calculate the ssim on the first 3 frames
    # 4 JSON frames - a b c d - comparisions: ab, bc, cd
    # if bc goes down far enough, that is considered a new shot
    frame_a = cv2.imread(full_frame_path+'frame0.jpg')
    frame_b = cv2.imread(full_frame_path+'frame1.jpg')
    frame_c = cv2.imread(full_frame_path+'frame2.jpg')
    frame_a_bw = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    frame_b_bw = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    frame_c_bw = cv2.cvtColor(frame_c, cv2.COLOR_BGR2GRAY)
    ssim_ab = ssim(frame_a_bw, frame_b_bw)
    ssim_bc = ssim(frame_b_bw, frame_c_bw)
    # now loop through all frames to look for "local minimums"
    # only go up to number of frames-3
    for i in range(0, num-3):
        # read in four frames to opencv
        frame_d = cv2.imread(full_frame_path+'frame'+str(i+3)+'.jpg')
        # convert them to grayscale
        frame_d_bw = cv2.cvtColor(frame_d, cv2.COLOR_BGR2GRAY)
        # calculate ssim between adjacent frames
        ssim_cd = ssim(frame_c_bw, frame_d_bw)
        # we have ssim_ab, ssim_bc, ssim_cd ... we want to know if ssim_bc is a "local minimum"
        # for now 0.6 is randomly chosen, seems to work OK
        # is similarity of bc low compared to ab and cd, don't accept shot similarity if less than 20 frames
        if (ssim_bc/ssim_ab < 0.6 and ssim_bc/ssim_cd < 0.6 and i-last_frame > 20):
            # print ('shot change frame '+ str(i+2))
            # for comparision to next shot
            last_frame = i+2
            # build the shot_change array with the frame numbers where change happens
            shot_change.append(i+2)
        # slide the window
        ssim_ab = ssim_bc
        ssim_bc = ssim_cd
        frame_c_bw = frame_d_bw
    print('shot change array is:')
    print(shot_change)
    return shot_change

# Take each Shot section and reduce it 1:6
# This is too simple, just taking every 6th frame, but identifying the shots
def MakeSummaryFrames(full_frame_path,summary_frame_path, shot_change):
    print ('storing summary frames in '+summary_frame_path+'...')
    num_shots = len(shot_change)
    # z = 0
    for i in range (0, num_shots-1):
        frame_a = shot_change[i]
        frame_b = shot_change[i+1]
        num_frames_in_shot = frame_b - frame_a
        num_frames_to_keep = int(num_frames_in_shot/6)
        save_frame = frame_a
        for y in range (0, num_frames_to_keep):
            # save every 6th frame
            save_frame = save_frame + 6
            summary_image = full_frame_path+'frame'+str(save_frame)+'.jpg'
            img = cv2.imread(summary_image)
            # write the shot numbers on the summary frames
            cv2.putText(
                img, #numpy image
                str(i), #text
                (10,60), #position
                cv2.FONT_HERSHEY_SIMPLEX, #font
                2, #font size
                (0, 0, 255), #font color red in BGR
                4) #font stroke
            #saves alphabetically
            alpha_number = str(save_frame).zfill(5)
            summary_frame_img = summary_frame_path+alpha_number+'.jpg'
            #write to summary frame folder
            cv2.imwrite(summary_frame_img,img)
            # z = z+1

# Convert frames folder to video using OpenCV
def FramesToVideo(summary_frame_path,pathOut,fps,frame_width,frame_height,audio_path,new_summary_audio_path):
    frame_array = []
    audio_frames = []
    files = [f for f in os.listdir(summary_frame_path) if isfile(join(summary_frame_path,f))]

    audio_object = wave.open(audio_path, 'r')
    framerate = audio_object.getframerate()
    print(audio_object.getnframes())

    # sort the files
    # see python reference https://docs.python.org/3/howto/sorting.html
    files.sort()
    for i in range(len(files)):
        filename=summary_frame_path+files[i]
        #reading each files
        FrameNum = int(os.path.splitext(files[i])[0])
        # Convert to audio frame
        AudioFrameNum = FrameNum * framerate // 30
        # print(AudioFrameNum)
        NumFramesToRead = (framerate // 30)
        # print(NumFramesToRead)
        audio_object.setpos(AudioFrameNum)
        NewAudioFrames = audio_object.readframes(NumFramesToRead)
        audio_frames.append(NewAudioFrames)

        img = cv2.imread(filename)
        # height, width, layers = img.shape
        # size = (width,height)
        # print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
    # define the parameters for creating the video
    # .mp4 is a good choice for playing videos, works on OSX and Windows
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(pathOut, fourcc, fps, (frame_width,frame_height))
    # create the video from frame array
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

    # Write new audio file
    sampleRate = framerate # hertz
    duration = len(audio_frames) / framerate # seconds
    obj = wave.open(new_summary_audio_path,'w')
    obj.setnchannels(2) # mono
    obj.setsampwidth(2)
    obj.setframerate(sampleRate)

    # Do audio analysis on audio_frames?
    # audio_object.setpos(10003423)
    # byte_obj = audio_object.readframes(1)
    # UnpackedChannels = struct.unpack("<hh", byte_obj)
    # LeftChannel = UnpackedChannels[0]
    # RightChannel = UnpackedChannels[1]
    # # LeftChannel = int.from_bytes(byte_obj[:2], byteorder=sys.byteorder)
    # # RightChannel = int.from_bytes(byte_obj[2:], byteorder=sys.byteorder)
    # # print(byte_obj)
    # # print(byte_obj[:2])
    # print(LeftChannel)
    # # print(byte_obj[2:])
    # print(RightChannel)
    # # print(len(byte_obj))

    # frame_magnitudes
    # for i in range(len(audio_frames)):
    #     print(len(audio_frames[i]))


    # os.system("pause")

    for i in range(len(audio_frames)):
        obj.writeframesraw(audio_frames[i])
    obj.close()

def ReadRGBFiles(full_frame_path):
    ims = []
    directory = full_frame_path
    dimensions = im_shape[0] * im_shape[1]
    for file in sorted(os.listdir(directory)):
        im_array = []
        filename = os.fsdecode(file)
        if filename.endswith(".rgb"): 
            Rbuf = []
            Gbuf = []
            Bbuf = []
            file_path = os.path.join(directory, filename)
            i = 0
            with open(file_path, "rb") as f:
                while (byte := f.read(1)):
                    if (i < dimensions):
                        Rbuf.append(struct.unpack('B',byte))
                    elif (i < 2*dimensions):
                        Gbuf.append(struct.unpack('B',byte))
                    elif (i < 3*dimensions):
                        Bbuf.append(struct.unpack('B',byte))
                    i+=1
            for j in range(im_shape[1]):
                row = []
                for k in range(im_shape[0]):
                    pixel = []
                    pixel.append(Rbuf[j*im_shape[0]+ k])
                    pixel.append(Gbuf[j*im_shape[0]+ k])
                    pixel.append(Bbuf[j*im_shape[0]+ k])
                    row.append(pixel)
                
                im_array.append(row)
            
            # for k in range(dimensions*3):
            #     print(im_array[k])

            im_array = np.array(im_array)
            im_array = im_array.reshape(im_shape[1], im_shape[0], 3)
            # print(im_array.shape)
            plt.imshow(im_array)
            plt.show()
            # cv2.imwrite('color_img.jpg', im_array)
            # cv2.imshow("image", im_array)
            # cv2.waitKey()

def SyncVideoWithAudio(old_video_name, video_name, audio_path):

    # images = [img for img in natsort.natsorted((os.listdir(full_frame_path))) if img.endswith(".jpg")]
    # frame = cv2.imread(os.path.join(full_frame_path, images[0]))
    # height, width, layers = frame.shape

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video = cv2.VideoWriter("temp.mp4", fourcc, 30, (width,height))

    # for image in images:
    #     video.write(cv2.imread(os.path.join(full_frame_path, image)))

    # cv2.destroyAllWindows()
    # video.release()

    my_clip = mpe.VideoFileClip(old_video_name)
    audio_background = mpe.AudioFileClip(audio_path)
    # final_audio = mpe.CompositeAudioClip([my_clip.audio, audio_background])
    final_clip = my_clip.set_audio(audio_background)
    final_clip.write_videofile(video_name,fps=30)

    my_clip.close()
    final_clip.close()
    audio_background.close()

def MakeAudioShots(audio_path):
    features = [1]
    [Fs, x] = audioBasicIO.read_audio_file(audio_path)
    x = audioBasicIO.stereo_to_mono(x)
    frame_size = (Fs // 30)
    F, f_names = ShortTermFeatures.feature_extraction(x, Fs, frame_size, frame_size, deltas=False)
    # plt.subplot(2,1,1); plt.plot(F[3,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[3]) 
    # plt.subplot(2,1,2); plt.plot(F[feature,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[feature]); plt.show()

    astd = []
    aave = []

    
    for i in range(len(features)):
        F[features[i],:] = [float(i)/sum(F[features[i],:]) for i in F[features[i],:]]
        astd.append(np.std(F[features[i],:]))
        aave.append(np.average(F[features[i],:]))

    which_shots = np.zeros(len(F[features[0],:])).flatten()
    print(which_shots.shape)

    for i in range(len(F[features[0],:])):
        for j in range(len(features)):
            if (abs(F[features[j],:][i]-aave[j]) > astd[j] * 4.5):
                which_shots[i] += F[features[j],:][i]
    
    prev_val = 0.0
    for i in range(len(which_shots)):
        # print(which_shots[i])
        vframe = i // 30
        minutes = vframe % 60
        min_str = str(minutes)
        if (minutes < 10):
            min_str = "0" + min_str
        if (prev_val == 0.0 and which_shots[i] > 0.0):
            print("[" + str(vframe // 60) + ":" + min_str + ", " + str(which_shots[i]) + ", ", end='')
        if (prev_val > 0.0 and which_shots[i] == 0.0):
            print(str(vframe // 60) + ":" + min_str + "]")

        prev_val = which_shots[i]

    return(which_shots)

def main():

    # directory of full video frames - ordered frame1.jpg, frame2.jpg, etc.
    # full_frame_path = "project_dataset/frames_rgb/soccer/"

    # audio path
    audio_path = "../project_files/project_dataset/audio/concert.wav"

    # directory for summary frames
    summary_frame_path = "summary/soccer/frames/"

    # directory for summary video
    # summary_video_path = "summary/soccer/summary.mp4"
    summary_video_path = "summary/soccer/video/soccer_video.mp4"

    # path for video with audio
    summary_video_audio_path = "summary/soccer/video/soccer_audio.mp4"

    new_summary_audio_path = "summary/soccer/video/sound.wav"

    # SyncVideoWithAudio(full_frame_path, summary_video_audio_path, audio_path)

    # ReadRGBFiles(full_frame_path)

    # get shot_change array
    # shot_change = ShotChange(full_frame_path)

    # # make summary frame folder
    # MakeSummaryFrames(full_frame_path,summary_frame_path,shot_change)

    # # make a video from the summary frame folder
    # FramesToVideo(summary_frame_path, summary_video_path, 30, 320, 180, audio_path, new_summary_audio_path)

    #SyncVideoWithAudio(summary_video_path, summary_video_audio_path, new_summary_audio_path)

    MakeAudioShots(audio_path)

    # vp.PlayVideo(summary_video_audio_path)

    # obj = wave.open(audio_path, 'r')
    # print( "Number of channels",obj.getnchannels())
    # print ( "Sample width",obj.getsampwidth())
    # print ( "Frame rate.",obj.getframerate())
    # print ("Number of frames",obj.getnframes())
    # print ( "parameters:",obj.getparams())

    # # Number of channels 2
    # # Sample width 2
    # # Frame rate. 48000
    # # Number of frames 25935872
    # # parameters: _wave_params(nchannels=2, sampwidth=2, framerate=48000, nframes=25935872, comptype='NONE', compname='not compressed')

    # # 1600 audio frames per video frame
    # # obj.close()


if __name__=="__main__":
    main()
