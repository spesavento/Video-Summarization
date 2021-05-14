
#!/usr/bin/env python

import numpy as np
import cv2
import numpy as np
import pickle as pl
import struct
import os
from os.path import isfile, join
from matplotlib import pyplot as plt
import moviepy.editor as mpe
import natsort
import wave
from PIL import Image
from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils
from array import *
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage import io
from sklearn import preprocessing
import oldvideoplayer as vp
from pathlib import Path
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures, MidTermFeatures
import matplotlib.pyplot as plt
import time
import motionvectors as BlockMatching

# FOR ML
# from transnetv2 import TransNetV2

from moviepy.editor import *
import pygame

def CacheImages(frames_jpg_path):
    files = [f for f in os.listdir(frames_jpg_path) if isfile(join(frames_jpg_path,f))]
    files.sort()
    # initialize array
    all_frames = []
    # number of adjacent frames
    numadj = len(files)
    # loop through all adjacent frames and calculate the ssi
    for i in range (0, numadj):
    # for i in range (0, 3000):
        all_frames.append(cv2.imread(frames_jpg_path+'frame'+str(i)+'.jpg'))

    return all_frames

def FrameSimilarity(frames_jpg_path, isCentered, all_frames):
    # calculates the "structured similarity index" between adjacent frames
    # ssim() looks at luminance, contrast and structure, it is a scikit-image function
    # we use ssim() for both (1) Shot Change detection, and (2) Action weight
    files = [f for f in os.listdir(frames_jpg_path) if isfile(join(frames_jpg_path,f))]
    files.sort()
    # initialize array
    ssi_array = []
    # number of adjacent frames
    numadj = len(files)-2
    # loop through all adjacent frames and calculate the ssi
    for i in range (0, numadj):
    # for i in range (0, 3000):
        # frame_a = cv2.imread(frames_jpg_path+'frame'+str(i)+'.jpg')
        # frame_b = cv2.imread(frames_jpg_path+'frame'+str(i+1)+'.jpg')
        frame_a = all_frames[i]
        frame_b = all_frames[i+1]
        # crop frame images to center-weight them
        if isCentered is True:
            crop_img_a = frame_a[20:160, 50:270] #y1:y2 x1:x2 orginal is 320 w x 180 h
            crop_img_b = frame_b[20:160, 50:270]
        else:
            crop_img_a = frame_a
            crop_img_b = frame_b
        frame_a_bw = cv2.cvtColor(crop_img_a, cv2.COLOR_BGR2GRAY)
        frame_b_bw = cv2.cvtColor(crop_img_b, cv2.COLOR_BGR2GRAY)
        ssim_ab = ssim(frame_a_bw, frame_b_bw)#, multichannel=False, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255)
        ssim_ab = round(ssim_ab, 3)
        ssi_array.append(ssim_ab)
    return (ssi_array)

def FrameChange(ssi_array, frames_jpg_path, all_frames):
    # this function finds the frames at the shot boundary
    # length of ssi_array, how many adjacent frames
    num = len(ssi_array)
    # initialize the shot_array variable
    framechange_array = [0]
    last_hit = 0
    for i in range (0, num-3):
    
        ssim_ab = ssi_array[i]
        ssim_bc = ssi_array[i+1]
        ssim_cd = ssi_array[i+2]

        firstCheckPass = False
   
        frame_a_index = i + 1
        if (frame_a_index < 0): 
            frame_a_index = 0

        frame_b_index = i + 2
        if (frame_b_index >= num-3):
            frame_b_index = num-3-1

        frame_a = all_frames[frame_a_index]
        frame_b = all_frames[frame_b_index]
        # frame_a = cv2.imread(frames_jpg_path+'frame'+str(frame_a_index)+'.jpg')
        # frame_b = cv2.imread(frames_jpg_path+'frame'+str(frame_b_index)+'.jpg')

        hist1 = []
        color = ('b','g','r')
        for j,col in enumerate(color):
            histr = cv2.calcHist([frame_a],[j],None,[256],[0,256])
            hist1.append(histr)

        hist2 = []
        for j,col in enumerate(color):
            histr = cv2.calcHist([frame_b],[j],None,[256],[0,256])
            hist2.append(histr)

        hist1a = np.asarray(hist1)
        hist2a = np.asarray(hist2)

        average_dist = 0

        hist1a_max = []
        hist2a_max = []
        SimilarColorCheck = False

        for j in range(3):
            dist = cv2.compareHist(hist1a[j], hist2a[j], 0)
            average_dist = average_dist + dist
            max_val = 0

            max_color = 0
            for k in range(len(hist1a[j])):
                if hist1a[j][k][0] > max_val:
                    max_val = hist1a[j][k][0]
                    max_color = k
            hist1a_max.append(max_color)

            max_val = 0
            max_color = 0
            for k in range(len(hist2a[j])):
                if hist2a[j][k][0] > max_val:
                    max_val = hist2a[j][k][0]
                    max_color = k
            hist2a_max.append(max_color)

        CumulativeColorDiffs = 0

        for j in range(3):
            CumulativeColorDiffs += abs(hist1a_max[j] - hist2a_max[j])

        if CumulativeColorDiffs < 20:
            SimilarColorCheck = True

        average_dist = average_dist / 3.0

        ssim_histo_val = min(max((1.0 - ssim_bc/ssim_ab), 0), 1.0) + min(max((1.0 - ssim_bc/ssim_cd), 0), 1.0) + (1.0 - average_dist)

        if (ssim_bc/ssim_ab < 0.6 and ssim_bc/ssim_cd < 0.6):
            if average_dist < 0.95:
                firstCheckPass = True
        elif (ssim_bc/ssim_ab < 0.6 or ssim_bc/ssim_cd < 0.6):
            if average_dist < 0.4:
                firstCheckPass = True
            elif ssim_histo_val > 1.0 and SimilarColorCheck is False:
                firstCheckPass = True
        elif ssim_histo_val > 1.0 and SimilarColorCheck is False:
            firstCheckPass = True
        
        # 0.6 is chosen because a 60% change in similarity works well for a shot change threshold
        if (firstCheckPass is True and i+2-last_hit > 15):
            framechange_array.append(i+2)
            last_hit = i+2


    # USE FOR A 2ND PASS THROUGH SHOTS TO MAKE SURE THEY ARE CORRECT

    new_frame_changes = []

    last_hit = 0

    # for x in range(len(framechange_array)-1):
    #     shots_in_frame = framechange_array[x+1] - framechange_array[x] - 1

    #     for k in range(0, shots_in_frame-10, 10):

    #         i = framechange_array[x] + k
    #         frame_a = cv2.imread(frames_jpg_path+'frame'+str(i)+'.jpg')
    #         frame_b = cv2.imread(frames_jpg_path+'frame'+str(i+10)+'.jpg')

    #         frame_a_bw = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    #         frame_b_bw = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    #         average_dist = ssim(frame_a_bw, frame_b_bw)
    #         #mad = np.sum(np.abs(np.subtract(frame_a_bw, frame_b_bw)))/(frame_a_bw.shape[0])

    #         # hist1 = []
    #         # color = ('b','g','r')
    #         # for j,col in enumerate(color):
    #         #     histr = cv2.calcHist([frame_a],[j],None,[256],[0,256])
    #         #     hist1.append(histr)

    #         # hist2 = []
    #         # for j,col in enumerate(color):
    #         #     histr = cv2.calcHist([frame_b],[j],None,[256],[0,256])
    #         #     hist2.append(histr)

    #         # hist1a = np.asarray(hist1)
    #         # hist2a = np.asarray(hist2)

    #         # average_dist = 0

    #         # for j in range(3):
    #         #     dist = cv2.compareHist(hist1a[j], hist2a[j], 0)
    #         #     average_dist = average_dist + dist

    #         # average_dist = average_dist / 3.0

    #         # average_dist, residual_frame = BlockMatching.main(frames_jpg_path+'frame'+str(i)+'.jpg', frames_jpg_path+'frame'+str(i+10)+'.jpg')

    #         # print(average_dist)

    #         if (average_dist < 0.05 and i - last_hit > 50):
    #             # Split up the frame here
    #             last_hit = i
    #             print(i)
    #             new_frame_changes.append(i+5)


    # framechange_array_copy = framechange_array.copy()

    # for i in range(len(new_frame_changes)):
    #     for k in range(len(framechange_array_copy)):
    #         if (new_frame_changes[i] > framechange_array_copy[k]):
    #             if k == len(framechange_array_copy) - 1 or (new_frame_changes[i] < framechange_array_copy[k+1]):
    #                 framechange_array.insert(k + 1, new_frame_changes[i])
    #                 break
                

    # add the last frame to the array to the end if last frame is more than last shot change
    if num-1 > framechange_array[-1] + 4:
        framechange_array.append(num-1)
    
    return (framechange_array)

# FOR ML
def FrameChangeDL(shot_array):
    framechange_array = [0]
    for i in range (1, len(shot_array)):
        framechange_array.append(int(shot_array[i][0]))
    
    framechange_array.append(int(shot_array[len(shot_array)-1][1]) + 1)
    return(framechange_array)

# FOR ML
# def ShotArrayDL(video_path):

#     model = TransNetV2()
#     video_frames, single_frame_predictions, all_frame_predictions = model.predict_video(video_path)

#     scenes = model.predictions_to_scenes(single_frame_predictions)

#     scenes = scenes.tolist()

#     # scenes = []

#     # text_file = open("soccer.mp4.scenes.txt", "r")

#     # for line in text_file.readlines():
#     #     scenes.append(line.split(' '))

#     # scenes = [[int(y) for y in x] for x in scenes]
#     # for i in range(len(scenes)):
#     #     start_frame = int(scenes[i][0])
#     #     end_frame = int(scenes[i][1])

#     #     scenes.append([start_frame, end_frame])

#     return(scenes)

def ShotArray(framechange_array):
    # from where the frames change, create an array of the video shots
    shot_array = []
    shot_begin = 0
    shot_end = 0
    for x in range (0, len(framechange_array)-1):
        shot_begin = framechange_array[x]
        shot_end = framechange_array[x+1]-1
        shot_array.append([shot_begin,shot_end])
    return(shot_array)

def FindMotion(framechange_array, frames_jpg_path, all_frames):
    files = [f for f in os.listdir(frames_jpg_path) if isfile(join(frames_jpg_path,f))]
    files.sort()

    motion_step_size = 1

    numadj = len(files)

    residual_metrics = [0]

    for i in range (0, numadj-motion_step_size, motion_step_size):
        frame_a = all_frames[i]
        frame_b = all_frames[i+motion_step_size]
        # frame_a = cv2.imread(frames_jpg_path+'frame'+str(i)+'.jpg')
        # frame_b = cv2.imread(frames_jpg_path+'frame'+str(i+motion_step_size)+'.jpg')

        residual_metric = BlockMatching.main(frame_a, frame_b, outfile="OUTPUT", saveOutput=False, blockSize = 48)

        residual_metrics.append(residual_metric)

    res_mets = np.asarray(residual_metrics)
    raverage = np.average(res_mets)
    rstd = np.std(res_mets)

    # for i in range(len(res_mets)):
    #     if res_mets[i] < (raverage + 2 * rstd):
    #         residual_metrics[i] = 0.0

    action_array = []

    for x in range (0, len(framechange_array)-1):
        frames_in_shot = framechange_array[x+1] - framechange_array[x] - 1
        residual_total = 0
        resdiual_average = 0
        for y in range (framechange_array[x], framechange_array[x+1]-1):
            index_to_use = y // motion_step_size
            if y < ((framechange_array[x+1]-1-framechange_array[x])/2.0):
                temp = y % motion_step_size
                index_to_use = index_to_use + (motion_step_size - temp)
            if index_to_use > len(residual_metrics):
                index_to_use = len(residual_metrics) - 1
            residual_total = residual_total + residual_metrics[index_to_use]
        resdiual_average = residual_total / frames_in_shot
        action_array.append(resdiual_average)
    # in the action array, a smaller value means more action (less similarity within shot frames)
    # return a normalized weighted array, value 0 to 1
    action_array_normalized = preprocessing.minmax_scale(action_array, feature_range=(0, 1))
    action_array = [round(num, 3) for num in action_array_normalized]
    return(action_array)

    

def FindAction(framechange_array, ssi_array):
    # initialize action array
    action_array = []
    for x in range (0, len(framechange_array)-1):
        frames_in_shot = framechange_array[x+1] - framechange_array[x] - 1
        ssi_total = 0
        ssi_average = 0
        for y in range (framechange_array[x], framechange_array[x+1]-1):
            ssi_total = ssi_total + ssi_array[y]
        ssi_average = ssi_total / frames_in_shot
        # instead of low is high action, make high is high action
        ssi_average = 1 - ssi_average
        action_array.append(ssi_average)
    # in the action array, a smaller value means more action (less similarity within shot frames)
    # return a normalized weighted array, value 0 to 1
    action_array_normalized = preprocessing.minmax_scale(action_array, feature_range=(0, 1))
    action_array = [round(num, 3) for num in action_array_normalized]
    return(action_array)

def FindFaces(framechange_array, frames_jpg_path, all_frames):
    # Load face classifier, using "Haar" classifier, basic but works fine
    face_classifier = cv2.CascadeClassifier('haarcascade_face_classifier.xml')
    # initialize array variable to record faces
    face_array = []
    # loop through the number of shots
    for x in range (0, len(framechange_array)-1):
        frames_in_shot = framechange_array[x+1] - framechange_array[x] - 1
        face_total = 0
        for y in range (framechange_array[x], framechange_array[x+1]-1, 5):
            # url of frame image to analyze
            filename=frames_jpg_path+'frame'+str(y)+'.jpg'
            # read it into OpenCV
            # img = cv2.imread(filename)
            img = all_frames[y]
            # convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # detect faces
            # scaleFactor - how much the image size is reduced at each image scale
            # minNeighbors = 4 gives few false positives, but misses a few faces
            faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            for (x, y, w, h) in faces:
                face_total = face_total + 1
        face_array.append(face_total)
    # return a normalized weighted array, value 0 to 1
    face_array_normalized = preprocessing.minmax_scale(face_array, feature_range=(0, 1))
    face_array = [round(num, 3) for num in face_array_normalized]
    return(face_array)

def FindPeople(framechange_array, frames_jpg_path, all_frames):
    # OpenCV has a pre-trained person model using Histogram Oriented Gradients (HOG)
    # and Linear SVM
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # initialize array variable to record faces
    people_array = []
    for x in range (0, len(framechange_array)-1):
        frames_in_shot = framechange_array[x+1] - framechange_array[x] - 1
        people_total = 0
        for y in range (framechange_array[x], framechange_array[x+1]-1, 5):
            # url of frame image to analyze
            filename=frames_jpg_path+'frame'+str(y)+'.jpg'
            # read it into OpenCV
            image = all_frames[y]
            # image = cv2.imread(filename)
            # resize the image to increase speed (may try this on face detect as well)
            image = imutils.resize(image, width=min(400, image.shape[1]))
            orig = image.copy()
            # detect people in the image
            (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
            for (x, y, w, h) in rects:
                people_total = people_total + 1
        people_array.append(people_total)
    # return a normalized weighted array, value 0 to 1
    people_array_normalized = preprocessing.minmax_scale(people_array, feature_range=(0, 1))
    people_array = [round(num, 3) for num in people_array_normalized]
    return(people_array)

def FindAudioShots(framechange_array, audio_path):
    features = [1]
    [Fs, x] = audioBasicIO.read_audio_file(audio_path)
    x = audioBasicIO.stereo_to_mono(x)
    frame_size = (Fs // 30)
    F, f_names = ShortTermFeatures.feature_extraction(x, Fs, frame_size, frame_size, deltas=False)

    astd = []
    aave = []
    for i in range(len(features)):
        astd.append(np.std(F[features[i],:]))
        aave.append(np.average(F[features[i],:]))

    which_shots = np.zeros(len(F[features[0],:])).flatten()
    # print(which_shots.shape)

    for i in range(len(F[features[0],:])):
        for j in range(len(features)):
            if (abs(F[features[j],:][i]-aave[j]) > astd[j] * 3.5):
                which_shots[i] += F[features[j],:][i]
    
    audioshotchange_list = []

    prev_val = 0.0
    last_start = 0
    for i in range(len(F[1,:])):
        # print(which_shots[i])
        if (prev_val == 0.0 and which_shots[i] > 0.0):
            last_start = i
        if (prev_val > 0.0 and which_shots[i] == 0.0):
            audioshotchange_list.append([last_start, i, which_shots[last_start]])

        prev_val = which_shots[i]
    
    audio_array = np.zeros(len(framechange_array)-1)

    for x in range (0, len(framechange_array)-1):
        first_frame = framechange_array[x]
        last_frame = framechange_array[x+1]
        for y in range(len(audioshotchange_list)):
            if audioshotchange_list[y][0] >= first_frame and audioshotchange_list[y][0] < last_frame:
                audio_array[x] += audioshotchange_list[y][2]
        audio_array[x] /= (last_frame - first_frame)

    audio_array = preprocessing.minmax_scale(audio_array, feature_range=(0, 1))
    audio_array = [round(num, 3) for num in audio_array]
    return(audio_array)

def TotalWeights(shot_array, action_array, face_array, people_array, audio_array):
    # use numpy to add the weight arrays
    # for now a simple addition of action, face, people weights
    face_array_scaled = [element * 0.5 for element in face_array]
    people_array_scaled = [element * 0.3 for element in people_array]
    audio_array_scaled = [element * 0.6 for element in audio_array]
    action_array_scaled = [element * 0.6 for element in action_array]
    arr = []
    arr.append(action_array_scaled)
    arr.append(face_array_scaled)
    arr.append(people_array_scaled)
    arr.append(audio_array_scaled)
    np_arr = np.array(arr)
    np_weight = np_arr.sum(axis=0)
    total_weight = list(np.around(np.array(np_weight),3))
    # total_weight = np_weight.tolist()
    for x in range (0, len(shot_array)):
        shot_array[x].append(total_weight[x])
    totalweight_array = shot_array
    # returns a multi-level weighted array [shot start, shot end, total weight]
    return(totalweight_array)

def SaveSummaryFrames(totalweight_array, summary_frame_path, frames_jpg_path, action_array, face_array, people_array, audio_array, all_frames):
    # with weighted shots, save the summary frames into summary_frame_path
    # sort the array by weight descending, best shots first
    sorted_array = sorted(totalweight_array, key=lambda x: x[2], reverse=True)
    print('\nsorted_array')
    print('shots ordered by highest weight first')
    print(str(sorted_array))
    frame_count = 0
    summary_array = []
    ordered_array = []
    # first truncated the shots that won't be used
    # do this by counting the top weighted shots until
    # frame count is < 2700 (90 seconds x 30 fps)
    for x in range (0, len(sorted_array)-1):
        start_frame = sorted_array[x][0]
        end_frame = sorted_array[x][1]
        num_frames = end_frame - start_frame

        if num_frames > 300:
            sorted_array[x][0] = sorted_array[x][1] - 300
            num_frames = 300

        if (num_frames < 60):
            continue

        if frame_count + num_frames >= 2850:
            num_frames = 2850 - frame_count
            if (num_frames < 60):
                break
            sorted_array[x][1] = sorted_array[x][0] + num_frames

        frame_count = frame_count + num_frames
        # stop if frame_count is 90 sec (90 sec * 30 fps = 2700)
        if (frame_count < 2850):
            for z in range(len(totalweight_array)):
                if (sorted_array[x][0] == totalweight_array[z][0] or sorted_array[x][1] == totalweight_array[z][1]):
                    print("Action: ", 0.6 * action_array[z], ", Face: ", 0.5 * face_array[z], ", People: ", 0.3 * people_array[z], ", Audio: ", 0.6 * audio_array[z])
                    break
            summary_array.insert(x, sorted_array[x])
    # ordered array sort by shot start frame number
    ordered_array = sorted(summary_array, key=lambda x: x[0])
    print('\nordered_array')
    print('shots trimmed down to < 2700 frames, ordered by scene number')
    print(str(ordered_array))
    num_shots=len(ordered_array)
    # create a numeric list 0000, 0001, to 9999
    numlist = ["%04d" % x for x in range(10000)]
    count = 0
    # print(str(num_shots))
    for y in range (0,num_shots):
        start = ordered_array[y][0]
        end = ordered_array[y][1]
        # print(str(start))
        for z in range (start, end):
            shot_image = frames_jpg_path+'frame'+str(z)+'.jpg'
            # img = cv2.imread(shot_image)
            img = all_frames[z]
            summary_image = summary_frame_path+str(z)+'.jpg'
            # add shot number to frame
            # cv2.putText(
            #     img, #numpy image
            #     str(y), #text
            #     (10,60), #position
            #     cv2.FONT_HERSHEY_SIMPLEX, #font
            #     2, #font size
            #     (0, 0, 255), #font color red
            #     4) #font stroke
            cv2.imwrite(summary_image,img)
            count = count+1

def sort(lst):
    return sorted(lst, key = str)

# Convert frames folder to video using OpenCV
def FramesToVideo(summary_frame_path,pathOut,fps,frame_width,frame_height,audio_path,new_audio_path, all_frames,isML=False, audioOnly=False):
    frame_array = []
    audio_frames = []

    audio_object = wave.open(audio_path, 'r')
    framerate = audio_object.getframerate()

    if isML:
        files = [int(os.path.splitext(f)[0].split("frame")[1]) for f in os.listdir(summary_frame_path) if isfile(join(summary_frame_path,f))]
    else:
        files = [int(os.path.splitext(f)[0]) for f in os.listdir(summary_frame_path) if isfile(join(summary_frame_path,f))]
    # sort the files
    # see python reference https://docs.python.org/3/howto/sorting.html
    files.sort()
    for i in range(len(files)):
        if isML:
            filename=summary_frame_path+"frame"+str(files[i])+".jpg"
        else:
            filename=summary_frame_path+str(files[i])+".jpg"
        FrameNum = files[i]

        # Convert to audio frame
        # print(files[i])
        # print(FrameNum)
        AudioFrameNum = ((FrameNum * framerate) // 30)
        # print(AudioFrameNum)
        NumFramesToRead = (framerate // 30)
        # print(NumFramesToRead)

        audio_object.setpos(AudioFrameNum)
        NewAudioFrames = audio_object.readframes(NumFramesToRead)
        audio_frames.append(NewAudioFrames)

        if audioOnly is False:
            #reading each files
            # img = cv2.imread(filename)
            img = all_frames[FrameNum]
            # height, width, layers = img.shape
            # size = (width,height)
            #inserting the frames into an image array
            frame_array.append(img)

    if audioOnly is False:
        # define the parameters for creating the video
        # .mp4 is a good choice for playing videos, works on OSX and Windows
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(pathOut, fourcc, fps, (frame_width,frame_height))
        # create the video from frame array
        for i in range(len(frame_array)):
            # writing to a image array
            out.write(frame_array[i])
        out.release()

    # Write new audio file
    sampleRate = framerate # hertz
    duration = len(audio_frames) / framerate # seconds
    obj = wave.open(new_audio_path,'w')
    obj.setnchannels(2) # mono
    obj.setsampwidth(2)
    obj.setframerate(sampleRate)

    for i in range(len(audio_frames)):
        obj.writeframesraw(audio_frames[i])
    obj.close()

def MakeCollage(framechange_array, frames_jpg_path, collage_path):
    # creates a collage of the shots in a video, the collage shows shot # and frame #
    # imporant - the top.jpg must be in the folder path, and it has to be exact width of 2240px
    # take the frame one forward of the shot change
    offset = 1
    i = 0
    # start with a blank image that is the same width (2240px) of 7 frames
    im_v = cv2.imread('top.jpg')
    # make a collage that is 7 frames wide
    for x in range (0, len(framechange_array)-7, 7):
        im_a = cv2.imread(frames_jpg_path+'frame'+str(framechange_array[x]+offset)+'.jpg')
        im_b = cv2.imread(frames_jpg_path+'frame'+str(framechange_array[x+1]+offset)+'.jpg')
        im_c = cv2.imread(frames_jpg_path+'frame'+str(framechange_array[x+2]+offset)+'.jpg')
        im_d = cv2.imread(frames_jpg_path+'frame'+str(framechange_array[x+3]+offset)+'.jpg')
        im_e = cv2.imread(frames_jpg_path+'frame'+str(framechange_array[x+4]+offset)+'.jpg')
        im_f = cv2.imread(frames_jpg_path+'frame'+str(framechange_array[x+5]+offset)+'.jpg')
        im_g = cv2.imread(frames_jpg_path+'frame'+str(framechange_array[x+6]+offset)+'.jpg')
        # add the shot numbers to the collage images
        cv2.putText(im_a, str(x), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.putText(im_b, str(x+1), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.putText(im_c, str(x+2), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.putText(im_d, str(x+3), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.putText(im_e, str(x+4), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.putText(im_f, str(x+5), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.putText(im_g, str(x+6), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        # add the frame numbers to the collage images
        cv2.putText(im_a, str(framechange_array[x]), (120,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
        cv2.putText(im_b, str(framechange_array[x+1]), (120,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
        cv2.putText(im_c, str(framechange_array[x+2]), (120,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
        cv2.putText(im_d, str(framechange_array[x+3]), (120,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
        cv2.putText(im_e, str(framechange_array[x+4]), (120,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
        cv2.putText(im_f, str(framechange_array[x+5]), (120,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
        cv2.putText(im_g, str(framechange_array[x+6]), (120,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
        # build the collage
        im_h = cv2.hconcat([im_a, im_b, im_c, im_d, im_e, im_f, im_g])
        im_v = cv2.vconcat([im_v, im_h])
    cv2.imwrite(collage_path, im_v)

def SyncVideoWithAudio(old_video_name, video_name, audio_path):

    my_clip = mpe.VideoFileClip(old_video_name)
    audio_background = mpe.AudioFileClip(audio_path)
    final_clip = my_clip.set_audio(audio_background)
    final_clip.write_videofile(video_name,fps=30)

    my_clip.close()
    final_clip.close()
    audio_background.close()

def main():

    video_names = ['test_video']

    for i in range(len(video_names)):

        start_time = time.time()

        # name of the video to process
        video_name = video_names[i]

        # jpg video frames to be analyzed - ordered frame0.jpg, frame1.jpg, etc.
        frames_jpg_path = '../../project_files/project_dataset/frames/'+video_name+'/'

        # video path
        video_path = '../../project_files/project_dataset/'+video_name+'.mp4'

        dataset_path =  '../../project_files/project_dataset/'

        # audio to process
        audio_path = '../../project_files/project_dataset/audio/'+video_name+'.wav'

        # new audio path
        new_audio_path = "../../project_files/summary/" +video_name+ "/sound.wav"

        # directory for summary frames and summary video
        summary_frame_path = '../../project_files/summary/'+video_name+'/frames/'
        summary_video_path = '../../project_files/summary/'+video_name+'/summary.mp4'
        summary_video_audio_path = '../../project_files/summary/'+video_name+'/summary_with_audio.mp4'
        collage_path = '../../project_files/summary/'+video_name+'/collage.jpg'

        # Make dir if it doesn't exist
        Path(summary_frame_path).mkdir(parents=True, exist_ok=True)

        # empty the summary folders and summary results
        print ('\nremoving all previous summary files in summary/shot folders')
        filesToRemove = [os.path.join(summary_frame_path,f) for f in os.listdir(summary_frame_path)]
        for f in filesToRemove:
            os.remove(f)
        if os.path.exists(summary_video_path):
            os.remove(summary_video_path)
        if os.path.exists(collage_path):
            os.remove(collage_path)

        all_frames = CacheImages(frames_jpg_path)

        print("Time taken: ", time.time()-start_time, "s")

        # get ssi_array, the structured similarity between adjacent frames
        print ('\nssi_array')
        print ('the similarity between adjacent frames ... takes a long minute')
        ssi_array = FrameSimilarity(frames_jpg_path, True, all_frames)
        print(str(ssi_array[0 : 50])+' ... more')

        print("Time taken: ", time.time()-start_time, "s")

        # get the framechange_array, which are the shot boundary frames
        print ('\nframechange_array')
        print ('these are the frames where the shot changed')
        framechange_array = FrameChange(ssi_array, frames_jpg_path, all_frames)
        print (str(len(framechange_array))+' framechangess in the video')
        print(str(framechange_array))

        # get the shot_array, showing the shot sequences start, end
        print ('\nshot_array')
        shot_array = ShotArray(framechange_array)
        print (str(len(shot_array))+' shots in the video')
        print(str(shot_array))

        print("Time taken: ", time.time()-start_time, "s")

        # FOR ML

        # FramesToVideo(frames_jpg_path,video_path,30,320,180,audio_path,new_audio_path, True)

        # print ('\nremoving all previous scenes files in dataset folders')
        # filesToRemove = [os.path.join(dataset_path,f) for f in os.listdir(dataset_path) if f.endswith('.txt')]
        # for f in filesToRemove:
        #     os.remove(f)

        # get the shot_array, showing the shot sequences start, end
        # print ('\nshot_array')
        # shot_array = ShotArrayDL(video_path)
        # print (str(len(shot_array))+' shots in the video')
        # print(str(shot_array))

        # # get the framechange_array, which are the shot boundary frames
        # print ('\nframechange_array')
        # print ('these are the frames where the shot changed')
        # framechange_array = FrameChangeDL(shot_array)
        # print (str(len(framechange_array))+' framechangess in the video')
        # print(str(framechange_array))

        # For motion estimation
        print ('\naction_array')
        action_array = FindMotion(framechange_array, frames_jpg_path, all_frames)
        print(str(len(action_array))+' action weights')
        print(str(action_array))

        print("Time taken: ", time.time()-start_time, "s")
        
        # get the audio array
        print('\naudio_array')
        audio_array = FindAudioShots(framechange_array, audio_path)
        print('there are '+str(len(audio_array))+' audio weights')
        print(str(audio_array))

        print("Time taken: ", time.time()-start_time, "s")

        # get action_array, shows the average action weight for each shot
        # print ('\naction_array')
        # action_array = FindAction(framechange_array, ssi_array)
        # print(str(len(action_array))+' action weights')
        # print(str(action_array))
        
        # FOR ML
        # action_array = np.zeros(len(audio_array))

        # get the face array
        print('\nface_array')
        face_array = FindFaces(framechange_array, frames_jpg_path, all_frames)
        print(str(len(face_array))+' face weights')
        print(str(face_array))

        print("Time taken: ", time.time()-start_time, "s")

        # get the people array
        print('\npeople_array')
        people_array = FindPeople(framechange_array, frames_jpg_path, all_frames)
        print('there are '+str(len(people_array))+' people weights')
        print(str(people_array))

        print("Time taken: ", time.time()-start_time, "s")

        # FOR ML
        # people_array = np.zeros(len(audio_array))

        # total the weights
        print('\ntotalweight_array')
        print('[shot start, shot end, total weight]')
        totalweight_array = TotalWeights(shot_array, action_array, face_array, people_array, audio_array)
        print(str(totalweight_array))

        # create summary frames in a folder
        SaveSummaryFrames(totalweight_array,summary_frame_path, frames_jpg_path, action_array, face_array, people_array, audio_array, all_frames)

        # create summary video
        print('\nfrom the summary frames, creating a summary video')
        FramesToVideo(summary_frame_path, summary_video_path, 30, 320, 180, audio_path, new_audio_path, all_frames, False, True)
        print('the summary video is stored as '+summary_video_path)

        print("Final time taken: ", time.time()-start_time, "s")

        FramesToVideo(summary_frame_path, summary_video_path, 30, 320, 180, audio_path, new_audio_path, all_frames, False, False)

        # Adding audio to video
        SyncVideoWithAudio(summary_video_path, summary_video_audio_path, new_audio_path)

        # # optional - make a photo collage of the shots
        print('\nbonus: photo collage of scenes saved as collage.jpg in the root folder')
        MakeCollage(framechange_array, frames_jpg_path, collage_path)

       

if __name__=="__main__":
    main()
