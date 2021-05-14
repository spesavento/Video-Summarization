#!/usr/bin/env python

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage import io
from sklearn import preprocessing
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

def FrameSimilarity(frames_jpg_path):
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
    # for i in range (0, 4000):
        frame_a = cv2.imread(frames_jpg_path+'frame'+str(i)+'.jpg')
        frame_b = cv2.imread(frames_jpg_path+'frame'+str(i+1)+'.jpg')
        frame_a_bw = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        frame_b_bw = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
        ssim_ab = ssim(frame_a_bw, frame_b_bw)
        ssim_ab = round(ssim_ab, 3)
        ssi_array.append(ssim_ab)
    return (ssi_array)

def FrameChange(ssi_array):
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
        # 0.6 is chosen because a 60% change in similarity works well for a shot change threshold
        if (ssim_bc/ssim_ab < 0.6 and ssim_bc/ssim_cd < 0.6 and i-last_hit > 20):
            framechange_array.append(i+2)
            last_hit = i+2
    return (framechange_array)

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

def FindFaces(framechange_array, frames_jpg_path):
    # Load face classifier, using "Haar" classifier, basic but works fine
    face_classifier = cv2.CascadeClassifier('haarcascade_face_classifier.xml')
    # initialize array variable to record faces
    face_array = []
    # loop through the number of shots
    for x in range (0, len(framechange_array)-1):
        frames_in_shot = framechange_array[x+1] - framechange_array[x] - 1
        face_total = 0
        for y in range (framechange_array[x], framechange_array[x+1]-1):
            # url of frame image to analyze
            filename=frames_jpg_path+'frame'+str(y)+'.jpg'
            # read it into OpenCV
            img = cv2.imread(filename)
            # convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # detect faces
            # scaleFactor – how much the image size is reduced at each image scale
            # minNeighbors = 4 gives few false positives, but misses a few faces
            faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            for (x, y, w, h) in faces:
                face_total = face_total + 1
        face_array.append(face_total)
    # return a normalized weighted array, value 0 to 1
    face_array_normalized = preprocessing.minmax_scale(face_array, feature_range=(0, 1))
    face_array = [round(num, 3) for num in face_array_normalized]
    return(face_array)

def FindPeople(framechange_array, frames_jpg_path):
    # OpenCV has a pre-trained person model using Histogram Oriented Gradients (HOG)
    # and Linear SVM
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # initialize array variable to record faces
    people_array = []
    for x in range (0, len(framechange_array)-1):
        frames_in_shot = framechange_array[x+1] - framechange_array[x] - 1
        people_total = 0
        for y in range (framechange_array[x], framechange_array[x+1]-1):
            # url of frame image to analyze
            filename=frames_jpg_path+'frame'+str(y)+'.jpg'
            # read it into OpenCV
            image = cv2.imread(filename)
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

def TotalWeights(shot_array, action_array, face_array, people_array):
    # use numpy to add the weight arrays
    # for now a simple addition of action, face, people weights
    face_array_scaled = [element * 0.5 for element in face_array]
    people_array_scaled = [element * 0.5 for element in people_array]
    arr = []
    arr.append(action_array)
    arr.append(face_array_scaled)
    arr.append(people_array_scaled)
    np_arr = np.array(arr)
    np_weight = np_arr.sum(axis=0)
    total_weight = list(np.around(np.array(np_weight),3))
    # total_weight = np_weight.tolist()
    for x in range (0, len(shot_array)):
        shot_array[x].append(total_weight[x])
    totalweight_array = shot_array
    # returns a multi-level weighted array [shot start, shot end, total weight]
    return(totalweight_array)

def SaveSummaryFrames(totalweight_array, summary_frame_path, frames_jpg_path):
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
        frame_count = frame_count + num_frames
        # stop if frame_count is 90 sec (90 sec * 30 fps = 2700)
        if (frame_count < 2700):
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
            img = cv2.imread(shot_image)
            summary_image = summary_frame_path+numlist[count]+'.jpg'
            cv2.imwrite(summary_image,img)
            count = count+1

# Convert frames folder to video using OpenCV
def FramesToVideo(summary_frame_path,pathOut,fps,frame_width,frame_height):
    frame_array = []
    files = [f for f in os.listdir(summary_frame_path) if isfile(join(summary_frame_path,f))]
    # sort the files
    # see python reference https://docs.python.org/3/howto/sorting.html
    files.sort()
    for i in range(len(files)):
        filename=summary_frame_path+files[i]
        #reading each files
        img = cv2.imread(filename)
        # height, width, layers = img.shape
        # size = (width,height)
        #inserting the frames into an image array
        frame_array.append(img)
    # define the parameters for creating the video
    # .mp4 is a good choice for playing videos, works on OSX and Windows
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(pathOut, fourcc, fps, (frame_width,frame_height))
    # create the video from frame array
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

def MakeCollage(framechange_array, frames_jpg_path, collage_path):
    # creates a collage of the shots in a video
    offset = 30
    i = 0
    # start with a blank image that is the same width (1600px) of 5 frames
    im_v = cv2.imread('top.jpg')
    for x in range (0, len(framechange_array)-5, 5):
        im_a = cv2.imread(frames_jpg_path+'frame'+str(framechange_array[x]+offset)+'.jpg')
        im_b = cv2.imread(frames_jpg_path+'frame'+str(framechange_array[x+1]+offset)+'.jpg')
        im_c = cv2.imread(frames_jpg_path+'frame'+str(framechange_array[x+2]+offset)+'.jpg')
        im_d = cv2.imread(frames_jpg_path+'frame'+str(framechange_array[x+3]+offset)+'.jpg')
        im_e = cv2.imread(frames_jpg_path+'frame'+str(framechange_array[x+4]+offset)+'.jpg')
        cv2.putText(im_a, str(x), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)
        cv2.putText(im_b, str(x+1), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)
        cv2.putText(im_c, str(x+2), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)
        cv2.putText(im_d, str(x+3), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)
        cv2.putText(im_e, str(x+4), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)
        im_h = cv2.hconcat([im_a, im_b, im_c, im_d, im_e])
        im_v = cv2.vconcat([im_v, im_h])
    cv2.imwrite(collage_path, im_v)


def main():

    # name of the video to process
    video_name = 'concert'

    # jpg video frames to be analyzed - ordered frame0.jpg, frame1.jpg, etc.
    frames_jpg_path = 'project_dataset/frames/'+video_name+'/'

    # directory for summary frames and summary video
    summary_frame_path = 'summary/'+video_name+'/frames/'
    summary_video_path = 'summary/'+video_name+'/summary.mp4'
    collage_path = 'summary/'+video_name+'/collage.jpg'

    # start processing the video

    # get ssi_array, the structured similarity between adjacent frames
    print ('\nssi_array')
    print ('the similarity between adjacent frames ... takes a long minute')
    ssi_array = FrameSimilarity(frames_jpg_path)
    print(str(ssi_array[0 : 50])+' ... more')

    # get the framechange_array, which are the shot boundary frames
    print ('\nframechange_array')
    print ('these are the frames where the shot changed')
    framechange_array = FrameChange(ssi_array)
    print(str(framechange_array))

    # get the shot_array, showing the shot sequences start, end
    print ('\nshot_array')
    shot_array = ShotArray(framechange_array)
    print (str(len(shot_array))+' shots in the video')
    print(str(shot_array))

    # get action_array, shows the average action weight for each shot
    print ('\naction_array')
    action_array = FindAction(framechange_array, ssi_array)
    print(str(len(action_array))+' action weights')
    print(str(action_array))

    # get the face array
    print('\nface_array')
    face_array = FindFaces(framechange_array, frames_jpg_path)
    print(str(len(face_array))+' face weights')
    print(str(face_array))

    # get the people array
    print('\npeople_array')
    people_array = FindPeople(framechange_array, frames_jpg_path)
    print('there are '+str(len(people_array))+' people weights')
    print(str(people_array))

    # total the weights
    print('\ntotalweight_array')
    print('[shot start, shot end, total weight]')
    totalweight_array = TotalWeights(shot_array, action_array, face_array, people_array)
    print(str(totalweight_array))

    # create summary frames in a folder
    SaveSummaryFrames(totalweight_array,summary_frame_path, frames_jpg_path)

    # create summary video
    print('\nfrom the summary frames, creating a summary video')
    FramesToVideo(summary_frame_path, summary_video_path, 30, 320, 180)
    print('the summary video is stored as '+summary_video_path)

    # optional - make a photo collage of the shots
    print('\nbonus: photo collage of scenes saved as collage.jpg in the root folder')
    MakeCollage(framechange_array, frames_jpg_path, collage_path)

    # Add audio

    # Play with video player
    # vp.PlayVideo(summary_video_path)



if __name__=="__main__":
    main()
