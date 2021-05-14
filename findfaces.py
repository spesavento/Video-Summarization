#!/usr/bin/env python

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage import io
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


def FindFaces(full_frame_path):
    # Load face classifier, using "Haar" classifier, basic but works fine
    face_classifier = cv2.CascadeClassifier('haarcascade_face_classifier.xml')
    # initialize array variable to record faces
    face_array = []
    # read the number of frames
    files = [f for f in os.listdir(full_frame_path) if isfile(join(full_frame_path,f))]
    files.sort()
    print ('detecting faces, takes a minute ...')
    for i in range(len(files)):
        # url of frame image to analyze
        filename=full_frame_path+'frame'+str(i)+'.jpg'
        # read it into OpenCV
        img = cv2.imread(filename)
        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # detect faces
        # scaleFactor – how much the image size is reduced at each image scale
        # minNeighbors = 5 gives few false positives, but misses a few
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            face_array.append(i)
            # # to display frame with faces
            # cv2.imshow('img', img)
            # cv2.waitKey()
    # the frames with faces
    print ("face_array = "+ str(face_array))


def main():

    # directory of full video frames - ordered frame1.jpg, frame2.jpg, etc.
    full_frame_path = "../project_files/project_dataset/frames/meridian/"

    FindFaces(full_frame_path)


if __name__=="__main__":
    main()
