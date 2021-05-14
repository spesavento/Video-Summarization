import numpy as np
import cv2
import random
import time
import os

# Adapted from https://github.com/gbanuru18/BlockMatching

debug = True

def YCrCb2BGR(image):

    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

def BGR2YCrCb(image):

    return cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)

def segmentImage(anchor, blockSize=16):

    h, w = anchor.shape
    hSegments = int(h / blockSize)
    wSegments = int(w / blockSize)
    totBlocks = int(hSegments * wSegments)

    return hSegments, wSegments

def getCenter(x, y, blockSize):

    return (int(x + blockSize/2), int(y + blockSize/2))

def getAnchorSearchArea(x, y, anchor, blockSize, searchArea):

    h, w = anchor.shape
    cx, cy = getCenter(x, y, blockSize)

    sx = max(0, cx-int(blockSize/2)-searchArea) 
    sy = max(0, cy-int(blockSize/2)-searchArea) 

    anchorSearch = anchor[sy:min(sy+searchArea*2+blockSize, h), sx:min(sx+searchArea*2+blockSize, w)]

    return anchorSearch

def getBlockZone(p, aSearch, tBlock, blockSize):

    px, py = p # coordinates of macroblock center
    px, py = px-int(blockSize/2), py-int(blockSize/2) # get top left corner of macroblock
    px, py = max(0,px), max(0,py) # ensure macroblock is within bounds

    aBlock = aSearch[py:py+blockSize, px:px+blockSize] # retrive macroblock from anchor search area


    try:
        assert aBlock.shape == tBlock.shape # must be same shape

    except Exception as e:
        print(e)
        print(f"ERROR - ABLOCK SHAPE: {aBlock.shape} != TBLOCK SHAPE: {tBlock.shape}")

    return aBlock

def getMAD(tBlock, aBlock):

    return np.sum(np.abs(np.subtract(tBlock, aBlock)))/(tBlock.shape[0]*tBlock.shape[1])

def getBestMatch(tBlock, aSearch, blockSize): #3 Step Search

    step = 4
    ah, aw = aSearch.shape
    acy, acx = int(ah/2), int(aw/2) 

    minMAD = float("+inf")
    minP = None

    while step >= 1:
        p1 = (acx, acy)
        p2 = (acx+step, acy)
        p3 = (acx, acy+step)
        p4 = (acx+step, acy+step)
        p5 = (acx-step, acy)
        p6 = (acx, acy-step)
        p7 = (acx-step, acy-step)
        p8 = (acx+step, acy-step)
        p9 = (acx-step, acy+step)
        pointList = [p1,p2,p3,p4,p5,p6,p7,p8,p9] # retrieve 9 search points

        for p in range(len(pointList)):
            aBlock = getBlockZone(pointList[p], aSearch, tBlock, blockSize) # get anchor macroblock
            MAD = getMAD(tBlock, aBlock) # determine MAD
            if MAD < minMAD: # store point with minimum mAD
                minMAD = MAD
                minP = pointList[p]

        step = int(step/2)

    px, py = minP # center of anchor block with minimum MAD
    px, py = px - int(blockSize / 2), py - int(blockSize / 2) # get top left corner of minP
    px, py = max(0, px), max(0, py) # ensure minP is within bounds
    matchBlock = aSearch[py:py + blockSize, px:px + blockSize] # retrieve best macroblock from anchor search area

    return matchBlock



def blockSearchBody(anchor, target, blockSize, searchArea=7):

    h, w = anchor.shape
    hSegments, wSegments = segmentImage(anchor, blockSize)


    predicted = np.ones((h, w))*255
    bcount = 0
    for y in range(0, int(hSegments*blockSize), blockSize):
        for x in range(0, int(wSegments*blockSize), blockSize):
            bcount+=1
            targetBlock = target[y:y+blockSize, x:x+blockSize] #get current macroblock

            anchorSearchArea = getAnchorSearchArea(x, y, anchor, blockSize, searchArea) #get anchor search area

            anchorBlock = getBestMatch(targetBlock, anchorSearchArea, blockSize) #get best anchor macroblock
            predicted[y:y+blockSize, x:x+blockSize] = anchorBlock #add anchor block to predicted frame



    assert bcount == int(hSegments*wSegments) #check all macroblocks are accounted for

    return predicted

def getResidual(target, predicted):
    return np.subtract(target, predicted)

def getReconstructTarget(residual, predicted):
    return np.add(residual, predicted)

def showImages(*kwargs):
    for k in range(len(kwargs)):
        cv2.imshow(f"Image: {k}", k)
        cv2.waitKey(-1)

def getResidualMetric(residualFrame):
    return np.sum(np.abs(residualFrame))/(residualFrame.shape[0]*residualFrame.shape[1])

def preprocess(anchor, target, blockSize):

    if isinstance(anchor, str) and isinstance(target, str):
        anchorFrame = ((cv2.cvtColor(cv2.imread(anchor), cv2.COLOR_BGR2YCrCb)))[:, :, 0] # get luma component
        targetFrame = ((cv2.cvtColor(cv2.imread(target), cv2.COLOR_BGR2YCrCb)))[:, :, 0] # get luma component

    elif isinstance(anchor, np.ndarray) and isinstance(target, np.ndarray):
        anchorFrame = BGR2YCrCb(anchor)[:, :, 0] # get luma component
        targetFrame = BGR2YCrCb(target)[:, :, 0] # get luma component

    else:
        raise ValueError

    #resize frame to fit segmentation
    hSegments, wSegments = segmentImage(anchorFrame, blockSize)
    anchorFrame = cv2.resize(anchorFrame, (int(wSegments*blockSize), int(hSegments*blockSize)))
    targetFrame = cv2.resize(targetFrame, (int(wSegments*blockSize), int(hSegments*blockSize)))

    return (anchorFrame, targetFrame)

def main(anchorFrame, targetFrame, outfile="OUTPUT", saveOutput=False, blockSize = 16):

    anchorFrame, targetFrame = preprocess(anchorFrame, targetFrame, blockSize) 

    predictedFrame = blockSearchBody(anchorFrame, targetFrame, blockSize)
    residualFrame = getResidual(targetFrame, predictedFrame)

    residualMetric = getResidualMetric(residualFrame)

    return residualMetric
