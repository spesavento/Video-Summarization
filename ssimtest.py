import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
import motionvectors as mv
import math

video_name = 'soccer'

# jpg video frames to be analyzed - ordered frame0.jpg, frame1.jpg, etc.
frames_jpg_path = '../project_files/project_dataset/frames/'+video_name+'/'

average_dists = []
for i in range(10839, 10940, 1):
    frame_a = cv2.imread(frames_jpg_path+'frame' + str(i) + '.jpg')
    frame_b = cv2.imread(frames_jpg_path+'frame' + str(i+1) + '.jpg')
    frame_c = cv2.imread(frames_jpg_path+'frame' + str(i+2) + '.jpg')
    frame_d = cv2.imread(frames_jpg_path+'frame' + str(i+3) + '.jpg')

    # frame_a_bw = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    # frame_b_bw = cv2.cvtColor(frame_c, cv2.COLOR_BGR2GRAY)
    # mad = np.sum(np.abs(np.subtract(frame_a_bw, frame_b_bw)))/(frame_a_bw.shape[0])
    # mad = ssim(frame_a_bw, frame_b_bw)
    hist1 = []
    color = ('b','g','r')
    for j in range(3):
        histr = cv2.calcHist([frame_b],[j],None,[256],[0,256])
        hist1.append(histr)

    hist2 = []
    for j in range(3):
        histr = cv2.calcHist([frame_c],[j],None,[256],[0,256])
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
        max_val /= (180.0 * 320.0)
        # print(str(max_color) + ": " + str(max_val))
        hist1a_max.append([max_color, max_val])

        max_val = 0
        max_color = 0
        for k in range(len(hist2a[j])):
            if hist2a[j][k][0] > max_val:
                max_val = hist2a[j][k][0]
                max_color = k
        max_val /= (180.0 * 320.0)
        # print(str(max_color) + ": " + str(max_val))
        hist2a_max.append([max_color, max_val])
    
    CumulativeColorDiffs = 0

    for j in range(3):
        CumulativeColorDiffs += abs(hist1a_max[j][0] - hist2a_max[j][0])

    average_dist = average_dist / 3.0
    # average_dist, residual_frame = mv.main(frames_jpg_path+'frame' + str(i) + '.jpg', frames_jpg_path+'frame' + str(i+10) + '.jpg')
    print(i + 2)
    # print(CumulativeColorDiffs)
    if CumulativeColorDiffs < 20:
        SimilarColorCheck = True
    
    print(SimilarColorCheck)
    # average_dists.append(mad)

# plt.plot(average_dists)
# plt.show()

    frame_a_bw = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    frame_b_bw = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    frame_c_bw = cv2.cvtColor(frame_c, cv2.COLOR_BGR2GRAY)
    frame_d_bw = cv2.cvtColor(frame_d, cv2.COLOR_BGR2GRAY)

# # hist1 = cv.calcHist([frame_a_bw],[0],None,[256],[0,256])
# # hist2 = cv.calcHist([frame_b_bw],[0],None,[256],[0,256])

# # # frame_a_bw = frame_a
# # # frame_b_bw = frame_b
# # # frame_c_bw = frame_c
# # # frame_d_bw = frame_d

# # cv2.imshow('RGB Image',frame_c_bw )
# # cv2.waitKey(0)

# # # print(np.min(frame_c_bw))

# ssim_ab = ssim(frame_a_bw, frame_b_bw, multichannel=False, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255)
# ssim_bc = ssim(frame_b_bw, frame_c_bw, multichannel=False, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255)
# ssim_cd = ssim(frame_c_bw, frame_d_bw, multichannel=False, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255)

    ssim_ab = ssim(frame_a_bw, frame_b_bw)
    ssim_bc = ssim(frame_b_bw, frame_c_bw)
    ssim_cd = ssim(frame_c_bw, frame_d_bw)

    ssim_ab = round(ssim_ab, 3)
    ssim_bc = round(ssim_bc, 3)
    ssim_cd = round(ssim_cd, 3)

# # # print(ssim_ab)
# # print(ssim_bc)
# # # print(ssim_cd)
    
    # print(ssim_bc/ssim_ab)
    # print(ssim_bc/ssim_cd)
