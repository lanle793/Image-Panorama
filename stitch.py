import cv2
import sys
import os
import glob
import numpy as np
import random
import imutils





# find distance using SSD
def distance(vec1, vec2):
    dif = np.subtract(vec1, vec2)
    sqDif = np.square(dif)
    return np.sum(sqDif)
    

# match descriptor using ratio test and return 2 lists of indices
def matchDescriptor(des1, des2):
    id1 = []
    id2 = []
    
    for i in range(len(des1)):
        vec1 = des1[i]
        matches = {}
        for j in range(len(des2)):
            vec2 = des2[j]
            dis = distance(vec1, vec2)
            matches[j] = dis
        matches = {k: v for k, v in sorted(matches.items(), key=lambda item: item[1])}
        disList = list(matches.values())
        if disList[0]/disList[1] < 0.8:
            id1.append(i)
            id2.append(list(matches.keys())[0])
            
    return id1, id2


# count number of matches between 2 images
def getMatches(img1, img2, sift):
    duplicate = False
    
    # convert images into grayscale images and compute SIFT descriptors
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)
    
    # find descriptor matches/corresponding points
    id1, id2 = matchDescriptor(des1, des2)
    
    # 2 images are a good match if there is at least 20% match
    # and they are not duplicates
    if len(id2) < (len(kp2) * 0.20) and len(id1) < (len(kp1) * 0.20):
        return None, None, False, False
    
    if len(id2) == len(kp2):
        duplicate = True
    
    pts1 = []
    pts2 = []
    
    for idc in id1:
        pts1.append(kp1[idc])

    for idx in id2:
        pts2.append(kp2[idx])
    
    return pts1, pts2, True, duplicate


# compute homography estimate from corresponding points
def computeHomography(p1, p2):
    # assemble matrix from pairs
    P = np.zeros((2*len(p1),9))
    for i in range(len(p1)):
        x = p1[i].pt[0]
        y = p1[i].pt[1]
        u = p2[i].pt[0]
        v = p2[i].pt[1]
        P[2*i,:] = [x, y, 1, 0, 0, 0, -u*x, -u*y, -u]
        P[2*i+1,:] = [0, 0, 0, x, y, 1, -v*x, -v*y, -v]
    
    # compute homography
    U, S, V = np.linalg.svd(P)
    H = np.reshape(V[8], (3,3))
    H = np.divide(H, H.item(8))
    
    return H


# get homography from a set of matching points using RANSAC
def getRansacHomography(srcPts, desPts, iters, threshold, require):
    maxInliers = 0
    finalH = None
    numPts = len(srcPts)
    
    for i in range(iters):
        # select 4 random pairs to compute homography
        p1 = []
        p2 = []
        index1 = random.randrange(0, numPts)
        p1.append(srcPts[index1])
        p2.append(desPts[index1])
        index2 = random.randrange(0, numPts)
        p1.append(srcPts[index2])
        p2.append(desPts[index2])
        index3 = random.randrange(0, numPts)
        p1.append(srcPts[index3])
        p2.append(desPts[index3])
        index4 = random.randrange(0, numPts)
        p1.append(srcPts[index4])
        p2.append(desPts[index4])
        
        H = computeHomography(p1, p2)
        inliers = 0
        
        # find the number of inliers for the current homography
        for i in range(numPts):
            srcPt = np.transpose(np.matrix([srcPts[i].pt[0], srcPts[i].pt[1], 1]))
            desPt = np.transpose(np.matrix([desPts[i].pt[0], desPts[i].pt[1], 1]))
            estPt = np.matmul(H, srcPt)
            if estPt.item(2) != 0:
                estPt = estPt/estPt.item(2)
            d = distance(desPt, estPt)
            if d < threshold:
                inliers = inliers + 1
                
        if inliers > maxInliers:
            maxInliers = inliers
            finalH = H
            
        if inliers > (numPts * require):
            break
        
    return finalH
    

# determine images' relative order
def findImageOrder(img1, img2, pts1, pts2):
    stitchFromRight = True
    
    # get homography for image transform
    H = getRansacHomography(pts2, pts1, 1000, 5, 0.97)
    
    # check for negative coordinates
    corner1 = np.transpose(np.matrix([0, 0, 1]))
    corner2 = np.transpose(np.matrix([0, img2.shape[0], 1]))
    newCorner1 = np.matmul(H, corner1)
    newCorner2 = np.matmul(H, corner2)
    if newCorner1.item(0) < 0 or newCorner2.item(0) < 0:
        stitchFromRight = False
        H = getRansacHomography(pts1, pts2, 1000, 5, 0.97)
        
    return H, stitchFromRight
    
    
# crop image to remove blank space
def cropImg(img):
    # finds contours from the image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # get the maximum contour area
    c = max(contours, key=cv2.contourArea)

    # get a rectangular box from the contour area
    x, y, w, h = cv2.boundingRect(c)

    # crop the image to the box coordinates
    img = img[y:y + h, x:x + w]
    
    return img


# stitch 2 images together
def stitchImages(img1, img2, H):
    width = img1.shape[1] + img2.shape[1]
    height = img1.shape[0] + img2.shape[0]
    
    result = cv2.warpPerspective(img2, H, (width, height))
    result[0:img1.shape[0], 0:img1.shape[1]] = img1                

    # crop result image
    result = cropImg(result)
    
    return result




# read images from data folder
folder = sys.argv[1]
direct = os.path.abspath(folder)
pattern = os.path.join(direct, '*.jpg')
fileList = glob.glob(pattern)


# create image list
imgList = []
for file in fileList:
    img = cv2.imread(file)
    imgList.append(img)


# initialize SIFT detector
sift = cv2.xfeatures2d.SIFT_create()


# sort images from left to right
hList = []
sortedImgList = []

# find the first image pair
while len(imgList) > 1:
    img1 = imgList[0]
    
    # find an image to match with the first image
    for i in range(1, len(imgList)):
        img2 = imgList[i]
        pts1, pts2, goodMatch, duplicate = getMatches(img1, img2, sift)
        
        if duplicate:
            del imgList[i]
            continue
        
        if goodMatch:
            # determine image order
            H, stitchFromRight = findImageOrder(img1, img2, pts1, pts2)
            
            if stitchFromRight:
                sortedImgList.append(img1)
                sortedImgList.append(img2)
            else:
                sortedImgList.append(img2)
                sortedImgList.append(img1)
            hList.append(H)
            
            del imgList[i]
            del imgList[0]
            break
    
    if len(sortedImgList) == 0:
        del imgList[0]
    else:
        break


# add the rest of the image to sorted list
match = False
while len(imgList) > 0:
    leftMost = sortedImgList[0]
    
    # find an image to match with the most leftward image
    for i in range(len(imgList)):
        img = imgList[i]
        pts1, pts2, goodMatch, duplicate = getMatches(leftMost, img, sift)
        
        if duplicate:
            del imgList[i]
            continue
        
        if goodMatch:
            match = True
            sortedImgList.insert(0, img)
            H = getRansacHomography(pts1, pts2, 1000, 5, 0.97)
            hList.insert(0, H)
            del imgList[i]
            break
            
    if not match:
        break

            
match = False
while len(imgList) > 0:
    rightMost = sortedImgList[len(sortedImgList)-1]
    
    # find an image to match with the most rightward image
    for i in range(len(imgList)):
        img = imgList[i]
        pts1, pts2, goodMatch, duplicate = getMatches(img, rightMost, sift)
        
        if duplicate:
            del imgList[i]
            continue
        
        if goodMatch:
            match = True
            sortedImgList.append(img)
            H = getRansacHomography(pts1, pts2, 1000, 5, 0.97)
            hList.append(H)
            del imgList[i]
            break
            
    if not match:
        break
    
    
# stitch images together
for i in range(1, len(hList)):
    hList[i] = np.matmul(hList[i], hList[i-1])
    
final = sortedImgList[0]
for i in range(1, len(sortedImgList)):
    cur = sortedImgList[i]
    final = stitchImages(final, cur, hList[i-1])


# save final result panorama
savePath = os.path.join(direct, 'panorama.jpg')
cv2.imwrite(savePath, final)