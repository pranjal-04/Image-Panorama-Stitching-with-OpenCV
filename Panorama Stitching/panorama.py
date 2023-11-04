import cv2
import streamlit as st
st.set_option('deprecation.showfileUploaderEncoding', False)
import numpy as np
import imutils
from PIL import Image

class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3(or_better=True)

    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)
        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None

        # otherwise, apply a perspective warp to stitch the images together
        (matches, H, status) = M
        result = cv2.warpPerspective(imageA, H,
                                     (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        # check to see if the keypoint matches should be visualized
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            # return a tuple of the stitched image and the visualization
            return (result, vis)
        # return the stitched image
        return result

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # check to see if we are using OpenCV 3.X
        if self.isv3:
          # detect and extract features from the image
          sift = cv2.SIFT_create()
          (kps, features) = sift.detectAndCompute(image, None)
        # otherwise, we are using OpenCV 2.4.X
        else:
          # detect keypoints in the image
          detector = cv2.FeatureDetector_create("SIFT")
          kps = detector.detect(gray)
          # extract features from the image
          extractor = cv2.DescriptorExtractor_create("SIFT")
          (kps, features) = extractor.compute(gray, kps)
        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])
        # return a tuple of keypoints and features
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                   ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                            reprojThresh)
            # return the matches along with the homography matrix and status of each matched point
            return (matches, H, status)

        # otherwise, no homography could be computed
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB
        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
          # only process the match if the keypoint was successfully
          # matched
          if s == 1:
            # draw the match
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        # return the visualization
        return vis



st.title("Image Stitching App")

# Sidebar
st.sidebar.title("Upload Images")

# Upload left and right images separately
left_image = st.sidebar.file_uploader("Upload left image", type=["jpg", "jpeg", "png"], accept_multiple_files=False )
if left_image is not None:
    left_image_pil = Image.open(left_image)
    left_image_np = np.array(left_image_pil)
    left_image_resized = imutils.resize(left_image_np, width=400)
    st.sidebar.subheader("Original Left Image")
    st.sidebar.image(left_image_pil, use_column_width=True)
right_image = st.sidebar.file_uploader("Upload right image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

if left_image is not None and right_image is not None:
    # Read and resize the uploaded images
    right_image_pil = Image.open(right_image)
    right_image_np = np.array(right_image_pil)
    right_image_resized = imutils.resize(right_image_np, width=400)
    
    st.sidebar.subheader("Original Right Image")
    st.sidebar.image(right_image_pil, use_column_width=True)
    # Stitch the left and right images together to create a panorama
    stitcher = Stitcher()
    (result, vis) = stitcher.stitch([left_image_resized, right_image_resized], showMatches=True)

    # Display the original left and right images and the stitched panorama

    if result is not None:
        st.subheader("Stitched Panorama Output:")
        st.image(result, use_column_width=True)
    else:
        st.error("Panorama stitching failed. Please ensure there are enough matching keypoints.")

    # Optionally, show matches (set showMatches=True in stitch method)
    # st.subheader("KeyPoint Matches")
    # st.image(vis, use_column_width=True)
else:
    st.warning("Please upload both left and right images for stitching.")