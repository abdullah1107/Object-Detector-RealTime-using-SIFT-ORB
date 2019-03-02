#run command: python object_detection_using_shift.py
import cv2
import matplotlib.pyplot as plt
import numpy as np



def sift_detector(new_image, image_template):
    #Function that compares input image to template
    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    image2 = image_template


    #create a SIFT detector object
    #sift = cv2.SIFT()
    sift = cv2.xfeatures2d.SIFT_create()
    # Obtain the keypoints and descriptors using SIFT
    keypoints_1, descriptors_1 = sift.detectAndCompute(image1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    # Create the Flann Matcher object
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors_1,descriptors_2,k=2)

    good_matches = []

    #ratio test
    for i,(match1,match2) in enumerate(matches):
        if match1.distance < 0.7*match2.distance:
            good_matches.append(match1)

    return len(good_matches)



#Running Camera
cap = cv2.VideoCapture(0)
# Load our image template, this is our reference image
image_template = cv2.imread('images/girl.jpg', 0)

while(True):

        ret, frame = cap.read()

        #Get height and width of webcam frame
        height, width = frame.shape[:2]

        # Define ROI Box Dimensions
        top_left_x = width / 3
        top_left_y = (height / 2) + (height / 4)
        bottom_right_x = (width / 3) * 2
        bottom_right_y = (height / 2) - (height / 4)

        #draw a rectangular window
        cv2.rectangle(frame, (int(top_left_x),int(top_left_y)), (int(bottom_right_x),int(bottom_right_y)), 255, 3)
        #Crop window of observation we defined above
        #cropped = frame[bottom_right_y:top_left_y , top_left_x:bottom_right_x]
        cropped = frame[int(bottom_right_y):int(top_left_y) , int(top_left_x):int(bottom_right_x)]

        #Flip frame orientation horizontally
        frme = cv2.flip(frame, 1)

        #Get number of SIFT matches
        matches = sift_detector(cropped, image_template)
        print(" Matches:",matches)
        # Display status string showing the current no. of matches
        cv2.putText(frame,str(matches),(450,450), cv2.FONT_HERSHEY_COMPLEX, 2,(0,255,0),1)
        # Our threshold to indicate object deteciton
        # We use 10 since the SIFT detector returns little false positves
        threshold = 10

        # If matches exceed our threshold then object has been detected
        if matches > threshold:
            cv2.rectangle(frame, (int(top_left_x),int(top_left_y)), (int(bottom_right_x),int(bottom_right_y)), (0,255,0), 3)
            cv2.putText(frame,'Object Found',(50,50), cv2.FONT_HERSHEY_COMPLEX, 2 ,(0,255,0), 2)


        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
