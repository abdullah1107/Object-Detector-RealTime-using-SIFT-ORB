#run commmand: python object_detection_using_orb.py
import cv2
import matplotlib.pyplot as plt
import numpy as np



def ORB_detector(new_image, image_template):
    #Function that compares input image to template
    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    #image2 = image_template

    #Create ORB detector with 1000 keypoints with a scaling pyramid factor of 1.2
    orb = cv2.ORB_create(1000, 1.2)

    #detect keypoints of original images
    (kp1, des1) = orb.detectAndCompute(image1, None)

    # Detect keypoints of rotated image
    (kp2, des2) = orb.detectAndCompute(image_template, None)

    #Create matcher
    #Note we are no longer using Flannbased matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    #Do matching
    matches = bf.match(des1, des2)

    #Sort the matches based on distance. Least distance
    #is better
    matches = sorted(matches, key=lambda val:val.distance)

    return len(matches)


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
        matches = ORB_detector(cropped, image_template)
        print(" Matches:", matches)
        # Display status string showing the current no. of matches
        cv2.putText(frame,str(matches),(50,450), cv2.FONT_HERSHEY_COMPLEX, 2,(250,0,150),3)
        # Our threshold to indicate object deteciton
        # We use 10 since the SIFT detector returns little false positves
        threshold = 500

        # If matches exceed our threshold then object has been detected
        if matches > threshold:
            cv2.rectangle(frame, (int(top_left_x),int(top_left_y)), (int(bottom_right_x),int(bottom_right_y)), (0,255,0), 3)
            cv2.putText(frame,'Object Found',(50,50), cv2.FONT_HERSHEY_COMPLEX, 2 ,(0,255,0), 2)


        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
