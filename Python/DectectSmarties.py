
import cv2
import numpy as np
import imutils

from matplotlib import pyplot as plt


# def captureImagesLive():
#         cap = cv2.VideoCapture(0)
#         count = 0
#         while True:
#             ret, frame = cap.read()
#             cv2.imshow('frame', frame)
#             if cv2.waitKey(1) & 0xFF == ord('c'):
#                 cv2.imwrite(f'imgs/smarties_image_{count}.png', frame)
#                 count += 1
#                 return frame, count - 1
            
#             elif cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         cap.release()
#         cv2.destroyAllWindows()
#         exit()

def undistortImage(count):
        # Load calibration data
        with np.load('calibration_data.npz') as data:
            mtx = data['mtx']
            dist = data['dist']

        img = cv2.imread(f'imgs/smarties_image_{count}.png')
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        # Undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # Crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        return dst


def getMask(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #since red wraps around we need two bounds
    #lower red bound
    lower_bound1 = np.array([15, 70, 40])
    upper_bound1 = np.array([16, 255, 255])  
    mask1 = cv2.inRange(image_hsv, lower_bound1, upper_bound1)

    #upper red bound
    lower_bound2 = np.array([170, 70, 40])    
    upper_bound2 = np.array([180, 255, 255])  
    mask2 = cv2.inRange(image_hsv, lower_bound2, upper_bound2)


    #combine the two bounds
    mask = cv2.bitwise_or(mask1, mask2)
    return mask

    #unncoment to display result 
    # plt.imshow(result, cmap="gray")


def erode(mask, iter = 3):
    mask_erode = mask.copy()
    mask_erode = cv2.erode(mask_erode, None, iterations = iter)
    return mask_erode

def dilate(mask_erode, iter = 10):
    mask_dilate = mask_erode.copy()
    mask_dilate = cv2.dilate(mask_dilate, None, iterations = iter )
    return mask_dilate

def getCentres(mask):
    thresh = mask.copy()
    #get contours
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    centres = []

    for i, cont in enumerate(contours):
        M = cv2.moments(cont) #get contour moments

        if M["m00"] != 0:  #check if the area is not 0
            x = int(M["m10"] / M["m00"])  #get x and y coord of centre
            y = int(M["m01"] / M["m00"])
            print(f"Object {i} Centre: ({x}, {y})")
            centres.append([x, y])
    return centres


def drawBox(mask, image):
    thresh = mask.copy()
    #get contours
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)


    for i, cont in enumerate(contours): 
        x, y, w, h = cv2.boundingRect(cont)
        print(f"Object {i} Bounding Box: x={x}, y={y}, w={w}, h={h}")
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("smarties detected", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    
        
if __name__ == "__main__":

    count = 0
    cap = cv2.VideoCapture(0)

    print("started video capture")
    
    while True:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.imwrite(f'imgs/smarties_image_{count}.png', frame)

            print("image captured, finding smarties")
            dst = undistortImage(count) 
            mask = getMask(dst)
            mask = erode(mask)
            mask = dilate(mask)
            # cv2.imshow('mask', mask)


            centres = getCentres(mask)

            print(f"centres: {centres}")

            # #for display in report
            drawBox(mask,  dst)
            count += 1
            

        
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    







   

    






            
    
