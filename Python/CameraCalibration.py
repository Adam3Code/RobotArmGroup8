"""
Camera calibration loop, used to find the intrinsic and extrinsic parameters.
"""

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

class CameraCalibration:
    def __init__(self):
        self.nb_vertical = 6
        self.nb_horizontal = 8

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((self.nb_horizontal*self.nb_vertical,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:self.nb_vertical,0:self.nb_horizontal].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.
        self.imgs = []

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    def getCorners(self):
        img = glob.glob('imgs/*.png')
        print("Found {} images for calibration.".format(len(img)))

        for fname in img:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (self.nb_vertical,self.nb_horizontal), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                self.objpoints.append(self.objp)

                # Refine the corner positions
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1), self.criteria)
                self.imgpoints.append(corners2)
                self.imgs.append(img)

        cv2.destroyAllWindows()


    def calibrateCamera(self):
        if len(self.objpoints) == 0 or len(self.imgpoints) == 0:
            print("No object points or image points found. Cannot calibrate camera. Run first getCorners() to find corners in images.")
            return None, None, None, None, None

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.imgs[0].shape[1::-1], None, None)
        np.savez('calibration_data.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
        print("Calibration data saved to calibration_data.npz")


    def undistortImage(self, img_path):
        # Load calibration data
        with np.load('calibration_data.npz') as data:
            mtx = data['mtx']
            dist = data['dist']

        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        # Undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # Crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        return dst


    def computeReprojectionError(self):
        if len(self.objpoints) == 0 or len(self.imgpoints) == 0:
            print("No object points or image points found. Cannot compute reprojection error.")
            return None

        with np.load('calibration_data.npz') as data:
            mtx = data['mtx']
            dist = data['dist']
            rvecs = data['rvecs']
            tvecs = data['tvecs']

        total_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            imgpoints2 = imgpoints2.reshape(-1, 2)  # Reshape to (N, 2)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            total_error += error

        mean_error = total_error/len(self.objpoints)
        print("Mean reprojection error: {}".format(mean_error))
        return mean_error

    
    def visualizeImages(self):
        if len(self.imgs) == 0:
            print("No images to visualize.")
            return
        
        for img in self.imgs:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (self.nb_vertical,self.nb_horizontal), None)
            if ret == True:
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1), self.criteria)
                cv2.drawChessboardCorners(img, (self.nb_vertical,self.nb_horizontal), corners2, ret)
                cv2.imshow('img', img)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
                else:
                    continue
            else:
                print("Chessboard corners not found in image.")
                continue
        cv2.destroyAllWindows()


    def captureImages(self):
        cap = cv2.VideoCapture(0)
        count = 0
        while True:
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                cv2.imwrite(f'imgs/calibration_image_{count}.png', frame)
                count += 1
            elif cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrator = CameraCalibration()
    calibrator.getCorners()
    calibrator.computeReprojectionError()
    calibrator.visualizeImages()