#from gevent.select import select
from sensors import sensor
import numpy as np
import threading

import cv2


class MyAlgorithm():

    def __init__(self, sensor):
        self.sensor = sensor
        self.imageRight=None
        self.imageLeft=None
        self.lock = threading.Lock()
        self.count = 0

    def cosa():
        px_dcha = 0
        px_izqda = 0

        if (ROI_inf_gray[239,320]>0):
            line = 1
            self.sensor.setW(0.0)

            primero = 1
            ultimo = 1
            j = ancho - 1
            #print ancho
            for i in range(ancho):
                print ROI_inf_gray[1,i]
                if (ROI_inf_gray[0,i] > 0 & primero==1):
                    px_dcha = i
                    print px_dcha
                    primero = 0
                if (ROI_inf_gray[0,j] > 0 & ultimo==1):
                    px_izqda = j
                    print px_izqda
                    ultimo = 0
                j = j - 1

            #print px_dcha, px_izqda

            lado = 10
            cv2.rectangle(final,(px_dcha,alto),(px_dcha+lado,alto-lado),(0,0,255),-1)
            cv2.rectangle(final,(px_izqda,alto),(px_izqda+lado,alto-lado),(0,0,255),-1)

            # final [0:5,i:i+5] = [0,0,150]
            # final [0:5,j:j+5] = [0,0,150]

        else:
            self.sensor.setV(0)
            self.sensor.setW(0.05)

    def colorFilter(self, image):
        # made all red pixels = 0
        # img[:,:,2]=0

        # Convert BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # define range of blue color in HSV
        lower_red = np.array([0,100,0])
        upper_red = np.array([10,255,255])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # planes = cv2.split(imageRight)
        #
        # cv2.imshow("split 1", planes[0])
        # cv2.imshow("split 2", planes[1])
        # cv2.imshow("split 3", planes[2])

        # Bitwise-AND mask and original image
        return cv2.bitwise_and(hsv,hsv, mask = mask)

    def img2gray_normalize (self, image):

        filteredImageGray = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        filteredImageGray = cv2.cvtColor(filteredImageGray, cv2.COLOR_RGB2GRAY)

        ret, mask = cv2.threshold(filteredImageGray, 10, 255, cv2.THRESH_BINARY)
        return cv2.bitwise_not(filteredImageGray,filteredImageGray, mask = mask)


    def execute(self):
        #GETTING THE IMAGES
        imageLeft = self.sensor.getImageLeft()
        #imageRight = self.sensor.getImageRight()


        # Add your code here
        #print "Runing"

        # #EXAMPLE OF HOW TO SEND INFORMATION TO THE ROBOT ACTUATORS
        # self.sensor.setV(0)
        # self.sensor.setW(0.05)

        # Filtramos el color rojo de la imagen
        filteredImage = self.colorFilter(imageLeft)

        grayNorm = self.img2gray_normalize(filteredImage)

        # cv2.imshow("img", grayNorm)


        ROI_sup = imageLeft [0:240,:]
        ROI_inf = imageLeft [240:480,:]

        ROI_inf_gray = self.colorFilter(ROI_inf)

        ROI_inf_gray = self.img2gray_normalize(ROI_inf_gray)

        #ROI_inf_izq = imageLeft [240:480,0:320]
        #ROI_inf_dcha = imageLeft [240:480,320:640]

        kernel = np.ones((20,20),np.uint8)
        erosion = cv2.erode(ROI_inf_gray,kernel,iterations = 1)

	#cv2.imshow("erosion", erosion)
	
	dibujo = erosion.copy()
	
	ret,thresh = cv2.threshold(dibujo,127,255,0)
	
	#cv2.imshow("thresh", thresh)
	
	contours,hierarchy = cv2.findContours(thresh, 1, 2)
	print type(contours)
 
	if contours:
		print "List is empty"

		cnt = contours[0]
	
		M = cv2.moments(cnt)
 		
 		if M['m00'] != 0:
 			print "M['m00']:  ",M['m00']
 		  
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])
	
			ancho = np.size(dibujo, 0)
			largo = np.size(dibujo, 0)

			imgFinal = imageLeft.copy()

			if (cx > 0 & cx < ancho & cy > 0 & cy < largo):
				radius = 5
				color = (0,255,0)
				cv2.circle(dibujo, (cx,cy), radius, color, -1)
				cv2.circle(imgFinal, (cx,cy+240), radius, color, -1)

			cv2.imshow("dibujo", dibujo)

			cv2.imshow("imgFinal", imgFinal)

        #cv2.imshow("ROI_inf", ROI_inf_gray)

        

        alto, ancho, _z = imageLeft.shape

        final = imageLeft

	self.sensor.setV(0)
        self.sensor.setW(0.05)




        # cv2.imshow("final", final)

        #SHOW THE FILTERED IMAGE ON THE GUI
        self.setRightImageFiltered(imageLeft)
        self.setLeftImageFiltered(filteredImage)

        self.count = self.count + 1

        # 480, 640
        # alto, ancho = filteredImageGray.shape
        # filteredImageGray[480,640]
        #width, height, channels = filteredImageGray.shape

        #print filteredImageGray

        # for i in range(alto):
        #     for j in range (ancho):
        #         if (filteredImageGray[i,j]>0):
        #             filteredImageGray[i,j] = 255



    def setRightImageFiltered(self, image):
        self.lock.acquire()
        self.imageRight=image
        self.lock.release()


    def setLeftImageFiltered(self, image):
        self.lock.acquire()
        self.imageLeft=image
        self.lock.release()

    def getRightImageFiltered(self):
        self.lock.acquire()
        tempImage=self.imageRight
        self.lock.release()
        return tempImage

    def getLeftImageFiltered(self):
        self.lock.acquire()
        tempImage=self.imageLeft
        self.lock.release()
        return tempImage
