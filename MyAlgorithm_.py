#from gevent.select import select
from sensors import sensor
import numpy as np
import threading
import math

import cv2

class MyAlgorithm():

    def __init__(self, sensor):
        self.sensor = sensor
        self.imageRight=None
        self.imageLeft=None
        self.lock = threading.Lock()
        self.count = 0
        self.countTime = 0

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

    def obtainCentroid (self, image, name):
        
        imgFinal = image.copy()

        ROI_inf_gray = self.colorFilter(image)

        ROI_inf_gray = self.img2gray_normalize(ROI_inf_gray)

        kernel = np.ones((20,20),np.uint8)
        erosion = cv2.erode(ROI_inf_gray,kernel,iterations = 1)

        dibujo = erosion.copy()

        ret,thresh = cv2.threshold(dibujo,127,255,0)

        contours,hierarchy = cv2.findContours(thresh, 1, 2)
	
        stop = 0
        centroid = np.array ([-1, -1])
        # Si detectamos contornos
        if contours:
            cnt = contours[0]
            M = cv2.moments(cnt)

            # Evitar division por cero
            if M['m00'] != 0:

                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

                ancho = np.size(dibujo, 1)
                largo = np.size(dibujo, 0)
                
                #print "ancho: ", ancho, "largo: ", largo
                #print "cx: ", cx, "cy: ", cy

                if (cx>0 & cx<ancho):   
                    if (cy>0 & cy<largo):
                        radius = 5
                        color = (0,255,0)
#                        cv2.circle(dibujo, (cx,cy), radius, color, -1)
                        cv2.circle(imgFinal, (cx,cy), radius, color, -1)
                        centroid = np.array ([cx, cy])
                else:
                    centroid = np.array ([-1, -1])
    
#                        if (cx>300 & cx<340):
#                            print "estoy", "cx: ", cx
#                            stop = 1
#                        else:
#                            stop = 0
#
#        cv2.imshow(name, dibujo)
        name2 = name + "_cp"
        
        # Muestra las secciones de la imagen con los centroides
        # cv2.imshow(name2, imgFinal)
        
        return centroid
#
#        self.sensor.setV(0)
#        self.sensor.setW(0.05)
    
    def execute(self):
        #GETTING THE IMAGES
        imageLeft = self.sensor.getImageLeft()
        #imageRight = self.sensor.getImageRight()

        # Filtramos el color rojo de la imagen
        filteredImage = self.colorFilter(imageLeft)

        section_1 = imageLeft [420:480,:]
        section_2 = imageLeft [360:420,:]
        section_3 = imageLeft [300:360,:]
        section_4 = imageLeft [240:300,:]
        
        section_1_c = self.obtainCentroid (section_1, "section_1")
        section_2_c = self.obtainCentroid (section_2, "section_2")
        section_3_c = self.obtainCentroid (section_3, "section_3")
        section_4_c = self.obtainCentroid (section_4, "section_4")
        
        radius = 5
        green_color  = (0,255,0)
        red_color    = (0,0,255)
        blue_color   = (255,0,0)
        yellow_color = (0,255,255)
        
        centroid_1 = np.array ([-1, -1])
        centroid_2 = np.array ([-1, -1])
        centroid_3 = np.array ([-1, -1])
        centroid_4 = np.array ([-1, -1])
        if (section_1_c[0] != -1):       
            cv2.circle(section_1, (section_1_c[0],section_1_c[1]), radius, green_color, -1)
            # Calculamos la posicion de los centroides en la imagen real
            cx = section_1_c[0]
            cy = section_1_c[1] + 180
            centroid_1 = np.array ([cx, cy])
    
        if (section_2_c[0] != -1):           
            cv2.circle(section_2, (section_2_c[0],section_2_c[1]), radius, red_color, -1)
            # Calculamos la posicion de los centroides en la imagen real
            cx = section_2_c[0]
            cy = section_2_c[1] + 120
            centroid_2 = np.array ([cx, cy])
        if (section_3_c[0] != -1):          
            cv2.circle(section_3, (section_3_c[0],section_3_c[1]), radius, blue_color, -1)
            # Calculamos la posicion de los centroides en la imagen real
            cx = section_3_c[0]
            cy = section_3_c[1] + 60
            centroid_3 = np.array ([cx, cy])
        if (section_4_c[0] != -1):         
            cv2.circle(section_4, (section_4_c[0],section_4_c[1]), radius, yellow_color, -1)
            # Calculamos la posicion de los centroides en la imagen real
            centroid_4 = section_4_c
            
        
        new_img = np.concatenate((section_4, section_3), axis=0) 
        new_img = np.concatenate((new_img, section_2), axis=0)
        new_img = np.concatenate((new_img, section_1), axis=0)
        
        
        #print "centroid_1", centroid_1, "centroid_2", centroid_2
        #print "centroid_3", centroid_3, "centroid_4", centroid_4
        
        if ((centroid_1[0] == -1) | (centroid_2[0] == -1) | (centroid_3[0] == -1)):
            ang_dif = -1
        elif ((centroid_1[0]==centroid_2[0]) | (centroid_1[0]==centroid_3[0])):
            aux12 = (centroid_1[0]-centroid_2[0])/float(centroid_1[1]-centroid_2[1])
            aux13 = (centroid_1[0]-centroid_3[0])/float(centroid_1[1]-centroid_3[1])
            
            ang_12 = math.degrees(math.atan(aux12))
            ang_13 = math.degrees(math.atan(aux13))
            
            # Si el ang_13 > ang_12 el giro es positivo (izquierda)
            ang_dif = ang_13 - ang_12
            
            #print "aux12", aux12, "ang_12", ang_12, "aux13", aux13, "ang_13", ang_13
            
        print "centroid_1", centroid_1[0]
        if (centroid_1[0] == -1):
            self.sensor.setV(0.0)
            self.sensor.setW(-0.03)
        elif ((centroid_1[0]>380) & (centroid_1[0]<420)):
            self.sensor.setV(0.2)
            self.sensor.setW(0.0)
        elif (centroid_1[0]<380):
            self.sensor.setV(0.005)
            self.sensor.setW(0.03)
        elif (centroid_1[0]>420):
            self.sensor.setV(0.005)
            self.sensor.setW(-0.03)
        else:
            self.sensor.setV(0.0)
            self.sensor.setW(-0.03)
        
        cv2.imshow ("prueba", new_img)
        
        self.countTime = self.countTime + 1        
        
#==============================================================================
#         print "countTime: ", self.countTime
#         
#         #Damos una vuelta 
#         if (self.countTime < 180):
#             self.sensor.setV(0)
#             self.sensor.setW(0.0)
#         else:
#             self.sensor.setW(0.0)
#==============================================================================
        
#        if (stop == 1):
#            self.sensor.setW(0.0)
#        else:
#            self.sensor.setW(0.05)

        #SHOW THE FILTERED IMAGE ON THE GUI
        self.setRightImageFiltered(imageLeft)
        self.setLeftImageFiltered(filteredImage)

        self.count = self.count + 1

        # 480, 640
        # alto, ancho = filteredImageGray.shape
        # filteredImageGray[480,640]
        # width, height, channels = filteredImageGray.shape

        # print filteredImageGray

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

class PID:
    def __init__(self, P=2.0, I=0.0, D=1.0, Derivator=0, Integrator=0, Integrator_max=500, Integrator_min=-500):
        self.Kp=P
        self.Ki=I
        self.Kd=D
        self.Derivator=Derivator
        self.Integrator=Integrator
        self.Integrator_max=Integrator_max
        self.Integrator_min=Integrator_min

        self.set_point=0.0
        self.error=0.0

    def update(self,current_value):
        """	Calculate PID output value for given reference input and feedback	"""
        self.error = self.set_point - current_value

        self.P_value = self.Kp * self.error
        self.D_value = self.Kd * (self.error - self.Derivator)
        self.Derivator = self.error

        self.Integrator = self.Integrator + self.error

        if self.Integrator > self.Integrator_max:
            self.Integrator = self.Integrator_max

        elif self.Integrator < self.Integrator_min:
            self.Integrator = self.Integrator_min

        self.I_value = self.Integrator * self.Ki

        PID = self.P_value + self.I_value + self.D_value

        return PID

    def setPoint(self,set_point):
        """	Initilize the setpoint of PID	"""
        self.set_point = set_point
        self.Integrator=0
        self.Derivator=0

    def setIntegrator(self, Integrator):
        self.Integrator = Integrator

    def setDerivator(self, Derivator):
        self.Derivator = Derivator

    def setKp(self,P):
        self.Kp=P

    def setKi(self,I):
        self.Ki=I

    def setKd(self,D):
        self.Kd=D

    def getPoint(self):
        return self.set_point

    def getError(self):
        return self.error

    def getIntegrator(self):
        return self.Integrator

    def getDerivator(self):
        return self.Derivator
