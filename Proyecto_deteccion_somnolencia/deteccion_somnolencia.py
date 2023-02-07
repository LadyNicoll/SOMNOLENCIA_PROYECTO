# importar librerias
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import pygame
import argparse
import imutils
import time
import dlib
import cv2

def sound_alarm(path) :
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()

def eye_aspect_ratio(eye):
	#Calculo distancia euclidiana de los puntos verticales.
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	##Calculo distancia euclidiana de los puntos horizontales.
	C = dist.euclidean(eye[0], eye[3])
	#Calculo del aspecto del ojo (EAR)
	ear = (A + B) / (2.0 * C)
	#Retornar EAR
	return ear
 
#Construcción de argumentos
ap = argparse.ArgumentParser()
#argumento --shape-predictor
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
#argumento --alarm
ap.add_argument("-a", "--alarm", type=str, default="",
	help="path alarm .WAV file")
#argumento --webcam
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
args = vars(ap.parse_args())
 
#constantes
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 30

#contador
COUNTER = 0
ALARM_ON = False

#Iniciacion de dlib (HOG-based)
print("Cargando predictor de puntos de referencia facial")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

#Indices de los puntos faciales del ojo izq y der
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
print("Comenzando transmisión de video")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

while True:
	# escala de grises 
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)
	#detección de rostro
	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		#contador de parpadeo para activar alarma
		if ear < EYE_AR_THRESH:
			COUNTER += 1
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				if not ALARM_ON:
					ALARM_ON = True
					if args["alarm"] != "":
						t = Thread(target=sound_alarm,
							args=(args["alarm"],))
						t.deamon = True
						t.start()
				#Alarma
				cv2.putText(frame, "ALERTA DE SOMNOLENCIA!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# reiniciar contador
		else:
			COUNTER = 0
			ALARM_ON = False

		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	#mostar frame
	cv2.imshow("Frame", frame)
	#cv2.createButton('Salir', lambda: cv2.destroyAllWindows(), None, cv2.QT_PUSH_BUTTON, 1)
	key = cv2.waitKey(1) & 0xFF
 
	#tecla para salir "q"
	if key == ord("q"):
		break
cv2.destroyAllWindows()
vs.stop()