import cv2
import numpy as np
import matplotlib.pyplot as plt


# Constrains points to be inside boundary.
def constraintPoint(p, w, h):
	'''
	The function has been defined from the following observations.
	 - When face moves outside the boundary through:
	   -- left side, X1 reaches max limit of int32.
	   -- right side, X2 increase linearly. Does not exceed double the width.
	   -- top, Y1 reaches max limit of int32.
	   -- bottom, Y2 increases linearly. Does not exceed double the height. 
	 - 
	'''
	# Convert to a mutable list.
	p = list(p)
	if p[0] > 2 * w:
		p[0] = 1 
	elif w < p[0] < 2 * w:
		p[0] = w - 1
	elif p[1] > 2 * h:
		p[1] = 1
	elif h < p[1] < 2 * h:
		p[1] = h - 1
	return p


def extractFaceAndFeatures(img):
	'''
	The feature extraction process is accomplished in three stages.
	 - Face detection.
	 - Face align and crop.
	 - Feature extraction
	'''
	# Size.
	h, w = img.shape[:2]

	# Set detector input size.
	detector.setInputSize((w, h))

	# Start tickmeter.
	tm.start()

	# Perform detection.
	faces = detector.detect(img)

	# Stop tickmeter.
	tm.stop()

	if faces[1] is not None:
		# Align and crop faces.
		face_align = recognizer.alignCrop(img, faces[1][0])
		# Extract features.
		face_features = recognizer.feature(face_align)
		return faces, face_align, face_features
	else:
		print('Unable to read face')
		return None, None, None


def enrollFace(img):
	enrolled_faces.append(img)


def deListFace(img):
	# Check if the de-list frame matches with any frame in the enrolled_face. If yes, remove the face.
	_, img_crop, img_features = extractFaceAndFeatures(img)
	for im in enrolled_faces:
		_, im_crop, im_features = extractFaceAndFeatures(im)
		# Match features.
		cosine_score = recognizer.match(img_features, im_features, cv2.FaceRecognizerSF_FR_COSINE)
		# Remove the enrolled face if de-list frame matches.
		if cosine_score >= cosine_similarity_thresh:
			enrolled_faces.remove(im)
		else:
			print('Face not enrolled previously')


def displayBoundingBox(img, landmarks, thickness=2, color=(0, 255, 0)):
	'''
	As the name suggests, the function renders bounding box around the detected face. If the face is enrolled,
	green bounding box is displayed; red otherwise.
	'''
	if landmarks[1] is not None:
		for idx, landmark in enumerate(landmarks[1]):
			coords = landmark[:-1].astype(np.uint32)
			tlc = (coords[0], coords[1])
			brc = (coords[0]+coords[2], coords[1]+coords[3])
			# Take care of the boundary condition.
			tlc = constraintPoint(tlc, img.shape[1], img.shape[0])
			brc = constraintPoint(brc, img.shape[1], img.shape[0])
			cv2.rectangle(img, tlc, brc, color, thickness)


def overlayInfoAndLock(img, logo, pos=(540, 10)):
  """
  This function overlays the image of lock/unlock if the authentication of the input frame is successful/failed.
  Along with the necessary run time informations.
  """

  # Resize the logo.
  logo = cv2.resize(logo, None, fx=2, fy=2)
  # Offset value for the image of the lock/unlock.
  symbol_x_offset = img.shape[1] - logo.shape[1] - 50
  symbol_y_offset = 50
 
  # Find top left and bottom right coordinates.
  # where to place the lock/unlock image.
  y1, y2 = symbol_y_offset, symbol_y_offset + logo.shape[0]
  x1, x2 = symbol_x_offset, symbol_x_offset + logo.shape[1]

  # Scale down alpha channel between 0 and 1.
  mask = logo[:, :, 3]/255.0
  # Inverse of the alpha mask
  inv_mask = 1-mask
 
  # Iterate over the 3 channels - R, G and B.
  for c in range(0, 3):
    # Add the lock/unlock image to the frame.
    img[y1:y2, x1:x2, c] = (mask * logo[:, :, c] + inv_mask * img[y1:y2, x1:x2, c])

  # Display Info.
  cv2.putText(img, 'Press E to enroll face',  (10, 30), font, 0.7, blue, 2)
  cv2.putText(img, 'Press D to De-list Face', (10, 60), font, 0.7, blue, 2)
  cv2.putText(img, 'Press Q to Quit',         (10, 90), font, 0.7, blue, 2)


def matchAndAuthenticate(img, fc_landmarks, fc_crop, fc_features):
	'''
	Once we have the face detection matrix, the cropped image and the features of both the images; we can 
	perform face recognition. In this function, we are comparing the enrolled face (or list of enrolled faces) 
	to the current frame. If match is found, the found_match list is populated with 1s.
	'''
	# List to confirm face match.
	found_match = [0]
	
	if (enrolled_faces is not None) and (fc_crop is not None):
		for im in enrolled_faces:
			_, im_crop, im_features = extractFaceAndFeatures(im)

			# Match features.
			cosine_score = recognizer.match(fc_features, im_features, cv2.FaceRecognizerSF_FR_COSINE)

			if cosine_score >= cosine_similarity_thresh:
				# If match is found append 1 to found_match list.
				found_match.append(1)

		# If any one face matches, authenticate.
		if 1 in found_match:
			print('Authorized')
			# Draw green bounding box.
			displayBoundingBox(img, fc_landmarks)
			# Display unlock sign.
			overlayInfoAndLock(img, unlocked_img)
		else:
			print('Unauthorized')
			# Draw red bounding box.
			displayBoundingBox(frame, fc_landmarks, 2, (0, 0, 255))
			# Lock sign display.
			overlayInfoAndLock(img, locked_img)
	else:
		print('No face detected')



#----------------------------------------GLOBAL CONSTS & INITIALIZATIONS------------------------------------------#
# Color.
blue = (200, 150, 0)
# Font.
font = cv2.FONT_HERSHEY_SIMPLEX

# List to store enrolled faces.
enrolled_faces = []

# Detector parameters(default).
score_threshold = 0.9
nms_threshold = 0.3
top_k = 5000

# Similarity threshold.
cosine_similarity_thresh = 0.363

# Model paths.
face_detection_model = 'face_detection_yunet_2021dec.onnx'
face_recognition_model = 'face_recognition_sface_2021dec.onnx'

# Create face detector object.
detector = cv2.FaceDetectorYN.create(face_detection_model, "", (320, 320), score_threshold, nms_threshold, top_k)

# Create face recognizer object.
recognizer = cv2.FaceRecognizerSF.create(face_recognition_model, "")

# Load lock/unlock images.
locked_img = cv2.imread('logo-locked.png', -1)
unlocked_img = cv2.imread('logo-unlocked.png', -1)

# Init video capture object.
cap = cv2.VideoCapture(0)

# Tickmeter.
tm = cv2.TickMeter()
#-----------------------------------------------------------------------------------------------------------------#


if __name__ == '__main__':

	while(True):
		ret, frame = cap.read()

		if not ret:
			print('Unable to read frames')
			break
			
		# Flip the image.
		frame = cv2.flip(frame, 1)

		# Extract cropped face and features of current frame.
		face_landmarks, face_crop, face_features = extractFaceAndFeatures(frame)

		# Perform face matching and authentication.
		matchAndAuthenticate(frame, face_landmarks, face_crop, face_features)

		# Show FPS.
		fps = 'FPS : {}'.format(int(tm.getFPS()))
		org = (10, 120)
		cv2.putText(frame, fps, org, font, 0.7, (0,150,0), 2)

		# Display.
		cv2.imshow('Output', frame)
		key = cv2.waitKey(1)
		if key == ord('q'):
			break
		elif key == ord('e'):
			enrollFace(frame)
		elif key == ord('d'):
			deListFace(frame)

	cap.release()
	cv2.destroyAllWindows()