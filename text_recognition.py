#import packages
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse #for command line arguments
import cv2

def decode_predictions(scores, geometry):
	# get the num Rows and Cols from scores
	(numRows, numCols) = scores.shape[2:4]
	# initialize our bounding boxes and confidence score lists
	rects = []
	confidences = []

	# loop over scores and associated bounding box geometry
	for y in range(0, numRows):
		#extract scores
		scoresData = scores[0, 0, y]

		#extract geometric bounding box coordinates
		corner0 = geometry[0, 0, y]
		corner1 = geometry[0, 1, y]
		corner2 = geometry[0, 2, y]
		corner3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over columns of data
		for x in range(0, numCols):
			#ignore scores that are too low
			if scoresData[x] < args["min_confidence"]:
				continue

			#compute offset factor (feature maps 4x smaller than input image)
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			#extract and compute angle data
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			#compute bounding box dimensions
			height = corner0[x] + corner2[x]
			width = corner1[x] + corner3[x]

			#compute start and end (x, y) coords for bounding box
			#note boxes will account for rotated text, but box not rotated
			endX = int(offsetX + (cos * corner1[x]) + (sin * corner2[x]))
			endY = int(offsetY - (sin * corner1[x]) + (cos * corner2[x]))
			startX = int(endX - width)
			startY = int(endY - height)

			#add coordinates to rects
			rects.append((startX, startY, endX, endY))
			#add score to confidence list
			confidences.append(scoresData[x])

	#return tuple of bounding boxes and confidence scores
	return (rects, confidences)

"""
#argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="images/example_03.jpg", help="path to input image")
ap.add_argument("-east", "--east", type=str, default="frozen_east_text_detection.pb", help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5, help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320, help="nearest multiple of 32 for resized width")
ap.add_argument("-e", "--height", type=int, default=320, help="nearest multiple of 32 for resized height")
ap.add_argument("-p", "--padding", type=float, default=0.20, help="amount of padding to add to each border of ROI")
args = vars(ap.parse_args())
"""

##text detection
args = {
	"image": "images/example_02.jpg",
	"east": "frozen_east_text_detection.pb",
	"min_confidence": .5,
	"width": 320,
	"height": 320,
	"x-padding": .10,
	"y-padding": .12,
}

#load image and take dimensions
image = cv2.imread(args["image"])
orig = image.copy()
(orig_height, orig_width) = image.shape[:2]

#set new height and width, calculate ratio difference for bounding boxes
(new_height, new_width) = (args["width"], args["height"])
ratio_height = orig_height / float(new_height)
ratio_width = orig_width / float(new_width)

#resize images, get dimensions
image = cv2.resize(image, (new_width, new_height))
(height, width) = image.shape[:2]

#define two output layers for EAST detector model
#one for probabilities, one for bounding box coordinates

layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

#load EAST text detector (pre-trained)
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

#construct blob from image, use model to create two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (width, height),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
						
net.setInput(blob)
(scores, geometry) = net.forward(layer_names) #pass through neural network

#decode predictions
(rects, confidences) = decode_predictions(scores, geometry)
#apply non-maxima suppression (suppresses weak overlapping boxes)
boxes = non_max_suppression(np.array(rects), probs=confidences)

##text dectection done, recognition:

results = []

#pad boxes to create overlap
index = 0
for (startX, startY, endX, endY) in boxes:
	#calculate padding
	dX = int((endX - startX) * args["x-padding"])
	dY = int((endY - startY) * args["y-padding"])
	
	#apply padding to improve results
	startX = max(0, startX - dX)
	startY = max(0, startY - dY)
	endX = min(orig_width, endX + (dX * 2))
	endY = min(orig_height, endY + (dY * 2))

	boxes[index] = startX, startY, endX, endY
	
	index = index + 1

#combine boxes:
box_i = 0
while (box_i < len(boxes)):

	remove = []
	x1, y1, ex1, ey1 = boxes[box_i]

	for j in range(0, len(boxes)):
		x2, y2, ex2, ey2 = boxes[j]
		if (x1 == x2 and y1 == y2 and ex1 == ex2 and ey1 == ey2):
			continue

		if (x1 > ex2 or x2 > ex1):
			continue

		if (y1 > ey2 or y2 > ey1):
			continue

		#combine
		nx = min(x1, x2)
		ny = min(y1, y2)
		nex = max(ex1, ex2)
		ney = max(ey1, ey2)

		boxes[box_i] = nx, ny, nex, ney
		x1, y1, ex1, ey1 = nx, ny, nex, ney
		remove.append(j)

	remove = remove[::-1]
	
	#print(boxes)
	#print(remove)
	for val in remove:
		boxes = np.delete(boxes, val, 0)
	#print(boxes)
	if (len(remove) == 0):
		box_i = box_i + 1
	
#apply tesseract
for (startX, startY, endX, endY) in boxes:
	#scale bounding boxes to ratio
	startX = int(startX * ratio_width)
	startY = int(startY * ratio_height)
	endX = int(endX * ratio_width)
	endY = int(endY * ratio_height)
	
	"""
	#calculate padding
	dX = int((endX - startX) * args["x-padding"])
	dY = int((endY - startY) * args["y-padding"])

	#apply padding to improve results
	startX = max(0, startX - dX)
	startY = max(0, startY - dY)
	endX = min(orig_width, endX + (dX * 2))
	endY = min(orig_height, endY + (dY * 2))
	"""

	#extract padded ROI
	roi = orig[startY:endY, startX:endX]

	#(1) language english, (2) 1 for LSTM neural network model, (3) 7 for ROI as single line of text
	config = ("-l eng --oem 1 --psm 4")
	#apply Tesseract (used to find text)
	text = pytesseract.image_to_string(roi, config=config) #returns predicted text
	
	#add bounding box coords and OCR'd tex to results
	results.append(((startX, startY, endX, endY), text))
	
#sort results, bounding boxes top to bottom
results = sorted(results, key=lambda r:r[0][1])

#loop over results
for ((startX, startY, endX, endY), text) in results:
	print("OCR TEXT")
	print("========")
	print("{}\n".format(text))
	
	##Draw results on image
	
	#strip non-ASCII
	text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
	
	#draw bounding box and text
	output = orig.copy()
	cv2.rectangle(output, (startX, startY), (endX, endY),
		(0, 0, 255), 2)
	cv2.putText(output, text, (startX, startY - 20),
		cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
		
	cv2.imshow("Text Detection", output)
	cv2.waitKey(0)
