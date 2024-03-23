#!./env/bin/python
#-*- encoding: utf8 -*-
""" Vehicul counting """

import os
import logging
import logging.config

import numpy as np
import pandas as pd
import cv2 as cv
# import torch
from ultralytics import YOLO
from tracker import Tracker


# logging setting:
logging.config.fileConfig(fname='logging.conf')
LOG = logging.getLogger(__name__)


COUNT_LINE_POSITION = 600
# ALPHA = 0.5

MIN_WIDTH_RECT = 80
MIN_HEIGHT_RECT = 80

CY1 = 310
CY2 = 340

OFFSET = 6  # Allowable error between pixel


class TrackingCounter:
	""" Program of counter of object tracked. """

	def __init__(self,
				 model_file_path: str,
				 count_line_pos: int = COUNT_LINE_POSITION,
				 min_width_rect: int = MIN_HEIGHT_RECT,
				 min_height_rect: int = MIN_HEIGHT_RECT,
				 offset: int = OFFSET,):

		self._count_line_pos = count_line_pos
		self._min_width_rect = min_width_rect
		self._min_height_rect = min_height_rect
		self._offset = offset

		self._model = YOLO(model=model_file_path)
		self._tracker = Tracker()
		self._indexes = set()
		self._center_point_detect = []
		# self._back_sups = cv.bgsegm.createBackgroundSubtractorMOG()

		# cv.namedWindow('RGB')
		# cv.setMouseCallback('RGB', self.rgb)

	@staticmethod
	def rgb(event, x, y, flags, param):
		if event == cv.EVENT_MOUSEMOVE:
			colorsRGB = [x, y]
			print(f"\r\033[2K{colorsRGB}", end='', flush=True)

	@staticmethod
	def compute_center_point(x, y, w, h):
		""" Function of point center computing """
		x1 = int(w / 2)
		y1 = int(h / 2)
		cx = x + x1
		cy = y + y1
		return [cx, cy]

	def run_loop(self, file_path: str = None, video_source: int|str = None):
		""" Loop function """
		if not file_path:
			raise NotImplemented

		# open a text file and load it
		text_file = open('coco.txt', 'r')
		text_data = text_file.read()
		class_list = text_data.split('\n')
		# LOG.debug("class_list: " + str(class_list))

		cap = cv.VideoCapture(file_path)
		ones = np.ones((5, 5))
		frame_cnt = 0   # to count frame

		prediction = None
		boxes = None
		px = None
		car_list = []
		vh_down = {}
		count = 0
		x = 0
		y = 0
		w = 0
		h = 0

		while True:
			ret, frame = cap.read()
			if frame is None:
				# stop loop if sequence of frame is terminated
				LOG.debug("Frame iteration is done.")
				break

			frame_cnt += 1
			# if frame_cnt % 10 != 0:
			# 	continue

			# resize frame
			frame = cv.resize(frame, (1020, 500))

			# prediction using model
			prediction = self._model.predict(frame)
			boxes = prediction[0].boxes.data
			# LOG.debug("boxes: " + str(boxes))
			px = pd.DataFrame(boxes)
			px = px.astype('float')
			car_list.clear()

			for index, row in px.iterrows():
				x1 = int(row[0])
				y1 = int(row[1])
				x2 = int(row[2])
				y2 = int(row[3])
				d = int(row[5])
				cls = class_list[d]

				if 'car' in cls \
					or 'bus' in cls \
					or 'truck' in cls \
					or 'train' in cls \
					or 'bicycle' in cls \
					or 'motorcycle' in cls \
					or 'person' in cls:
					car_list.append([x1, y1, x2, y2])

			bbox_id = self._tracker.update(car_list)
			# LOG.debug("bbox_id: " + str(bbox_id))
			for bbox in bbox_id:
				x3, y3, x4, y4, idx = bbox
				w = x4 - x3
				h = y4 - y3
				cx = int(x3 + x4) // 2
				cy = int(y3 + y4) // 2
				if CY1 > (cy - self._offset) and CY1 < (cy + self._offset):
					vh_down[idx] = cy
					cv.rectangle(
						img=frame,
						pt1=(x3, y3),
						pt2=(x3 + w, y3 + h),
						color=(0, 250, 0),
						thickness=2)

				if idx in vh_down:
					# if CY2 > (cy - self._offset) and CY2 < (cy + self._offset):
					cv.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
					cv.putText(
                        img=frame,
                        text=str(idx),
                        org=(cx, cy),
                        fontFace=cv.FONT_HERSHEY_COMPLEX,
                        fontScale=0.8,
                        color=(0, 255, 255),
                        thickness=2)
					# self._counter += 1
					self._indexes.add(idx)

			# Following line overlays transparent rectangle
			# over the image
			# frame1 = cv.addWeighted(frame, ALPHA, frame1, (1 - ALPHA), 0)
			count = len(self._indexes)

			cv.line(frame, (135, CY1), (450, CY1), (127, 127, 127), 1)
			cv.line(frame, (564, CY1), (863, CY1), (127, 127, 127), 1)
			# cv.line(frame, (167, CY2), (932, CY2), (255, 255, 255), 1)
			# cv.putText(frame, ('1line'), (274, 318), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
			# cv.putText(frame, ('2line'), (181, 363), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
			cv.putText(
				img=frame,
				text=('vh count:' + str(count)),
				org=(60, 40),
				fontFace=cv.FONT_HERSHEY_COMPLEX,
				fontScale=0.8,
				color=(0, 127, 0),
				thickness=2)
			# LOG.debug('vh_down: ' + str(vh_down))
			LOG.debug("indexes: " + str(count))

			cv.imshow("RGB", frame)
			# cv.imshow("Dilata", dilatada)
			if cv.waitKey(1) & 0xFF == 27:
				LOG.info("Closed by user!")
				break

		# free video resources
		cv.destroyAllWindows()
		cap.release()


def main():
	""" Main function """
	if len(os.sys.argv) < 2:
		print("No video file path is defined.")
		return

	file_path = os.sys.argv[1]
	# tracking_counter = TrackingCounter('./models/yolov8s.pt')
	tracking_counter = TrackingCounter('./models/yolov8n.pt')
	tracking_counter.run_loop(file_path)


if __name__ == '__main__':
  main()
  os.sys.exit(0)
