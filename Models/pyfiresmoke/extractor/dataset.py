"""
This library creates the full dataset based on the row-wise feature extractor
from extract.py.

It operates by:
	-   Taking as input a single image file (e.g. img1.jpg) and a corresponding
		labelled bounding box vector in the form of [label, xmin, ymin, w_bbox, h_bbox],
		(e.g. [0, 0.1531, 0.7187, 0.6562, 0.9625]), where 0 is the class, and
		the rest is normalised coordinates. There may be N such instances in a
		.txt file, since there may be N bounding boxes in a single image.
	-   Extracting the ROI from the bounding box data, and returns a 3d array
		representation of the ROI. From here, the feature extractor computes the
		Haralick texture features of this ROI, and returns a feature vector for
		each ROI.
	-   Producing a Pandas DataFrame representing the full dataset, which has
		the following columns:
		-   image name
		-   Haralick features
		-   label

Functions design:
	1.  Read in all image data + label data -> return: list of pandas series
	2.  Read in list of Pandas series -> return: pandas DataFrame
"""
import os
import glob
import warnings
from PIL import Image

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

from .extract import HaralickFeatureExtractor
from utils import to_array


class Dataset(object):
	def __init__(
			self,
			image_dir='/data/dataset1',
			annot_dir='/data/dataset1',
			colour_space="HSV"):
		"""
		:param dir:
			The relative directory to the dataset folder. For example,
			'/data/dataset1'
		"""
		self.image_dir = image_dir
		self.annot_dir = annot_dir
		self.colour_space = colour_space
		self.data = []
		self.all_rois = 0
		self.skipped_rois = 0

	def unnormalise(
			self,
			image_arr: np.ndarray,
			xcen_norm: float,
			ycen_norm: float,
			w_norm: float,
			h_norm: float
	) -> tuple[int, int, int, int]:
		"""
		Returns the unnormalised indices of normalised coordinates.
		The normalised coordinates in [x_center, y_center, width, height] format
		is that from YOLO.
		Unnormalisation tutorial: https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/

		:param image_arr:
		:param xcen_norm:
		:param ycen_norm:
		:param w_norm:
		:param h_norm:
		:return:
		"""
		# Get the x-y dimensions of the original array
		arr_shape = image_arr.shape
		w_arr = arr_shape[1]
		h_arr = arr_shape[0]
		# Unnormalise the coordinates and convert to [xmin, ymin, w_bbox, h_bbox]
		w_bbox = int(w_norm * w_arr)
		h_bbox = int(h_norm * h_arr)
		xmin = int(xcen_norm * w_arr - w_bbox/2)
		ymin = int(ycen_norm * h_arr - h_bbox/2)

		return xmin, ymin, w_bbox, h_bbox

	def crop(self, image_arr, xmin, ymin, w_bbox, h_bbox):
		"""
		Crops the original array according to the xyxy coordinates, and returns
		the cropped array.

		:param image_arr:
		:param xmin:
		:param ymin:
		:param xmax:
		:param ymax:
		:return:
		"""
		cropped_arr = image_arr[ymin-1:ymin+h_bbox, xmin-1:xmin+w_bbox, :]
		return cropped_arr

	def parse_labels(self, label_name):
		"""
		Parse the label file and returns a numpy array of size (n_bbox, 5),
		where 5 is [label, xcen_norm, ycen_norm, w_norm, h_norm], and n_bbox is
		the number of bounding boxes.
		:param label_name:
		:return:
		"""
		label_path = f"{self.annot_dir}/{label_name}"
		labels_arr = np.genfromtxt(label_path, delimiter=" ")
		return labels_arr

	def add_data(
			self,
			image_name: str,
			label_name: str,
			should_resize: bool = True,
			resize_dim: tuple[int, int] = (1280, 720),
			verbose: int = 1
	) -> None:
		"""
		Adds a single row of data to the self.data attribute. This row will be
		represented as a Pandas series, containing the following fields:
			-   Image name (e.g. img1.jpg)
			-   Haralick features (e.g. energy_H, energy_S, energy_V)
			-   Label, in numeric (e.g. 0)
		:param image_name:
			The name of the image file, e.g. 'img1.jpg'
		:param label_name:
			The name of the label file, e.g. 'img1.txt'
		:param should_resize:
			Whether to resize the image.
		:param resize_dim:
			The resized dimensions.
		:return:
		"""
		# Extract the PIL image as a numpy array, and resize according to the
		# dataset.
		image_path = f"{self.image_dir}/{image_name}"
		image_pil = Image.open(image_path).convert(self.colour_space)
		if should_resize:
			image_pil = image_pil.resize(resize_dim)
		image_arr = to_array(image_pil)
		# Extract the labels and rois as a numpy array
		labels_arr = self.parse_labels(label_name=label_name)

		# Extract the ROIs using the label array and the original image array
		label_and_rois = []

		# ======================================================================
		# Case 1: More than 1 rows of ROI
		if labels_arr.ndim == 2:
			for row in labels_arr:
				xywh_norm2 = row[1:]
				label = row[0]
				xmin, ymin, w_bbox, h_bbox = self.unnormalise(
					image_arr=image_arr,
					xcen_norm=xywh_norm2[0],
					ycen_norm=xywh_norm2[1],
					w_norm=xywh_norm2[2],
					h_norm=xywh_norm2[3]
				)
				# print(xmin, ymin, w_bbox, h_bbox)
				cropped_arr = self.crop(image_arr, xmin, ymin, w_bbox, h_bbox)

				# TO VISUALISE, ENABLE THIS CODE BLOCK
				# # ============================================================
				# if cropped_arr.size != 0:
				# 	plt.imshow(cropped_arr, cmap='hsv', vmin=0, vmax=255)
				# 	plt.show()
				# # ============================================================

				label_and_rois.append((label, cropped_arr))

		# ======================================================================
		# Case 2: Only 1 row of ROI
		elif labels_arr.ndim == 1 and labels_arr.size != 0:
			xywh_norm1 = labels_arr[1:]
			label = labels_arr[0]
			xmin, ymin, w_bbox, h_bbox = self.unnormalise(
				image_arr=image_arr,
				xcen_norm=xywh_norm1[0],
				ycen_norm=xywh_norm1[1],
				w_norm=xywh_norm1[2],
				h_norm=xywh_norm1[3]
			)
			cropped_arr = self.crop(image_arr, xmin, ymin, w_bbox, h_bbox)

			# TO VISUALISE, ENABLE THIS CODE BLOCK
			# # ================================================================
			# if cropped_arr.size != 0:
			# 	plt.imshow(cropped_arr, cmap='hsv', vmin=0, vmax=255)
			# 	plt.show()
			# # ================================================================

			label_and_rois.append((label, cropped_arr))

		# ======================================================================
		# Case 3: 0 ROIs (i.e. no annotations in the case of negative examples)
		elif labels_arr.size == 0:
			cropped_arr = image_arr
			label = 2
			label_and_rois.append((label, cropped_arr))

			# TO VISUALISE, ENABLE THIS CODE BLOCK
			# # ================================================================
			# if cropped_arr.size != 0:
			# 	plt.imshow(cropped_arr, cmap='hsv', vmin=0, vmax=255)
			# 	plt.show()
			# # ================================================================

		# Add the ROIs found to the running total of ROIs
		self.all_rois += len(label_and_rois)
		for each_roi in label_and_rois:
			if each_roi[1].size != 0:
				feature_extractor = HaralickFeatureExtractor(
					colour_space=self.colour_space
				)
				feature_vector = feature_extractor.create_feature_vector(
					image_array=each_roi[1]
				)
				feature_vector['label'] = each_roi[0]
				feature_vector['image'] = image_name
				self.data.append(feature_vector)
			else:
				self.skipped_rois += 1
				pass

		if verbose > 0:
			print(f"{image_name}: {len(label_and_rois)} ROIs found in this run, added {len(self.data)} to dataframe, skipped {self.skipped_rois}.")

		return

	def to_dataframe(self):
		return pd.DataFrame(self.data)

	def export_data(self, filename: str) -> None:
		dataset_full = self.to_dataframe()
		dataset_full.to_csv(filename, encoding='utf-8', index=False)
		print(f"Total ROIs found: {self.all_rois}. Exported {len(self.data)} ROIs to {os.getcwd()}/{filename}. Skipped {self.skipped_rois}.")
		return
