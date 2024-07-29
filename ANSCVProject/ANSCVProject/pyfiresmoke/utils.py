"""
Utility functions for various purposes.
"""

import os
import json
from PIL import Image

import numpy as np
import pandas as pd
import cv2


def to_array(image: Image):
	return np.array(image)


def to_grayscale(image: Image):
	"""
	:param image:
		The PIL object representing the image.
	:return:
	"""
	# Convert to grayscale from PIL object
	image_arr = np.array(image)
	image_gray = cv2.cvtColor(image_arr, cv2.COLOR_RGB2GRAY)
	return image_gray


def crop(arr: np.ndarray, xmin, ymin, xmax, ymax):
	"""
	Crop a 3d array
	:param arr:
	:param xmin:
	:param ymin:
	:param xmax:
	:param ymax:
	:return:
	"""
	assert len(arr.shape) == 3, "arr is not a 3d array."
	arr_cropped = arr[ymin:ymax, xmin:xmax, :]
	is_empty = arr_cropped.size == 0
	return is_empty, arr_cropped


def parse(
		dataset_dir: str,
		dataset_name: str,
		split_dir: str,
		file_name: str
) -> pd.Series:
	full_path = f"{dataset_dir}/{dataset_name}/{split_dir}/{file_name}"
	with open(full_path, "r") as file:
		split = [os.path.basename(line.rstrip()) for line in file]
		return pd.Series(split, name='image')


def list_argmax(lst: list[float]) -> int:
	"""
	Find the argmax of a list of numbers.
	:param lst: list[float]
		The list of numbers.
	:return:
	"""
	return lst.index(max(lst))


def list_argmin(lst: list[float]) -> int:
	"""
	Find the argmin of a list of numbers.
	:param lst: list[float]
		The list of numbers.
	:return:
	"""
	return lst.index(min(lst))


class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		if isinstance(obj, np.floating):
			return float(obj)
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return super(NumpyEncoder, self).default(obj)


def list_argmax(lst: list[float]) -> int:
	"""
	Find the argmax of a list of numbers.
	:param lst: list[float]
		The list of numbers.
	:return:
	"""
	return lst.index(max(lst))


def list_argmin(lst: list[float]) -> int:
	"""
	Find the argmin of a list of numbers.
	:param lst: list[float]
		The list of numbers.
	:return:
	"""
	return lst.index(min(lst))


class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		if isinstance(obj, np.floating):
			return float(obj)
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return super(NumpyEncoder, self).default(obj)
