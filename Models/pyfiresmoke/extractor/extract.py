"""
This library extracts the Haralick texture features for each region of interest
(ROI) of an image frame. It inputs the following:
- The original ROI in PIL format, which is assumed to be colour-space agnostic.
  It works in any colour space.
- The texture features (properties) to be extracted, in list[str] format. If
  none is specified, a default of 4 features is given, which are:
  ["energy", "homogeneity", "dissimilarity", "contrast"]

The main method, create_feature_vector, inputs the LBP and GLCM parameters
and calculates the Haralick texture features for each channel. It then
concatenates all channel features and returns a complete feature vector in
Pandas Series format, which represents a row of data for the subsequent machine
learning model.
"""
from PIL import Image

import numpy as np
import pandas as pd

from features.glcm import GLCM


class HaralickFeatureExtractor(object):
	def __init__(
			self,
			colour_space: str,
			props: list[str] = None,
	) -> None:
		if not props:
			self.props = ["energy", "homogeneity", "dissimilarity", "contrast"]
		elif props:
			self.props = props

		self.channels = [i for i in colour_space]

	def create_feature_vector(
			self,
			image_array: np.ndarray,
			# lbp_num_points: int = 8,
			# lbp_radius: int = 1,
			glcm_distances: list[int] = [5],
			glcm_angles: list[float] = [0.],
			glcm_levels: int = 256,
			glcm_symmetric: bool = True,
			glcm_normed: bool = True
	) -> pd.Series:
		"""
		Creates the feature vector from all Haralick texture features. The
		individual channel features are concatenated into a final feature vector.

		:param image_array:
		:param lbp_num_points:
		:param lbp_radius:
		:param glcm_distances:
		:param glcm_angles:
		:param glcm_levels:
		:param glcm_symmetric:
		:param glcm_normed:
		:return:
		"""

		channel_0 = image_array[:, :, 0]
		channel_1 = image_array[:, :, 1]
		channel_2 = image_array[:, :, 2]

		# Extract local binary patterns from each channel
		# lbp = LocalBinaryPatterns(num_points=lbp_num_points, radius=lbp_radius)
		# lbp_channel_0 = lbp.get_local_binary_patterns(image_channel=channel_0)
		# lbp_channel_1 = lbp.get_local_binary_patterns(image_channel=channel_1)
		# lbp_channel_2 = lbp.get_local_binary_patterns(image_channel=channel_2)

		# Extract GLCM secondary texture statistics from each channel
		glcm_channel_0 = GLCM(channel=self.channels[0])
		glcm_channel_1 = GLCM(channel=self.channels[1])
		glcm_channel_2 = GLCM(channel=self.channels[2])

		# Calculate GLCM matrix for each channel
		glcm_channel_0_matx = glcm_channel_0.get_glcm(
			# image_channel=lbp_channel_0,
			image_channel=channel_0,
			distances=glcm_distances,
			angles=glcm_angles,
			levels=glcm_levels,
			symmetric=glcm_symmetric,
			normed=glcm_normed
		)
		glcm_channel_1_matx = glcm_channel_1.get_glcm(
			# image_channel=lbp_channel_1,
			image_channel=channel_1,
			distances=glcm_distances,
			angles=glcm_angles,
			levels=glcm_levels,
			symmetric=glcm_symmetric,
			normed=glcm_normed
		)
		glcm_channel_2_matx = glcm_channel_2.get_glcm(
			# image_channel=lbp_channel_2,
			image_channel=channel_2,
			distances=glcm_distances,
			angles=glcm_angles,
			levels=glcm_levels,
			symmetric=glcm_symmetric,
			normed=glcm_normed
		)

		# Get property for each channel
		for prop in self.props:
			glcm_channel_0.get_stock_prop_from_glcm(
				glcm_channel_0_matx,
				prop
			)
			glcm_channel_1.get_stock_prop_from_glcm(
				glcm_channel_1_matx,
				prop
			)
			glcm_channel_2.get_stock_prop_from_glcm(
				glcm_channel_2_matx,
				prop
			)

		# Concatenate channel information into a feature vector
		feature_vector = pd.concat(
			(
				pd.Series(glcm_channel_0.props),
				pd.Series(glcm_channel_1.props),
				pd.Series(glcm_channel_2.props)
			)
		)
		return feature_vector
