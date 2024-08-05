import numpy as np
from skimage.feature import graycomatrix, graycoprops


class GLCM(object):
	def __init__(self, channel: str):
		self.channel = channel
		self.props = {}

	def get_glcm(
			self,
			image_channel: np.ndarray,
			distances: list[int] = [5],
			angles: list[float] =[0., np.pi/4., np.pi/2., 3.*np.pi/4.],
			levels: int = 256,
			symmetric: bool = False,
			normed: bool = False
	) -> np.ndarray:
		"""
		Calculates the gray-level co-occurrence matrix for a given 2d image
		array.

		:param image_channel:
			The 2d image array.
		:param distances:
			The list of offsets for calculation.
		:param angles:
			The list of angles for calculation.
		:param levels:
			The image's bit depth, i.e. the number of gray-levels. The default
			is 256, and can be quantised to a lower bit depth for better memory.
		:param symmetric:
		:param normed:
		:return:
		"""
		image_channel = image_channel.astype(np.uint8)
		glcm = graycomatrix(image_channel, distances=distances, angles=angles,
							levels=levels, symmetric=symmetric, normed=normed)

		return glcm

	def get_stock_prop_from_glcm(
			self,
			glcm: np.ndarray,
			prop: str
	) -> None:
		"""
		Gets the texture property from a GLCM matrix.

		:param glcm:
			The GLCM matrix.
		:param prop:
			The property. Properties of interest include contrast, dissimilarity,
			homogeneity, and energy (angular second moment).
		:param channel:
			The string representing the channel of the image.
		:return:
		"""
		# Ravelled Numpy array of shape (distances*angles, )
		feature_vector_arr = graycoprops(glcm, prop=prop).ravel()

		# Append features to the self.props dictionary
		for i in range(len(feature_vector_arr)):
			self.props[f"{prop}{i+1}_channel{self.channel}"] = feature_vector_arr[i]
		return
