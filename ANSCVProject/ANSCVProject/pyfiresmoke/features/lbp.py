from skimage import feature
import numpy as np
from utils.utils import to_grayscale


class LocalBinaryPatterns(object):
	def __init__(self, num_points: int, radius: int):
		"""
		:param num_points: int
			The number of point surrounding the pixel.
		:param radius:
			The radius from the central pixel to the surrounding points.
		"""
		self.num_points = num_points
		self.radius = radius

	def get_local_binary_patterns(self, image_channel, method="uniform"):
		"""
		Calculates the local binary pattern matrix.

		:param image_channel:
			The single channel of the image (if coloured), or the image itself
			(if grayscale). Must be a 2d matrix.
		:param method:
			The method from which to extract the LBP matrix. Default is
			"uniform", where a maximum of two 0-1 or 1-0 transitions are
			accepted.
		:return:
		"""
		lbp = feature.local_binary_pattern(
			image_channel,
			self.num_points,
			self.radius,
			method=method
		)
		return lbp

	def hist_describe(self, image, eps=1e-7):
		"""
		Describe the local binary patterns as a histogram feature vector.
		:param image:
			The image in PIL form.
		:param eps:
			Floating point correction for histogram normalisation.
		:return:
		"""
		image_gray = to_grayscale(image)

		lbp = self.get_local_binary_patterns(image_gray)
		(hist, _) = np.histogram(
			lbp.ravel(),
			bins=np.arange(0, self.num_points+3),
			range=(0, self.num_points+2))
		# normalise the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)
		# return the histogram of Local Binary Patterns
		return hist
