"""
Mapper class for different parameters. The self.mapper attribute can be updated
to serve a variety of purposes.
"""

import torch.nn as nn
import cv2


class StringToFunctionMapper(object):
	def __init__(self):
		self.mapper = {
			"activation": {
				"relu": nn.ReLU,
				"relu6": nn.ReLU6,
				"prelu": nn.PReLU,
				"silu": nn.SiLU,
				"selu": nn.SELU,
				"tanh": nn.Tanh,
				"hardswish": nn.Hardswish,
				"hardtanh": nn.Hardtanh
			},
			"cv2.cvtColor": {
				"bgr2rgb": cv2.COLOR_BGR2RGB,
				"bgr2hsv": cv2.COLOR_BGR2HSV,
				"hsv2rgb": cv2.COLOR_HSV2RGB
			}
		}

	def map(self, mapper_type: str, to_map: str):
		return self.mapper[mapper_type][to_map]
