import glob
import os
from pathlib import Path

from extractor.dataset import Dataset


if __name__ == "__main__":
	dataset_dir = '../../data/dataset1'
	names = [Path(x).stem for x in glob.glob(f"{dataset_dir}/*.jpg")]
	dataset = Dataset(dir=dataset_dir)
	for name in names:
		image_name = f"{name}.jpg"
		label_name = f"{name}.txt"
		dataset.add_data(image_name=image_name, label_name=label_name)
	dataset.export_data(filename="dataset1.csv")
