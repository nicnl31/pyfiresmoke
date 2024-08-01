import glob
import os
from pathlib import Path
import warnings

from extractor.dataset import Dataset

warnings.filterwarnings("ignore")

if __name__ == "__main__":
	dataset_dir = 'data/fasdd_dataset'
	image_dir = 'images'
	annot_dir = 'annotations/YOLO_CV/labels'
	csv_dir = 'csv'
	names = [Path(x).stem for x in glob.glob(f"{dataset_dir}/{image_dir}/*.jpg")]
	# names = ["neitherFireNorSmoke_CV030217"]
	colour_spaces = ["RGB", "HSV"]
	for colour in colour_spaces:
		dataset = Dataset(
			image_dir=f"{dataset_dir}/{image_dir}",
			annot_dir=f"{dataset_dir}/{annot_dir}",
			colour_space=colour
		)
		for name in names:
			image_name = f"{name}.jpg"
			label_name = f"{name}.txt"
			dataset.add_data(
				image_name=image_name,
				label_name=label_name,
				should_resize=False
			)
		# print(dataset.to_dataframe())
		dataset.export_data(filename=f"{dataset_dir}/{csv_dir}/dataset_fasdd_no_lbp_{colour}.csv")
