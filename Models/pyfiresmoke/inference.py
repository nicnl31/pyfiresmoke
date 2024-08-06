"""
This script implements inference for the best model. Its goal is to:
1. Measure inference time per frame
2. Measure inference performance, by exporting, for each ROI:
- Unique IDs
- The predicted class

The outputs are:
- A folder contain ROI images for inference analysis
- A csv file containing:
    - ROI unique ID
    - Generated feature vector
    - Predicted class
    - Inference time (in ms):
        - Feature vector generation time
        - Prediction time
        - Total time (from vector generation to prediction)
"""
import os
import time
import uuid
import json
from typing import Type

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import crop
from extractor.extract import HaralickFeatureExtractor
from mapper import StringToFunctionMapper
from utils import create_dir_if_not_exists


class LatencyCallback(object):
    def __init__(self):
        self.start = None
        self.end = None
        self.latency_ms = None

    def on_start(self):
        self.start = time.perf_counter()

    def on_end(self):
        self.end = time.perf_counter()
        self.latency_ms = (self.end - self.start) * 1e3


class Inference(object):
    """
    The objective of the Inference class is to process ROIs, prepare feature
    vectors, and perform prediction using a trained model.

    The general pipeline is:
    - For each frame, extract the corresponding NumPy array
    - Convert to the target colour space using cv2, for example HSV:
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    - Compute the feature vector
    - Load a trained model
    - Feed the feature vector into the trained model and return a prediction
    - Save the prediction to a csv, and save the ROI as an image, under the same
      UUID.
    """
    def __init__(
            self,
            model_class: Type[nn.Module],
            model_kwargs: dict,
            model_weights_path: str,
            model_inference_params_path: str,
            data_colour_space: str,
            roi_out_path: str,
            csv_out_dir: str,
            device: str = "cpu",
            torch_dtype: torch.dtype = torch.float32,
        ):
        self.model = model_class(**model_kwargs)
        self.model.load_state_dict(torch.load(model_weights_path, weights_only=True))
        self.device = device
        self.model.to(self.device)
        self.torch_dtype = torch_dtype
        self.roi_out_path = roi_out_path
        self.model_inference_params_path = model_inference_params_path
        self.data_colour_space = data_colour_space
        self.csv_out_dir = csv_out_dir
        self.classes = {
            0: "fire",
            1: "smoke",
            2: "neither"
        }
        self.out_data = {
            "img_uuid": [],
            "pred_label": [],
            "comp_time_ms": [],
            "inference_time_ms": [],
            "total_time_ms": []
        }
        # String mapper class for parameter mapping
        self.str_mapper = StringToFunctionMapper()

    def infer(
            self,
            video_path: str,
            annot_path: str,
            verbose: int = 1
    ):
        """
        Perform inference on a single video. Process all ROIs and make
        predictions. Saves the cropped ROI to the ROI out path, along with the
        predictions, to enable manual inspection.
        :return:
        """
        with open(self.model_inference_params_path, mode='r') as g:
            inference_params = json.load(g)

        inference_mean = torch.tensor(inference_params["preprocessing"]["mean"])
        inference_stdev = torch.tensor(inference_params["preprocessing"]["stdev"])

        cap = cv2.VideoCapture(video_path)
        annot = pd.read_csv(annot_path)

        while not cap.isOpened():
            cap = cv2.VideoCapture(video_path)
            cv2.waitKey(10)
            if verbose and verbose > 0:
                print("Waiting for video sequence to open...")
        if verbose and verbose > 0:
            print("Video sequence opened.")

        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        total_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
                if verbose and verbose > 0:
                    print("Frame is not ready. Retrying...")
                cv2.waitKey(10)
            elif ret:
                pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if verbose and verbose > 0:
                    print(f"Frame {pos_frame} of {total_frame_count}")
                if pos_frame == total_frame_count:
                    len_data = self.export_inference_data()
                    if verbose and verbose > 0:
                        print(f"Finished inference: {len_data} ROIs "
                              f"saved to {self.csv_out_dir}/inference_results."
                              f"csv).")
                    break
                annot_this_offset = annot[annot['counter'] == pos_frame]
                if len(annot_this_offset) == 0:
                    continue
                # Convert frame to the correct colour space
                colour_mapper = self.str_mapper.map(
                    mapper_type="cv2.cvtColor",
                    to_map=f"bgr2{self.data_colour_space.lower()}"
                )
                frame = cv2.cvtColor(frame, colour_mapper)
                # Process each ROI for this offset value
                total_frame_latency = 0.0
                for roi in range(len(annot_this_offset)):
                    roi_uuid = str(uuid.uuid4())
                    # COMP TIME LATENCY MEASURE
                    comp_time_latency = LatencyCallback()
                    comp_time_latency.on_start()
                    is_empty, frame_cropped = crop(
                        arr=frame,
                        xmin=annot_this_offset.iloc[roi]['min_x'],
                        ymin=annot_this_offset.iloc[roi]['min_y'],
                        xmax=annot_this_offset.iloc[roi]['max_x'],
                        ymax=annot_this_offset.iloc[roi]['max_y']
                    )
                    feature_extractor = HaralickFeatureExtractor(colour_space=self.data_colour_space)
                    feature_vector = torch.tensor(feature_extractor.create_feature_vector(
                        image_array=frame_cropped).values)
                    # Standardise the feature vector
                    feature_vector = (feature_vector - inference_mean) / inference_stdev
                    feature_vector = feature_vector.type(self.torch_dtype)
                    comp_time_latency.on_end()
                    total_frame_latency += comp_time_latency.latency_ms

                    self.model.eval()
                    with torch.no_grad():
                        feature_vector_device = feature_vector.to(self.device)
                        inf_latency = LatencyCallback()
                        inf_latency.on_start()
                        pred = self.model(feature_vector_device)
                        pred_softmax = torch.softmax(pred, dim=0)
                        pred = torch.argmax(pred).item()
                        inf_latency.on_end()
                        total_frame_latency += inf_latency.latency_ms

                    # Store and export results
                    self.out_data["img_uuid"].append(roi_uuid)
                    self.out_data["pred_label"].append(self.classes[pred])
                    self.out_data["comp_time_ms"].append(comp_time_latency.latency_ms)
                    self.out_data["inference_time_ms"].append(inf_latency.latency_ms)
                    self.export_roi(arr=frame_cropped, 
                                    roi_uuid=roi_uuid, 
                                    pred=self.classes[pred],
                                    softmax_vec=pred_softmax)

                    if verbose and (verbose > 0):
                        print(f"ROI ID: {roi_uuid}, Prediction: {self.classes[pred]}, "
                              f"Comp latency: {comp_time_latency.latency_ms:.2f}ms, "
                              f"Inf latency: {inf_latency.latency_ms:.2f}ms")

                for _ in range(len(annot_this_offset)):
                    self.out_data["total_time_ms"].append(total_frame_latency)
        cap.release()
        cv2.destroyAllWindows()
        return

    def export_roi(
            self, 
            arr: np.ndarray, 
            roi_uuid: str, 
            pred: str, 
            softmax_vec: torch.tensor
            ) -> None:
        create_dir_if_not_exists(self.roi_out_path)
        if self.data_colour_space not in ["RGB", "rgb"]:
            try:
                cvt_rgb = self.str_mapper.map(
                    mapper_type="cv2.cvtColor",
                    to_map=f"{self.data_colour_space.lower()}2rgb"
                )
                arr = cv2.cvtColor(arr, cvt_rgb)
            except:
                raise KeyError(f"Mapping {self.data_colour_space.lower()}2rgb "
                               f"not found in the StringToFunctionMapper class."
                               f" Please add to it and try again.")
        
        # EXTRACT PROBABILITY OF FIRE/SMOKE/NEITHER
        fire_prob = softmax_vec[0].item()
        smoke_prob = softmax_vec[1].item()
        neither_prob = softmax_vec[2].item()

        # VISUALISE ROI
        plt.imshow(arr)
        plt.title(f"prediction:{pred}\n{fire_prob:.2f}fire, {smoke_prob:.2f}smoke, {neither_prob:.2f}neither")
        plt.savefig(f"{self.roi_out_path}/{roi_uuid}.jpg",
                    bbox_inches='tight')
        return

    def export_inference_data(self) -> int:
        df = pd.DataFrame(self.out_data)
        create_dir_if_not_exists(self.csv_out_dir)
        df.to_csv(f"{self.csv_out_dir}/inference_results.csv",
                  encoding='utf-8', index=False, header=True)
        return len(df)
