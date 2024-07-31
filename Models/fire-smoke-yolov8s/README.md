# Fire and Smoke Detection with YOLOv8s

This project uses `YOLOv8s` as the base model for the detection of fire and smoke. The model was transfer-learned using pretrained weights from Ultralytics' YOLOv8.

The custom dataset used to perform transfer learning and fine-tuning of the model can be found on the Roboflow website, at https://universe.roboflow.com/custom-thxhn/fire-wrpgm.

# Steps to perform inference

## Load the pre-trained model
The model can be downloaded using the following:

```
from ultralytics import YOLO
model = YOLO("yolov8s.pt")
```

## Download the dataset
The custom dataset can be downloaded via the Roboflow API. For that, you need an API key, which is unique to you as you proceed to download the dataset from the source:

```
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="xxxxxxxxxxxxxxxxx")
project = rf.workspace("custom-thxhn").project("fire-wrpgm")
dataset = project.version(8).download("yolov8")
```

## Train the model, starting with pretrained weights
From this point, the model can be trained using the `train()` method from the YOLO object created above. For this training, the following parameters are used following experimentations:
- Device: `cpu`
- Epochs: `50`
- Optimizer: `Adam`
- Early stopping patience: `10`
- Batch size: `32`
- Initial learning rate: `2.5e-4`
- Final learning rate as a ratio of initial learning rate: `1e-2`
- Image size: `800`$\times$`800`

```
results = model.train(
    pretrained=True,
    data="data.yaml", 
    epochs=50, 
    device="cpu", 
    optimizer="Adam",
    patience=10,
    batch=32,
    lr0=2.5e-4,
    lrf=1e-2,
    imgsz=800
)
```

## Inference
The best weights are stored as `best.pt`. The following command can be executed to make inference on a `source`, which can be an image, a video sequence, or a live camera:

```
!yolo task=detect mode=predict model="best.pt" source="xxxxxxxxx"
```

More inference sources can be found on the Ultralytics documentation for YOLOv8: https://docs.ultralytics.com/modes/predict/#inference-sources

# Results

## Inference results
The below are some out-of-sample fast-forwarded camera footage videos that show `fire` and `smoke` predictions by the model. More results can be found in the `inference` folder of this repo.

It is generally found that the model detects fire and smoke better with stable video footage, and smoke is better detected than fire.

(Source: https://www.youtube.com/watch?v=BJ9ng9L1CA0)

https://github.com/ANSCENTER-PROJECTS/ANSCV/assets/86213993/1c2e476f-6ca4-481d-817f-08e43af3f6a0

(Source: https://www.youtube.com/watch?v=wBtiWyKvJJo)

https://github.com/ANSCENTER-PROJECTS/ANSCV/assets/86213993/779cb62b-9b5a-4500-8519-d12494e1f916

These are some sample batches from the test set.
![val_batch1_pred](https://github.com/ANSCENTER-PROJECTS/ANSCV/assets/86213993/2372bfd8-4b52-4ece-9ada-fea8835a5393)

![val_batch2_pred](https://github.com/ANSCENTER-PROJECTS/ANSCV/assets/86213993/f0f5c4b6-f482-4eb8-b9f4-859f307d2f75)

## Classification results and learning curves
Below is the model's test set confusion matrix and learning curves.
![confusion_matrix_normalized](https://github.com/ANSCENTER-PROJECTS/ANSCV/assets/86213993/6217702e-9fcf-47df-9ab9-de50189983b6)

![results](https://github.com/ANSCENTER-PROJECTS/ANSCV/assets/86213993/994e8e45-1533-42da-810c-7910f329adaa)
