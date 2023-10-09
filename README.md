# perception_test_iccv2023
Solutions repository for Perception Test challenges in ICCV2023 workshop.

## Introduction  

We achieves the best performance in Temporal Sound Localisation task and runner-up in Temporal Action Localisation task. In this repository, we provide the pretrained video\&audio features, checkpoints, and codes for feature extraction, training, and inference.

## Get Started  

Please refer to INSTALL.md to install the prerequisite packages.  

## Feature Extraction  

For the video feature, we use the UMT large model pre-trained on Something Something-V2 and fine-tuned on the perception test temporal action localisation training set. For the audio features, we use the BEATs model as feature extractor and adopt its iter3+ checkpoints pre-trained on the AudioSet-2M dataset.

| Features | Modality | Download Link |
| BEATs_iter2 | Audio | Download |
| UMT-L Sth Sth-V2 ft | Video | Download |
| UMT-L Sth Sth-V2 | Video | Download |

## Temporal Sound Localisation  

### Training  

`python train.py configs/perception_tsl_multi_train.yaml`  

### Inference  

Inference on the validation set:  

`python eval.py configs/perception_tsl_multi_valid.yaml ./ckpt/XXX -epoch=XX`  

Inference on the test set:  

`python eval.py configs/perception_tsl_multi_test.yaml ./ckpt/XXX -epoch=XX --saveonly`  

## Temporal Action Localisation  

`python train.py configs/perception_tal_multi_train.yaml`  

### Inference  

Inference on the validation set:  

`python eval.py configs/perception_tal_multi_valid.yaml ./ckpt/XXX -epoch=XX`  

Inference on the test set:  

`python eval.py configs/perception_tal_multi_test.yaml ./ckpt/XXX -epoch=XX --saveonly`  

## Checkpoints  

We release the checkpoint in the below table.  

| Method | Task | Download |
| BEATs + UMT ft | tsl | ckpt \| log |
| BEATs + UMT | tsl | ckpt \| log |
| BEATs + UMT ft | tal | ckpt \| log |  

## Contact  

If you have any questions, please contact [Jiashuo Yu](mailto:yujiashuo[at]pjlab.org.cn) and [Guo Chen](chenguo1177[at]gmail.com)