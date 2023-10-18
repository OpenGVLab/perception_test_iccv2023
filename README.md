# perception_test_iccv2023
Solutions repository for Perception Test challenges in ICCV2023 workshop.

## Introduction  

We achieves the best performance in Temporal Sound Localisation task and runner-up in Temporal Action Localisation task. In this repository, we provide the pretrained video\&audio features, checkpoints, and codes for feature extraction, training, and inference.

## Get Started  

Please refer to INSTALL.md to install the prerequisite packages.  

## Feature Extraction  

### TAL  

For the video features, we use the UMT large model pre-trained on Something Something-V2 and the VideoMAE model pre-trained on Ego4D-Verb dataset. The weights of Ego4d can be found [here](https://github.com/OpenGVLab/ego4d-eccv2022-solutions). These two features are concatenated before putting into the ActionFormer model during both training and inference stages.

For the audio features, we use the BEATs model as feature extractor and adopt its iter3+ checkpoints pre-trained on the AudioSet-2M dataset. we provide scripts to extract BEATs and CAV-MAE (although not used), please use `python audio_feat_extract.py` to extract audio features.

### TSL  

For the video feature, we use the [UMT large model](https://github.com/OpenGVLab/unmasked_teacher) pre-trained on Something Something-V2 and fine-tuned on the perception test temporal action localisation training set. 

For the audio features, we use the BEATs model as feature extractor and adopt its iter3+ checkpoints pre-trained on the AudioSet-2M dataset. we provide scripts to extract BEATs and CAV-MAE (although not used), please use `python audio_feat_extract.py` to extract audio features.

### Download  

| Features | Modality | Task | Download Link |
|---|---|---|---|
| BEATs_iter2 | Audio | TAL\&TSL | [Download](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/opengvlab/perception_test_iccv2023/pt_tsl_beats_iter3_feature.zip) |
| Ego4d_verb | Video | TAL | [Download](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/opengvlab/perception_test_iccv2023/pt_tal_videomae_large_ego4d_verb_feature_s4.zip) |
| UMT-L Sth Sth-V2 | Video | TAL | [Download](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/opengvlab/perception_test_iccv2023/pt_tal_umt_large_sthv2_feature_s4.zip) |
| UMT-L Sth Sth-V2 ft | Video | TSL | [Download](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/opengvlab/perception_test_iccv2023/pt_tal_umt_large_sthv2_perception_test_ft1_feature_s2.zip) |


## Temporal Sound Localisation  

### Training  

`cd ./tsl/` 

`python train.py configs/perception_tsl_multi_train.yaml`  

### Inference  

Inference on the validation set:  

`cd ./tsl/`  

`python eval.py configs/perception_tsl_multi_valid.yaml ./ckpt/XXX -epoch=XX`  

Inference on the test set:  

`cd ./tsl/` 

`python eval.py configs/perception_tsl_multi_test.yaml ./ckpt/XXX -epoch=XX --saveonly`  

## Temporal Action Localisation  

`cd ./tal/` 

`python train.py configs/perception_tal_multi_train.yaml`  

### Inference  

Inference on the validation set:  

`cd ./tal/` 

`python eval.py configs/perception_tal_multi_valid.yaml ./ckpt/XXX -epoch=XX`  

Inference on the test set:  

`cd ./tal/` 

`python eval.py configs/perception_tal_multi_test.yaml ./ckpt/XXX -epoch=XX --saveonly`  

## Checkpoints  

We release the checkpoint in the below table.  

| Method | Task | mAP (Valid) | Download |
|---|---|---|---|
| BEATs + UMT | tsl | 26.70 | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/opengvlab/perception_test_iccv2023/tsl_multi_epoch20.pth.tar ) |
| BEATs + UMT ft | tsl | 39.25 | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/opengvlab/perception_test_iccv2023/tsl_multi_ft_epoch20.pth.tar ) |
| BEATs + UMT | tal | 44.14 | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/opengvlab/perception_test_iccv2023/tal_multi_umtonly.pth.tar) |
| BEATs + UMT\&VideoMAE | tal | 46.75 | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/opengvlab/perception_test_iccv2023/tal_multi.pth.tar) |


## Contact  

If you have any questions, please contact [Jiashuo Yu](mailto:yujiashuo[at]pjlab.org.cn) and [Guo Chen](chenguo1177[at]gmail.com)
