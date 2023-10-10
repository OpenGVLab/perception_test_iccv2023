python train.py configs/perception_tal_mm_train.yaml --output o
python eval.py configs/perception_tal_mm_valid.yaml ./ckpt/perception_tal_mm_train_o/ -epoch 35
python eval.py configs/perception_tal_mm_valid.yaml ./ckpt/perception_tal_mm_train_o/ -epoch 25
python eval.py configs/perception_tal_mm_valid.yaml ./ckpt/perception_tal_mm_train_o/ -epoch 30


