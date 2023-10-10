python train.py configs/perception_tal_mm_multi_train.yaml --output o
python eval.py configs/perception_tal_mm_multi_valid.yaml ./ckpt/perception_tal_mm_multi_train_o/ -epoch 35
python eval.py configs/perception_tal_mm_multi_valid.yaml ./ckpt/perception_tal_mm_multi_train_o/ -epoch 30
python eval.py configs/perception_tal_mm_multi_valid.yaml ./ckpt/perception_tal_mm_multi_train_o/ -epoch 25
python eval.py configs/perception_tal_mm_multi_valid.yaml ./ckpt/perception_tal_mm_multi_train_o/ -epoch 20


