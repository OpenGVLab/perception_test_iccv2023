python train.py configs/single_feature/tal_umt_k700_train.yaml --output o
python eval.py configs/single_feature/tal_umt_k700_val.yaml ./ckpt/tal_umt_k700_train_o/ -epoch 25
python eval.py configs/single_feature/tal_umt_k700_val.yaml ./ckpt/tal_umt_k700_train_o/ -epoch 30
python eval.py configs/single_feature/tal_umt_k700_val.yaml ./ckpt/tal_umt_k700_train_o/ -epoch 35


