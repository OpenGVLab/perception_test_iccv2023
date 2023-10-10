python train.py configs/single_feature/tal_umt_sthv2_ptft_train.yaml --output o
python eval.py configs/single_feature/tal_umt_sthv2_ptft_val.yaml ./ckpt/tal_umt_sthv2_ptft_train_o/ -epoch 30
python eval.py configs/single_feature/tal_umt_sthv2_ptft_val.yaml ./ckpt/tal_umt_sthv2_ptft_train_o/ -epoch 35
python eval.py configs/single_feature/tal_umt_sthv2_ptft_val.yaml ./ckpt/tal_umt_sthv2_ptft_train_o/ -epoch 25


