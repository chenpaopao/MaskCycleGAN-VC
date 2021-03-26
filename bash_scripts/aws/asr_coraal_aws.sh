python -W ignore::UserWarning -m asr.main \
    --name coraal \
    --data_dir ~/data/datasets \
    --save_dir ~/data/results \
    --coraal \
    --num_epochs 100 \
    --batch_size 10 \
    --gpu_ids 0 \
    --num_workers 1 \
    --n_feats 80 \
    --epochs_per_save 2 \
    --pretrained_ckpt_path ~/data/results/librispeech/ckpts/best.pth.tar \
    --continue_train \