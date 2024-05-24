python3 src/test.py \
        --dataset 'pathvqa' \
        --data_path './data/Annotations/PathVQA' \
        --feature_path './data/PathVQA'\
        --batch_size 128 \
        --visible \
        --method 'biomed' \
        --checkpoint './checkpoints/pathvqa/biomed_freeze/ckpt_best_model.pth'


python3 src/test_peir.py \
        --dataset 'pathvqa' \
        --data_path './data/Annotations/PathVQA' \
        --feature_path './data/PathVQA'\
        --batch_size 32 \
        --visible \
        --method 'biomed' \
        --checkpoint './checkpoints/peir/biomed_freeze/ckpt_best_model.pth'