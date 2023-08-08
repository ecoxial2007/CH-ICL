python3 src/trainval.py \
        --dataset 'peir' \
        --data_path './data/Annotations/PEIR' \
        --feature_path './data/PEIR'\
        --batch_size 32 \
        --freeze \
        --epochs 20 \
        --d_input 768 \
        --method 'biomed'

python3 src/trainval.py \
        --dataset 'radvqa' \
        --data_path './data/Annotations/RadVQA' \
        --feature_path './data/RadVQA'\
        --batch_size 256 \
        --freeze \
        --d_input 768 \
        --method 'pubmed'

python3 src/trainval.py \
        --dataset 'slakevqa' \
        --data_path './data/Annotations/Slake' \
        --feature_path './data/Slake'\
        --batch_size 256 \
        --freeze \
        --d_input 768 \
        --method 'biomed'

python3 src/trainval.py \
        --dataset 'pathvqa' \
        --data_path './data/Annotations/PathVQA' \
        --feature_path './data/PathVQA'\
        --batch_size 128 \
        --freeze \
        --d_input 768 \
        --method 'biomed'
