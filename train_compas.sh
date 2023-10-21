export CUBLAS_WORKSPACE_CONFIG=:4096:8
DATASET="compas"

for seed in 2021 2022 2023
do 
    ALG="adaboost"
    python train.py --dataset $DATASET --data_root data/ --alg $ALG  --seed $seed \
                    --save_file ${DATASET}_${ALG}.mat \
                    --iters_per_epoch 500 --epochs 5 \
                    --scheduler 400 --warmup 3

    ALG="raigame"
    python train.py --dataset $DATASET --data_root data/ --alg $ALG  --seed $seed \
                    --save_file ${DATASET}_${ALG}.mat \
                    --iters_per_epoch 500 --epochs 5 \
                    --alpha 0.95 \
                    --type greedy --constraints chi2 --mult 50 \
                    --scheduler 400 --warmup 3 --gen_adaboost
    
    ALG="raigame"
    python train.py --dataset $DATASET --data_root data/ --alg $ALG  --seed $seed \
                    --save_file ${DATASET}_${ALG}.mat \
                    --iters_per_epoch 500 --epochs 5 \
                    --alpha 0.95 \
                    --type greedy --constraints chi2 --mult 50 \
                    --scheduler 400 --warmup 3 --chi2

    ALG="raigame"
    python train_gdro.py --dataset $DATASET --data_root data/ --alg $ALG  --seed $seed \
                    --save_file ${DATASET}_${ALG}.mat \
                    --iters_per_epoch 500 --epochs 20 \
                    --alpha 0.95 \
                    --type greedy_gdro --constraints chi2 --mult 50 \
                    --scheduler 400 --warmup 3 --dec 1e-1,1e-1 --gen_adaboost

done