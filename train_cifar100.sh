export CUBLAS_WORKSPACE_CONFIG=:4096:8
DATASET="cifar100"

for seed in 2021 2022 2023
do 
    ALG="adaboost"
    python train.py --dataset $DATASET --data_root data/ --alg $ALG  --seed $seed \
                    --save_file ${DATASET}_${ALG}.mat \
                    --iters_per_epoch 2000 --epochs 5 \
                    --scheduler 1000,1700 --warmup 20

    ALG="raigame"
    python train.py --dataset $DATASET --data_root data/ --alg $ALG  --seed $seed \
                    --save_file ${DATASET}_${ALG}.mat \
                    --iters_per_epoch 2000 --epochs 5 \
                    --alpha 0.95 \
                    --type greedy --constraints chi2 --mult 50 \
                    --scheduler 1000,1700 --warmup 20 --gen_adaboost
    
    ALG="raigame"
    python train.py --dataset $DATASET --data_root data/ --alg $ALG  --seed $seed \
                    --save_file ${DATASET}_${ALG}.mat \
                    --iters_per_epoch 2000 --epochs 5 \
                    --alpha 0.95 \
                    --type greedy --constraints chi2 --mult 50 \
                    --scheduler 1000,1700 --warmup 20 --chi2
    
    ALG="raigame"
    python train_gdro.py --dataset $DATASET --data_root data/ --alg $ALG  --seed $seed \
                    --save_file ${DATASET}_${ALG}.mat \
                    --iters_per_epoch 2000 --epochs 5 \
                    --alpha 0.7 \
                    --type greedy_gdro --constraints chi2 --mult 10 \
                    --scheduler 1000,1700 --warmup 20 --dec 1e-3,1e-4 --gen_adaboost
    
    ALG="raigame"
    python train_gdro.py --dataset $DATASET --data_root data/ --alg $ALG  --seed $seed \
                    --save_file ${DATASET}_${ALG}.mat \
                    --iters_per_epoch 2000 --epochs 5 \
                    --type greedy_gdro --constraints none --mult 10 \
                    --scheduler 1000,1700 --warmup 20 --dec 1e-2,1e-3

    ALG="raigame"
    python train_gdro.py --dataset $DATASET --data_root data/ --alg $ALG  --seed $seed \
                    --save_file ${DATASET}_${ALG}.mat \
                    --iters_per_epoch 2000 --epochs 5 \
                    --alpha 0.7 \
                    --type greedy_gdro --constraints none --mult 10 \
                    --scheduler 1000,1700 --warmup 20 \
                    --dec 0,0 --gen_adaboost
done