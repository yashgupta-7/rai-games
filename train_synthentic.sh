export CUBLAS_WORKSPACE_CONFIG=:4096:8
DATASET="synthetic"

for seed in 2021 2022 2023
do 
    ALG="adaboost"
    python train.py --dataset $DATASET --data_root data/ --alg $ALG  --seed $seed \
                    --save_file ${DATASET}_${ALG}.mat &

    ALG="lpboost"
    python train.py --dataset $DATASET --data_root data/ --alg $ALG  --seed $seed \
                    --save_file ${DATASET}_${ALG}.mat \
                    --alpha 0.7 &

    ALG="raigame"
    python train.py --dataset $DATASET --data_root data/ --alg $ALG  --seed $seed \
                    --save_file ${DATASET}_${ALG}.mat \
                    --alpha 0.7 --type greedy --constraints cvar  &
    
    ALG="raigame"
    python train.py --dataset $DATASET --data_root data/ --alg $ALG  --seed $seed \
                    --save_file ${DATASET}_${ALG}.mat \
                    --alpha 0.7 --type greedy --constraints cvar --gen_adaboost
    
done
wait