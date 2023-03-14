# 6 task RL tasks: [ k-fu, boxing, JB, krull, RR, SpaceIn]

# # PPO AGENT : SGP      
CUDA_VISIBLE_DEVICES=$1 python main_gpm_rl.py --algo ppo --use-gae --custom_adam --network_bias \
    --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 \
    --num-processes 16 --num-steps 125 --num-mini-batch 8 \
    --log-interval 100 --eval-interval 100 \
    --use-linear-lr-decay --entropy-coef 0.01 \
    --approach sgp --experiment atari --threshold 0.995 --seed 1 --threshold_inc 0.0 \
    --date cfg1 --log-dir './logs/SGP/cfg_1/' --scale_coff 25 --gpm-mini-batch 32 #--num-env-steps 4000


# # PPO AGENT : GPM   
# CUDA_VISIBLE_DEVICES=$1 python main_gpm_rl.py --algo ppo --use-gae --custom_adam --network_bias \
#     --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 \
#     --num-processes 16 --num-steps 125 --num-mini-batch 8 \
#     --log-interval 100 --eval-interval 100 \
#     --use-linear-lr-decay --entropy-coef 0.01 \
#     --approach gpm --experiment atari --threshold 0.995 --seed 1 --threshold_inc 0.0 \
#     --date cfg1 --log-dir './logs/GPM/cfg_1/' --gpm-mini-batch 32 #--num-env-steps 4000


# # PPO AGENT : BLIP  
# CUDA_VISIBLE_DEVICES=$1 python main_rl.py --algo ppo --use-gae \
#     --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 \
#     --num-processes 16 --num-steps 125 --num-mini-batch 8 \
#     --log-interval 100 --eval-interval 100 \
#     --use-linear-lr-decay --entropy-coef 0.01 \
#     --approach blip --experiment atari --F-prior 5e-18 --seed 1 \
#     --date cfg1 --log-dir './logs/BLIP/cfg_1/' #--num-env-steps 4000


# # PPO AGENT : EWC   
# CUDA_VISIBLE_DEVICES=$1 python main_rl.py --algo ppo --use-gae \
#     --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 \
#     --num-processes 16 --num-steps 125 --num-mini-batch 8 \
#     --log-interval 100 --eval-interval 100 \
#     --use-linear-lr-decay --entropy-coef 0.01 \
#     --approach ewc --experiment atari --ewc-lambda 5000 --seed 1 --ewc-online True \
#     --date cfg1 --log-dir './logs/EWC/cfg_1/' #--num-env-steps 4000

# # PPO AGENT : FT  
# CUDA_VISIBLE_DEVICES=$1 python main_rl.py --algo ppo --use-gae \
#     --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 \
#     --num-processes 16 --num-steps 125 --num-mini-batch 8 \
#     --log-interval 100 --eval-interval 100 \
#     --use-linear-lr-decay --entropy-coef 0.01 \
#     --approach fine-tuning --experiment atari --seed 1 \
#     --date cfg1 --log-dir './logs/FT/cfg_1/' #--num-env-steps 4000