@echo off
python train.py -m ac -n 1000 -e "CartPole-v1"
python train.py -m ac-off-policy -n 500 -e "CartPole-v1"
python train.py -m ppo -n 500 -e "CartPole-v1"
python train.py -m sac -n 500 -e "CartPole-v1" 

python train.py -m sac --continuous -e "Pendulum-v1"
python train.py -m ddpg --continuous -e "Pendulum-v1"
python train.py -m td3 --continuous -e "Pendulum-v1"