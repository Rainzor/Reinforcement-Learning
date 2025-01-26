python train.py -m ac -n 1000 -e 'CartPole-v1'
python train.py -m ac-off-policy -e 'CartPole-v1'
python train.py -m ppo -e 'CartPole-v1'
python train.py -m sac -e 'CartPole-v1'

python train.py -m sac --continuous -e 'Pendulum-v0' 
python train.py -m ddpg --continuous -e 'Pendulum-v1'
python train.py -m td3 --continuous -e 'Pendulum-v1'