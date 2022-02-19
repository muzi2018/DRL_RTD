# DRL_RTD
This repository contains code for the paper,"Improving the Ability to Traverse Crowded Environment Safely Using Deep Reinforcement Learing".

## Installation Requirements

To run the code in this repository, you will need the following:

1. gym 0.15.7
2. ipopt 3.13.4
3. numpy 1.19.2
4. pytorch 1.3.1

## Running

1. open the working directory

```bash
cd ./spinup/my_env
```

2. then run the following command to starting training

```bas
python -m spinup.run ppo --env FRSEnv-v0 --clip_ratio 0.1
```

