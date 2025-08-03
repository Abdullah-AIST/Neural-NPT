

# Neural-NPT: A Reinforcement Learning Perspective to Dynamic Non-Prehensile Object Transportation

<!-- [xx](xx)<sup>1</sup>
<sup>1</sup> xx, <sup>2</sup> xxx -->


Submitted to IEEE Robotics and Automation Letters (RA-L).

[Paper](https://abdullah-aist.github.io/Neural-NPT/) | [Arxiv](https://abdullah-aist.github.io/Neural-NPT/) | [Video](https://www.youtube.com/watch?v=MJLgvKNcebw) | [Website](https://abdullah-aist.github.io/Neural-NPT/)

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)


## Installation

1. Follow IsaacLab installation guide https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html . 

    Our setup is based Ubuntu 22.4, IsaacSim 4.5, and IsaacLab 2.1.0. We use RL_games for training.

2. Source the neural_npt task -- The task was developed using IsaacLab template (https://isaac-sim.github.io/IsaacLab/v2.1.0/source/overview/developer-guide/template.html)
    ```
    python -m pip install -e source/neural_npt/
    ```

3. Install other dependencies -- for UR5e control, we use [UR-RTDE](https://sdurobotics.gitlab.io/ur_rtde/) package 
    ```bash
    pip install --user ur_rtde
    ```

## Train & Play & Eval

### Train
* Pretrained weights are included for the four policies.
```bash
# in the root directory of Neural-NPT
python scripts/train.py --task=Neural_NPT --num_envs 4096 --headless --experiment_name Optimal --seed 0  
#python scripts/train.py --task=Neural_NPT --num_envs 4096 --headless --experiment_name Optimal_DG --seed 0     # Make neccessary changes to reward
#python scripts/train.py --task=Neural_NPT --num_envs 4096 --headless --experiment_name COM_Robust --seed 0     # Make neccessary changes to COM randomization
#python scripts/train.py --task=Neural_NPT --num_envs 4096 --headless --experiment_name Multi_Robust --seed 0   # Make neccessary changes to object position randomization  
```

### Play
```bash
# in the root directory of Neural-NPT  -- Make sure to set training flag to False
python scripts/play.py --task=Neural_NPT --num_envs 32 --experiment_name Optimal --seed 0
#python scripts/play.py --task=Neural_NPT --num_envs 32 --experiment_name Optimal_DG --seed 0     # Make neccessary changes to reward
#python scripts/play.py --task=Neural_NPT --num_envs 32 --experiment_name COM_Robust --seed 0     # Make neccessary changes to COM randomization
#python scripts/play.py --task=Neural_NPT --num_envs 32 --experiment_name Multi_Robust --seed 0   # Make neccessary changes to object position randomization  

# Use playZero during enviroment setup for debugging.
# python scripts/playZero.py --task=Neural_NPT --num_envs 32 --experiment_name Optimal --seed 0

```

### Eval
```bash
# in the root directory of Neural-NPT  -- Make sure to set training flag to False
# For the environment configuration file, select a target object and a set of evaluation states (Sim, Real, Unseen ...) 
python scripts/Eval.py --task=Neural_NPT --num_envs 32 --headless --experiment_name Optimal --seed 0 --envSeed 0 --targetObject woodBlock
#python scripts/Eval.py --task=Neural_NPT --num_envs 32 --headless --experiment_name Optimal_DG --seed 0 --envSeed 0 --targetObject woodBlock   # Make neccessary changes to reward
#python scripts/Eval.py --task=Neural_NPT --num_envs 32 --headless --experiment_name COM_Robust --seed 0 --envSeed 0 --targetObject woodBlock   # Make neccessary changes to COM randomization
#python scripts/Eval.py --task=Neural_NPT --num_envs 32 --headless --experiment_name Multi_Robust --seed 0 --envSeed 0 --targetObject woodBlock # Make neccessary changes to object position randomization  
```
### Deploy
1. Process the trajectories using the "processTrajs_real.ipynb" notebook to analyze and generate neccessary trajectories.
2. Deploy based of UR-RTDE package 

```bash
# in the root directory of Neural-NPT  -- Make sure to set training flag to False
# For the environment configuration file, select a target object and a set of evaluation states (Sim, Real, Unseen ...) 
python scripts/deploy.py
```


## Citation
If you use this code in your research, please cite our paper:
```bibtex
@article{neuralNPT,
  title={Neural-NPT: A Reinforcement Learning Perspective to Dynamic Non-Prehensile Object Transportation},
  author={author one},
  journal={RAL},
  year={2025}
}
```

## License

This codebase is under [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/deed.en). You may not use the material for commercial purposes, e.g., to make demos to advertise your commercial products.


## Acknowledgements
- [IsaacLab](https://github.com/isaac-sim/IsaacLab): We use the `isaaclab` library for the RL training and evaluation.
- [UR-RTDE](https://sdurobotics.gitlab.io/ur_rtde/): We use the `UR-RTDE` package for UR5e real-time control.

## Contact

Feel free to open an issue or discussion if you encounter any problems or have questions about this project.
