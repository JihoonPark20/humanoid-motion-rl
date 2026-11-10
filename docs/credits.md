# Credits and retained licenses

- **ZEST paper:** [Sleiman et al., Zero-shot Embodied Skill Transfer for Athletic Robot Control](https://arxiv.org/abs/2602.00401). This implementation is unofficial and is not endorsed by the authors.
- **Imported scaffold:** The workspace initially contained `legged-gym-mujoco`, derived from ETH Zurich's [legged_gym](https://github.com/leggedrobotics/legged_gym). Its CPU MuJoCo batch structure and G1 motor setup informed the new package. The original BSD notices are retained in [LICENSE](../LICENSE).
- **G1 XML and meshes:** [Unitree RL Mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab), revision `1425b15f73bd4095f0df53709d7c389c3eb9e790`; see [asset provenance](../zest/resources/robots/g1/unitree_mjlab.json), [Unitree license](../zest/resources/robots/UNITREE_LICENSE) and retained notices under [licenses](../licenses).
- **PPO:** [rsl_rl](https://github.com/leggedrobotics/rsl_rl), installed as a dependency, with its original license retained.
- **Default motion data:** [HOLD](https://github.com/rai-opensource/hold), released by RAI Institute and Boston Dynamics and used in [HuMBLE](https://arxiv.org/abs/2610.10489), under CC BY 4.0. This project uses the provided G1-retargeted trajectories. Dataset terms are separate from this project's code license; no dataset is included in source distributions or wheels.

The imported repository's license files are retained even when their associated optional code is no longer used. The original imported repository is not a runtime dependency and is not packaged.
