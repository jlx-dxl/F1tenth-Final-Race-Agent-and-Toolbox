# F1tenth Race Agent and Toolbox

This repo contains an agent for the F1tenth head-to-head final race, which can track an optimal trajectory and overtake the opponent automatically, and a toolbox which contains an opponent-agent which is a ros2 node called `opp_agent`, and a trajectory generator called `traj_killer`.

## Installation

### Download the package

To use this agent and toolbox in your workspace, you need to clone them into your

## Traj_killer

This a GUI app to adjust trajectory interactively

```
├── build
├── install
├── log
└── src
    ├── config
    ├── csv
    ├── dummy_car
    ├── trajectory_generator
    ├── lane_follow
    ├── maps
    ├── opponent_predictor
    ├── scripts
    └── <YOUR OTHER PACKAGES>
```