# F1tenth Race Agent and Toolbox

This repo contains an agent for the F1tenth head-to-head final race, which can track an optimal trajectory and overtake the opponent automatically, and a toolbox which contains an opponent-agent which is a ros2 node called `opp_agent`, and a trajectory generator called `traj_killer`.

## Installation

### Download the package

To use this agent and toolbox in your workspace, you need to put them into your workspace, and then in your workspace, run the following command to build the package.

```
$ colcon build --packages-select final_race_pkg
$ source install/setup.bash
```

After that, the structure of your workspace should be like this:

```
├── build
├── install
├── log
└── src/final_race_pkg
    ├── final_race_pkg
        ├── final_race_pkg
        ├── __init__.py
        ├── main_agent.py
        ├── opp_agent.py
        ├── traj_killer_script.py
        ├── traj_killer.py
        ├── waypoints1.json
        └── waypoints2.json
    ├── resource
    ├── test
    ├── package.xml
    ├── setup.cfg
    └── setup.py
```

## Traj_killer

This a GUI app to adjust trajectory interactively, there are two versions: script version and ROS2 node version.

### traj_killer_script.py

The GUI app is built based on PyQt5 under python3.8, to run this node, make sure you have installed all the required packages, you can do this by the following command:

```
# in your workspace
$ pip install -r requirements.txt
```

Then run the script in the folder where it is:

```
$ cd src/final_race_pkg/final_race_pkg/
$ python3 traj_killer_scripts.py
```
The two json file are used to memorize your configs, it will be tried to load every time the app is launched, so that you do not need to input all the waypoints and parameters every time. If you want to set up your own waypoints, just delete them, you will get an empty input area and you can add waypoints yourself.
