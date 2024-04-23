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
├── README.md
├── requirements.txt
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
$ python3 traj_killer_script.py
```
The two json file are used to memorize your configs, the app will try to load them every time it is launched, so that you do not need to input all the waypoints and parameters every time. If you want to set up your own waypoints, just delete them, you will get an empty input area like below and you can add waypoints yourself (remember you need to click the `Save Curve` botton to save the json files).

![empty interface.png](https://img2.imgtp.com/2024/04/23/2bs76gTt.png)

If the following shows up, it means the json files are loaded successfully, then you can directly click the `Plot All Curves` to see the plot.

![initialed interface.png](https://img2.imgtp.com/2024/04/23/RfUVBaAv.png)

Then you can change the coordinates, weights of the waypoints and parameters of the spline curve with interactive input boxes, slides and bottons. Every time you click a botton or change the value of a slide, the plot will update itself.

### traj_killer.py

This is the ROS2 node version of this GUI app, it will publish the trajectory into two topics called `traj_inner` (corresponding to Curve1) and `traj_outer` (corresponding to Curve2). Run the node by the below command:

```
$ ros2 run final_race_pkg traj_killer
```

The user interface is the same as the script version, but remember only when you click the `Save Curve` botton, will the trajectory be published.

Then if you launch the f1tenth_gym and run the two agents, you can see the trajectories are changed once you publish any new trajectory. (You need to copy the two json file into the workspace folder to get the node initialized)

```
# in different terminals, run:
$ ros2 launch f1tenth_gym_ros gym_bridge_launch.py 
$ ros2 run final_race_pkg main_agent
$ ros2 run final_race_pkg opp_agent 
```

![ROS node running.png](https://img2.imgtp.com/2024/04/23/Sbw65zIT.png)

