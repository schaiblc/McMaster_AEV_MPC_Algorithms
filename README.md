# This Repo contains the source code used for implementation of the STLMPC, QBMPC, P-STLMPC & P-QBMPC algorithms
<!-- Add respective publications -->

* The navigation nodes for each of the listed algorithms are provided in the node directory.
* The f1tenth_simulator framework for ROS1 (https://github.com/f1tenth/f1tenth_simulator) was used for implementation of these local path planning techniques.
* Nonholonomic constraints are assumed according to the kinematic bicycle model, tunable parameters are given in params.yaml.
* AMCL localization can be enabled in params.yaml as can vehicle detection via YOLO for dynamic obstacle avoidance and vehicle pursuit.
* The specific navigation algorithm used can be selected in the simulator.launch and experiment.launch files for testing in simulation and experimental environments respectively.
* simulator.cpp can be modified to produce an arbitrary detected vehicle path in simulation for testing of dynamic obstacle avoidance and pursuit in simulation.


<!-- Include the YOLO CNN if possible -->
<!-- Include video of operation? Both simulation and experimental? -->
