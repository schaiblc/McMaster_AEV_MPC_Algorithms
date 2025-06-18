# This Repo contains the source code used for implementation of the STLMPC, QBMPC, P-STLMPC & P-QBMPC algorithms
<!-- Add respective publications, brief descriptions -->
## STLMPC
Successive local tracking lines are generated via QPs (https://github.com/liuq/QuadProgpp), then a nonlinear SLSQP solver (https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#slsqp) performs optimization over a future time horizon, yielding a predicted trajectory. The first free control inputs are applied and the process repeats each control step. Dynamic obstacle avoidance is integrated via a CNN based on the YOLO architecture where EKF tracking is integrated into path planning for multi-vehicle contexts. Both implementations of STLMPC assuming a constant velocity and varying velocity are given (the varying velocity scheme encourages higher speeds for racing).

## QBMPC
A fourth order Bezier curve is used for simultaneous path planning and control, removing the need for a tracking path. Now, a potential field function is used to avoid local obstacles and constraints on vehicle dynamics are directly incorporated into the curve's formulation. An arbitrary future time horizon can be used for path planning while maintaining the fixed parameterization of the Bezier curve. An SLSQP solver is again used for the nonlinear optimization and control inputs are derived from the curve shape (governed by the control points) and subsequently applied at each control step.

## P-STLMPC & P-QBMPC
These adaptive pursuit algorithms modify the prior algorithms to fit the case of cooperative multi-vehicle navigation in a modular leader-follower scheme which can be extended to more complex fleet formations with more vehicles. Objective terms that achieve both safe navigation and pursuit in arbitrary formation are used and weighted dynamically based on the nearest obstacle proximity to the predicted trajectory over time. If the nearest obstacle proximity is low over a sustained time, safe navigation is prioritized meanwhile if the minimum obstacle proximity becomes high over a sustained period, increased license is granted to achieve pursuit in formation (where path safety based on nearby obstacles becomes less of a concern). This process also requires detections via YOLO and tracking via EKF where a minimum following distance is maintained between vehicles for safety, incorporated into the optimization which is still uses the SLSQP solver. No fixed leader is required, inter-vehicle communication does not occur and thus, vehicles can independently join or break formation dynamically subject to local environmental safety conditions.


## Additional Notes
* The navigation nodes for each of the listed algorithms are provided in the node directory.
* The f1tenth_simulator framework for ROS1 (https://github.com/f1tenth/f1tenth_simulator) was used for implementation of these local path planning techniques.
* Nonholonomic constraints are assumed according to the kinematic bicycle model, tunable parameters are given in params.yaml.
* AMCL localization can be enabled in params.yaml as can vehicle detection via YOLO for dynamic obstacle avoidance and vehicle pursuit.
* The specific navigation algorithm used can be selected in the simulator.launch and experiment.launch files for testing in simulation and experimental environments respectively.
* simulator.cpp can be modified to produce an arbitrary detected vehicle path in simulation for testing of dynamic obstacle avoidance and pursuit in simulation.
* Required packages are indicated in the source code, specifically in the package.xml & CMakeLists.txt files.
* Operation should be first confirmed in simulation to mitigate the risk of vehicle damage while testing in experimentation.


<!-- Include the YOLO CNN if possible -->
<!-- Include video of operation? Both simulation and experimental? -->
