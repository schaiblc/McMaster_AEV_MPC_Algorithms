# This repository contains the source code used for the implementation of the STLMPC, QBMPC, P-STLMPC & P-QBMPC algorithms, which achieve safe local path planning in multi-vehicle environments in the absence of a global planner, map and known goal location
<!-- Add respective publications, brief descriptions -->
## STLMPC
Successive local tracking lines are generated via QPs (https://github.com/liuq/QuadProgpp), then a nonlinear SLSQP solver (https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#slsqp) performs optimization over a future time horizon, yielding a predicted trajectory. The first free control inputs are applied, and the process repeats each control step. Dynamic obstacle avoidance is integrated via a CNN based on the YOLO architecture, where EKF tracking is integrated into path planning for multi-vehicle contexts. Both implementations of STLMPC, assuming a constant velocity and varying velocity, are given (the varying velocity scheme encourages higher speeds for racing).

**Use the clickable thumbnails below to see the STLMPC algorithm in action:**
<table>
  <tr>
    <td>
      <a href="https://www.youtube.com/watch?v=hOUxxvMQrGM">
        <img src="https://img.youtube.com/vi/hOUxxvMQrGM/0.jpg" width="450">
      </a>
    </td>
    <td>
      <a href="https://www.youtube.com/watch?v=uKgcKcMBytk">
        <img src="https://img.youtube.com/vi/uKgcKcMBytk/0.jpg" width="450">
      </a>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <a href="https://www.youtube.com/watch?v=yKPFWdbwx-4">
        <img src="https://img.youtube.com/vi/yKPFWdbwx-4/0.jpg" width="400">
      </a>
    </td>
  </tr>
</table>



## QBMPC
A fourth-order Bezier curve is used for simultaneous path planning and control, removing the need for a tracking path. Now, a potential field function is used to avoid local obstacles, and constraints on vehicle dynamics are directly incorporated into the curve's formulation. An arbitrary future time horizon can be used for path planning while maintaining the fixed parameterization of the Bezier curve. An SLSQP solver is again used for the nonlinear optimization, and control inputs are derived from the curve shape (governed by the control points) and subsequently applied at each control step.

**Use the clickable thumbnails below to see the QBMPC algorithm in action:**
<table>
  <tr>
    <td>
      <a href="https://www.youtube.com/watch?v=3j0edNW95D0">
        <img src="https://img.youtube.com/vi/3j0edNW95D0/0.jpg" width="450">
      </a>
    </td>
    <td>
      <a href="https://www.youtube.com/watch?v=m4K5vlIFxEA">
        <img src="https://img.youtube.com/vi/m4K5vlIFxEA/0.jpg" width="450">
      </a>
    </td>
  </tr>
</table>



## P-STLMPC & P-QBMPC
These adaptive pursuit algorithms modify the prior algorithms to fit the case of cooperative multi-vehicle navigation in a modular leader-follower scheme, which can be extended to more complex fleet formations with more vehicles. Objective terms that achieve both safe navigation and pursuit in arbitrary formation are used and weighted dynamically based on the nearest obstacle's proximity to the predicted trajectory over time. If the clearance to the nearest obstacle is low over a prolonged time, safe navigation is prioritized; meanwhile, if the minimum obstacle clearance becomes high over a sustained period, increased license is granted to achieve pursuit in formation (where path safety based on nearby obstacles becomes less of a concern). This process also requires detections via YOLO and tracking via EKF, where a minimum following distance is maintained between vehicles for safety, incorporated into the optimization which still uses the SLSQP solver. No fixed leader is required, inter-vehicle communication does not occur and thus, vehicles can independently join or break formation dynamically subject to local environmental safety conditions.

**Use the clickable thumbnails below to see the P-STLMPC & P-QBMPC algorithms in action:**
<table>
  <tr>
    <td>
      <a href="https://www.youtube.com/watch?v=49ws64lPL-c">
        <img src="https://img.youtube.com/vi/49ws64lPL-c/0.jpg" width="450">
      </a>
    </td>
    <td>
      <a href="https://www.youtube.com/watch?v=zwTHNDGbHSE">
        <img src="https://img.youtube.com/vi/zwTHNDGbHSE/0.jpg" width="450">
      </a>
    </td>
  </tr>
</table>



## Additional Notes
* The navigation nodes for each of the listed algorithms are provided in the node directory, as well as a non-predictive PD approach which uses a single tracking line from STLMPC (for comparison).
* The f1tenth_simulator framework for ROS1 (https://github.com/f1tenth/f1tenth_simulator) was used for the implementation of these local path planning techniques.
* Nonholonomic constraints are assumed according to the kinematic bicycle model, the full set of tunable parameters is given in params.yaml.
* AMCL localization can be enabled in params.yaml as can vehicle detection via YOLO for dynamic obstacle avoidance and vehicle pursuit.
* The specific navigation algorithm used can be selected in the simulator.launch and experiment.launch files for testing in simulation and experimental environments, respectively.
* simulator.cpp can be modified to produce an arbitrary detected vehicle path in simulation for testing of dynamic obstacle avoidance and pursuit in simulation.
* Required packages are indicated in the source code, specifically in the package.xml & CMakeLists.txt files.
* The custom trained YOLO v5s model for MacAEV detection is provided in release v1.0.0 (https://github.com/schaiblc/McMaster_AEV_MPC_Algorithms/releases/tag/v1.0.0) with the appropriate .engine & .onnx files. These files are referenced by path in YOLO.py via ~/catkin_ws/src/f1tenth_simulator/learning_models/Custom_yolo5.engine & ~/catkin_ws/src/f1tenth_simulator/learning_models/yolov5CustomTrained.onnx for real-time inference.
* Operation should be first confirmed in simulation to mitigate the risk of vehicle damage while testing in experimentation.

## Configurable Parameters
**General Path Planning Parameters**
| Parameter     | Description   | Default    |
|:--------------|:--------------|------------|
| wheelbase             | Front to rear car axle distance             | 0.287 m          |
| max_speed             | Maximum allowed forward velocity             | 3 m/s          |
| min_speed             | Minimum allowed forward velocity             | 0 m/s (STLMPC); 0.5 m/s (QBMPC)          |
| max_steering_angle             | Maximum steering angle magnitude             | 0.4189 rad          |
| max_accel             | Maximum acceleration/deceleration             | 2.5 m/s²          |
| max_steering_vel             | Maximum steering angle rate             | 3.2 rad/s          |
| angle_al             | Lower angular bound on left obstacle cluster             | 200/180*pi rad          |
| angle_bl             | Upper angular bound on left obstacle cluster             | 270/180*pi rad          |
| angle_br             | Lower angular bound on right obstacle cluster             | 90/180*pi rad          |
| angle_ar             | Upper angular bound on right obstacle cluster             | 160/180*pi rad          |
| safe_distance             | Range of obstacles that pose an immediate collision risk             | 2 m          |
<br>

**PD Additional Parameters**
| Parameter     | Description   | Default    |
|:--------------|:--------------|------------|
| k_p             | Proportional gain             | 3.5 s⁻²          |
| k_d             | Derivative gain             | 4.0 s⁻¹          |
<br>

**STLMPC Additional Parameters**
| Parameter     | Description   | Default    |
|:--------------|:--------------|------------|
| nMPC             | Number of tracking lines             | 2          |
| kMPC             | Samples per tracking line             | 8          |
| vehicle_velocity             | Desired constant velocity             | 1.5 m/s          |
| use_map             | Enable/disable AMCL, map obstacles             | 0 (Disabled)          |
| veh_det_length             | Detected vehicle's box outline length             | 0.5 m          |
| veh_det_width             | Detected vehicle's box outline width             | 0.4 m          |
| use_neural_net             | Enable/disable YOLO for vehicle detection             | 0 (Disabled)          |
| d_factor_STLMPC             | Distance objective term base weight             | 1          |
| d_dot_factor_STLMPC             | Distance derivative objective term base weight             | 30          |
| delta_factor_STLMPC             | Steering angle objective term base weight             | 1          |
<br>

**Variable-Velocity STLMPC Additional Parameters**
| Parameter     | Description   | Default    |
|:--------------|:--------------|------------|
| stop_distance             | Obstacle proximity at which vehicle stops             | 0.8 m          |
| stop_distance_decay             | Velocity slowdown limit decay rate             | 0.5 m          |
| theta_band_smooth             | Forward angular passband transition sharpness             | 200 rad⁻¹          |
| theta_band_diff             | Forward angular passband magnitude limit             | pi/8 rad          |
| vel_beta             | Soft minimum approximation sharpness             | 10 m⁻¹          |
| d_factor_STLMPC_vary_v             | Distance objective term base weight             | 1          |
| d_dot_factor_STLMPC_vary_v             | Distance derivative objective term base weight             | 30          |
| delta_factor_STLMPC_vary_v             | Steering angle objective term base weight             | 1          |
| vel_factor_STLMPC_vary_v             | Forward velocity objective term base weight             | 1          |
<br>

**QBMPC Additional Parameters**
| Parameter     | Description   | Default    |
|:--------------|:--------------|------------|
| bez_ctrl_pts             | Bezier curve control points             | 5 (for quartic Bezier curve)          |
| bez_curv_pts             | Discretized Bezier curve path points             | 10          |
| bez_alpha             | Potential field objective distance decay factor             | 5.5 m⁻²         |
| bez_beta             | Soft minimum approximation sharpness            | 10 m⁻¹          |
| bez_min_dist             | Minimum obstacle proximity permitted             | 0.3 m          |
| bez_t_end             | Predicted trajectory time horizon             | 1.5 s          |
| obs_sep             | Minimum separation between subsampled obstacles              | 0.2 m          |
| max_obs             | Maximum size of the subsampled obstacle set             | 50          |
<br>

**Pursuit MPC Additional Parameters**
| Parameter     | Description   | Default    |
|:--------------|:--------------|------------|
| A             | B             | C          |
| D             | E             | F          |
<br>

**P-STLMPC Additional Parameters**
| Parameter     | Description   | Default    |
|:--------------|:--------------|------------|
| A             | B             | C          |
| D             | E             | F          |
<br>

**P-QBMPC Additional Parameters**
| Parameter     | Description   | Default    |
|:--------------|:--------------|------------|
| A             | B             | C          |
| D             | E             | F          |
