#define _USE_MATH_DEFINES //M_PI

#include "ros/ros.h"

#include "std_msgs/Bool.h"
#include "std_msgs/String.h"
#include "std_msgs/Int32MultiArray.h"
#include "sensor_msgs/LaserScan.h" //receive msgs from lidar
#include "sensor_msgs/Imu.h" //receive msgs from Imu
#include "geometry_msgs/Point.h"
#include "geometry_msgs/Pose2D.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h" //map localization via AMCL
#include "visualization_msgs/Marker.h" //plot marker line
#include "sensor_msgs/Image.h" // ros image
#include <vesc_msgs/VescStateStamped.h>
#include <std_msgs/Float64.h>

#include <tf/tf.h> //Quaternions
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>


#include "ackermann_msgs/AckermannDriveStamped.h" //Ackermann Steering

#include "nav_msgs/Odometry.h" //Odometer
#include <nav_msgs/OccupancyGrid.h> //Map
#include <f1tenth_simulator/YoloData.h> //Neural Network, vehicle detection msg

#include <string>
#include <vector>


//CV includes
#include <cv_bridge/cv_bridge.h>
#include <librealsense2/rs.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include  <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <realsense2_camera/Extrinsics.h>




//standard and external
#include <stdio.h>
#include <math.h> //cosf
#include <cmath> //M_PI, round
#include <sstream>
#include <algorithm>

#include <QuadProg++.hh>
#undef inverse // Remove the conflicting macro
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp> 
#include <nlopt.hpp>
#include <Eigen/Dense>

//C++ will auto typedef float3 data type
int nMPC=0; //Defined outside class to be used in predefined functions for nlopt MPC calculation
int kMPC=0;
double d_factor=1; //Change weighting of d vs d_dot vs delta terms in params.yaml
double d_dot_factor=30;
double delta_factor=1; //Using defaults here


struct float3
{
	float x;
	float y;
	float z;
};

struct plane
{
	float A;
	float B;
	float C;
	float D; 
};

struct tf_data
{
	double tf_x=0;
	double tf_y=0;
	double tf_theta=0;
	double tf_time=0;
};

struct vehicle_detection
{
	int init=0; //Whether the KF for this detetction has been initialized
	std::vector<int> bound_box={0,0,0,0}; //The CNN bounding box for the last cycle to try to match appropriately in the case of multi-detection
	//{miny, minx, maxy, maxx}
	std::vector<double> meas={0.0,0.0}; //The most recent measurement of x & y for the vehicle detected
	tf_data meas_tf; //The tf corresponding to the last measurement time, to align frames in KF

	int miss_fr=0; //Consecutive frames missed, if above a set threshold then stop tracking this vehicle
	int last_det=0; //Flag for identifying whether a measurement was made in the last cycle or not, affects how that time's KF works
	//Kalman Filter Parameters
	Eigen::VectorXd state = Eigen::VectorXd::Zero(5); //x, y, theta, vs, delta
	Eigen::MatrixXd cov_P = Eigen::MatrixXd::Zero(5,5); //Initial cov and proc noise are set upon initialization in yolo_callback

	//State, covariance as well as meas, proc noises which can evolve depending on conditions
	Eigen::MatrixXd proc_noise = cov_P;
	Eigen::MatrixXd meas_noise = Eigen::Vector2d(0.01, 0.01).asDiagonal();
	//State transition and observation matrices assumed equal for all detection structs so provided commonly outside struct

};


double myfunc(unsigned n, const double *x, double *grad, void *my_func_data) //NLOPT cost function
{
	//Gradient calculated based on three parts, d part, d_dot due to p_dot for both current and then next point (obj is only nonzero partial x & y)
	double (*track_line)[nMPC*kMPC] = (double (*)[nMPC*kMPC]) my_func_data; //track_line is now the normal double array
	double funcreturn=0; //Create objective function as the sum of d and d_dot squared terms (d_dot part assumes constant w)
	if(grad){
		for(int i=0;i<n;i++){
			grad[i]=0;
		}
	}
	for (int i=0;i<nMPC*kMPC;i++){
			funcreturn=funcreturn+d_factor*(pow(track_line[0][i]*x[2*nMPC*kMPC+i]+track_line[1][i]*x[3*nMPC*kMPC+i]+1,2)/(pow(track_line[0][i],2)+pow(track_line[1][i],2)));
			if(grad){
				grad[2*nMPC*kMPC+i]=d_factor*(2*track_line[0][i]*(track_line[0][i]*x[2*nMPC*kMPC+i]+track_line[1][i]*x[3*nMPC*kMPC+i]+1)/(pow(track_line[0][i],2)+pow(track_line[1][i],2)));
				grad[3*nMPC*kMPC+i]=d_factor*(2*track_line[1][i]*(track_line[0][i]*x[2*nMPC*kMPC+i]+track_line[1][i]*x[3*nMPC*kMPC+i]+1)/(pow(track_line[0][i],2)+pow(track_line[1][i],2)));
			}
			if(grad&&i>0){
				grad[2*nMPC*kMPC+i]=grad[2*nMPC*kMPC+i]+d_dot_factor*2*track_line[0][i-1]*(track_line[0][i-1]*(x[2*nMPC*kMPC+i]-x[2*nMPC*kMPC+i-1])+track_line[1][i-1]*(x[3*nMPC*kMPC+i]-x[3*nMPC*kMPC+i-1]))/(pow(track_line[0][i-1],2)+pow(track_line[1][i-1],2));
				grad[3*nMPC*kMPC+i]=grad[3*nMPC*kMPC+i]+d_dot_factor*2*track_line[1][i-1]*(track_line[0][i-1]*(x[2*nMPC*kMPC+i]-x[2*nMPC*kMPC+i-1])+track_line[1][i-1]*(x[3*nMPC*kMPC+i]-x[3*nMPC*kMPC+i-1]))/(pow(track_line[0][i-1],2)+pow(track_line[1][i-1],2));
			}
			if(i<nMPC*kMPC-1){
				funcreturn=funcreturn+d_dot_factor*pow(track_line[0][i]*(x[2*nMPC*kMPC+i+1]-x[2*nMPC*kMPC+i])+track_line[1][i]*(x[3*nMPC*kMPC+i+1]-x[3*nMPC*kMPC+i]),2)/(pow(track_line[0][i],2)+pow(track_line[1][i],2));
				if(grad){
					grad[2*nMPC*kMPC+i]=grad[2*nMPC*kMPC+i]-d_dot_factor*2*track_line[0][i]*(track_line[0][i]*(x[2*nMPC*kMPC+i+1]-x[2*nMPC*kMPC+i])+track_line[1][i]*(x[3*nMPC*kMPC+i+1]-x[3*nMPC*kMPC+i]))/(pow(track_line[0][i],2)+pow(track_line[1][i],2));
					grad[3*nMPC*kMPC+i]=grad[3*nMPC*kMPC+i]-d_dot_factor*2*track_line[1][i]*(track_line[0][i]*(x[2*nMPC*kMPC+i+1]-x[2*nMPC*kMPC+i])+track_line[1][i]*(x[3*nMPC*kMPC+i+1]-x[3*nMPC*kMPC+i]))/(pow(track_line[0][i],2)+pow(track_line[1][i],2));
				}
			}
			funcreturn=funcreturn+delta_factor*pow(x[nMPC*kMPC+i],2); //The scaling factor of this term may need to be param, depends on speed (tuning)
			if(grad){
				grad[i]=0; //Gradients wrt theta = 0
				grad[nMPC*kMPC+i]=2*delta_factor*x[nMPC*kMPC+i];
			}
	}
	return funcreturn;
}

void theta_equality_con(unsigned m, double *result, unsigned n, const double* x, double* grad, void* f_data){ //Theta kinematics equality
	double *opt_params = (double (*))f_data; //[0]-> velocity/sample; [1]-> wheelbase (l) parameter
	if(grad){
		for(int i=0;i<n*m;i++){
			grad[i]=0;
		}
	}
	for (int i=0;i<nMPC*kMPC-1;i++){
		result[i]=x[i+1]-x[i]-opt_params[0]/opt_params[1]*tan(x[nMPC*kMPC+i]);
		if(grad){
			grad[i*n+i]=-1;
			grad[i*n+i+1]=1;
			grad[i*n+nMPC*kMPC+i]=-opt_params[0]/opt_params[1]*pow(1/cos(x[nMPC*kMPC+i]),2);
		}
	}
}

void x_equality_con(unsigned m, double *result, unsigned n, const double* x, double* grad, void* f_data){ //X kinematics equality
	double *opt_params = (double (*))f_data; //[0]-> velocity/sample; [1]-> wheelbase (l) parameter
	if(grad){
		for(int i=0;i<n*m;i++){
			grad[i]=0;
		}
	}
	for (int i=0;i<nMPC*kMPC-1;i++){
		result[i]=x[2*nMPC*kMPC+i+1]-x[2*nMPC*kMPC+i]-opt_params[0]*cos(x[i]);
		if(grad){		
			grad[i*n+2*nMPC*kMPC+i]=-1;
			grad[i*n+2*nMPC*kMPC+i+1]=1;
			grad[i*n+i]=opt_params[0]*sin(x[i]);
		}
	}
}

void y_equality_con(unsigned m, double *result, unsigned n, const double* x, double* grad, void* f_data){ //Y kinematics equality
	double *opt_params = (double (*))f_data; //[0]-> velocity/sample; [1]-> wheelbase (l) parameter
	if(grad){
		for(int i=0;i<n*m;i++){
			grad[i]=0;
		}
	}
	for (int i=0;i<nMPC*kMPC-1;i++){
		result[i]=x[3*nMPC*kMPC+i+1]-x[3*nMPC*kMPC+i]-opt_params[0]*sin(x[i]);
		if(grad){
			grad[i*n+3*nMPC*kMPC+i]=-1;
			grad[i*n+3*nMPC*kMPC+i+1]=1;
			grad[i*n+i]=-opt_params[0]*cos(x[i]);
		}
	}
}

void delta_inequality_con(unsigned m, double *result, unsigned n, const double* x, double* grad, void* f_data){ //Delta kinematics inequality
	double *opt_params = (double (*))f_data; //[0]-> velocity/sample; [1]-> wheelbase (l) parameter; [2]-> max change in delta
	if(grad){
		for(int i=0;i<n*m;i++){
			grad[i]=0;
		}
	}
	for (int i=0;i<nMPC*kMPC-1;i++){
		result[2*i]=x[nMPC*kMPC+i+1]-x[nMPC*kMPC+i]-opt_params[2];
		result[2*i+1]=-x[nMPC*kMPC+i+1]+x[nMPC*kMPC+i]-opt_params[2];
		if(grad){
			grad[2*i*n+nMPC*kMPC+i]=-1;
			grad[2*i*n+nMPC*kMPC+i+1]=1;
			grad[(2*i+1)*n+nMPC*kMPC+i]=1;
			grad[(2*i+1)*n+nMPC*kMPC+i+1]=-1;
		}

	}
	result[2*nMPC*kMPC-2]=x[nMPC*kMPC]-opt_params[3]; //opt_params[3] is the last delta (from previous iteration)
	result[2*nMPC*kMPC-1]=-x[nMPC*kMPC]+opt_params[3]; //Can't change servo instantly so delta[0] is fixed
	if(grad){
		grad[(2*nMPC*kMPC-2)*n+nMPC*kMPC]=1;
		grad[(2*nMPC*kMPC-1)*n+nMPC*kMPC]=-1;
	}

}





class GapBarrier 
{
	private:
		ros::NodeHandle nf;


		//Subscriptions
		ros::Subscriber lidar;
		ros::Subscriber image, info, confidence; 
		ros::Subscriber imu;
		ros::Subscriber mux;
		ros::Subscriber vesc_state_sub;
		ros::Subscriber servo_sub;
		// ros::Subscriber odom;
		// ros::Subscriber localize;
		ros::Subscriber amcl_sub;
		ros::Subscriber tf_sub;
		ros::Subscriber map_sub;
		ros::Subscriber yolo_sub;
		ros::Subscriber cmdvel_sub;

		//More CV data members, used if use_camera is true
		ros::Subscriber depth_img;
		ros::Subscriber color_img;
		ros::Subscriber depth_info;
		ros::Subscriber color_info;
		ros::Subscriber cam_extrinsics;
		ros::Subscriber depth_img_confidence;
		sensor_msgs::LaserScan cv_ranges_msg;
		int cv_rows, cv_cols;
		xt::xarray<int> cv_sample_rows_raw;
		xt::xarray<int> cv_sample_cols_raw;
		
		


		
		//Publications
		ros::Publisher lidar_pub;
		ros::Publisher marker_pub;
		ros::Publisher mpc_marker_pub;
		ros::Publisher wall_marker_pub;
		ros::Publisher lobs;
		ros::Publisher robs;
		ros::Publisher vehicle_detect;
		ros::Publisher driver_pub;
		ros::Publisher cv_ranges_pub;


		
		//topics
		std::string depth_image_topic, depth_info_topic, cv_ranges_topic, depth_index_topic, color_image_topic, color_info_topic, cam_extr_topic,
		depth_points_topic,lidarscan_topic, drive_topic, odom_topic, mux_topic, imu_topic, map_topic, yolo_data_topic;

		//time
		double current_time = ros::Time::now().toSec();
		double prev_time = current_time;
		double time_ref = 0.0; 
		double heading_beam_angle;

		//lidar-preprocessing
		int scan_beams; double right_beam_angle, left_beam_angle;
		double right_beam_angle_MPC, left_beam_angle_MPC;
		int right_ind_MPC, left_ind_MPC;
		int ls_str, ls_end, ls_len_mod, ls_len_mod2; double ls_fov, angle_cen, ls_ang_inc;
		double max_lidar_range, safe_distance;

		//obstacle point detection
		std::string drive_state; 
		double angle_bl, angle_al, angle_br, angle_ar;
		int n_pts_l, n_pts_r; double max_lidar_range_opt;

		//walls
		double tau;
		std::vector<double> wl0; std::vector<double> wr0;
		int optim_mode;


		//markers
		visualization_msgs::Marker marker;
		visualization_msgs::Marker mpc_marker;
		visualization_msgs::Marker wall_marker;
		visualization_msgs::Marker lobs_marker;
		visualization_msgs::Marker robs_marker;
		visualization_msgs::Marker vehicle_detect_path;

		//steering & stop time
		double vel;
		double CenterOffset, wheelbase;
		double stop_distance, stop_distance_decay;
		double k_p, k_d;
		double max_steering_angle;
		double vehicle_velocity; double velocity_zero;
		
		double stopped_time;
		double stop_time1, stop_time2;

		double yaw0, dtheta; double turn_angle; 
		double turn_velocity;

		double max_servo_speed;

		//MPC parameters
		//int nMPC, kMPC;
		double angle_thresh;
		std::vector<double> deltas, thetas, x_vehicle, y_vehicle;
		double last_delta;
		int num1=0;
		int num2=0;
		int missing_pts=0;
		double velocity_MPC;
		double default_dt;
		int startcheck=0;
		int forcestop=0;
		std::vector<float> local_ranges;
		double robx=0;
		double roby=0;
		double robtheta=0;

		double speed_to_erpm_gain, speed_to_erpm_offset;
		double steering_angle_to_servo_gain, steering_angle_to_servo_offset;
		std_msgs::Float64 last_servo_state;
		double vel_adapt=0.1;

		double testx, testy, testtheta;

		//odom and map transforms for map localization and tf of occupancy grid points
		double mapx=0, mapy=0, maptheta=0;
		double odomx=0, odomy=0, odomtheta=0;
		double locx=0, locy=0, loctheta=0;
		double simx=0, simy=0, simtheta=0;
		std::vector<tf_data> past_tf;

		//MPC MAP localization parameters
		std::vector<std::vector<double>> map_pts;
		int map_saved=0;
		double map_thresh;
		int use_map=0; //Whether we use the pre-defined map as part of MPC

		int yolo_rows; //Rows & columns preset for depth camera
		int yolo_cols;
		std::vector<vehicle_detection> car_detects;
		double lastcmdvel=0;
		double lastcmdsteer=0;

		Eigen::Matrix<double, 2, 5> meas_observability; //Measurement observability matrix, used in KF & same for all

		double lastx=0, lasty=0, lasttheta=0;

		int use_neural_net=0; //Whether one of the neural networks is being used for vehicle detection

		double veh_det_length=0;
		double veh_det_width=0;

		std::vector<std::vector<double>> save_map;
		int callbackcount=0;

		ros::Time timestamp_tf1; ros::Time timestamp_tf2;
		ros::Time timestamp_cam1; ros::Time timestamp_cam2;

		//imu
		double imu_roll, imu_pitch, imu_yaw;

		//mux
		int nav_mux_idx; int nav_active; 

		//odom
		double yaw;
		double max_accel;
		double max_speed;


		//camera and cv
		
		int use_camera;
		double min_cv_range;
        double max_cv_range;
        double cv_distance_to_lidar;
        double num_cv_sample_rows;
        double num_cv_sample_cols;

        double cv_ground_angle;
        double cv_lidar_range_max_diff;
        double camera_height;
		double camera_min,camera_max;
        double cv_real_to_theo_ground_range_ratio;
        double cv_real_to_theo_ground_range_ratio_near_horizon;
        double cv_ground_range_decay_row;
        double cv_pitch_angle_hardcoded;


		rs2_intrinsics intrinsics_depth;
		rs2_intrinsics intrinsics_color;
		rs2_extrinsics extrinsics;
		bool intrinsics_d_defined; bool intrinsics_c_defined;
		sensor_msgs::Image cv_image_data;
		bool cv_image_data_defined;

		std::vector<sensor_msgs::ImageConstPtr> depth_imgs;

		//ground plane parameters
		float cv_groundplane_max_height; 
		float cv_groundplane_max_distance; 

		

	public:
		
		GapBarrier(){

			nf = ros::NodeHandle("~");
			// topics	
			nf.getParam("depth_image_topic", depth_image_topic);
			nf.getParam("rgb_image_topic", color_image_topic);
			nf.getParam("depth_info_topic", depth_info_topic);
			nf.getParam("rgb_info_topic", color_info_topic);
			nf.getParam("cam_extrinsics_topic", cam_extr_topic);
			nf.getParam("cv_ranges_topic", cv_ranges_topic);
			nf.getParam("depth_index_topic", depth_index_topic);
			nf.getParam("depth_points_topic", depth_points_topic);
			nf.getParam("scan_topic", lidarscan_topic);
			nf.getParam("nav_drive_topic", drive_topic);
			nf.getParam("odom_topic", odom_topic);
			nf.getParam("mux_topic", mux_topic);
			nf.getParam("imu_topic", imu_topic);
			nf.getParam("map_topic", map_topic);
			nf.getParam("yolo_data_topic", yolo_data_topic);

			nf.getParam("speed_to_erpm_gain", speed_to_erpm_gain);
			nf.getParam("speed_to_erpm_offset", speed_to_erpm_offset);



			//lidar params
			nf.getParam("scan_beams", scan_beams);
			nf.getParam("right_beam_angle", right_beam_angle);
			nf.getParam("left_beam_angle", left_beam_angle);
			nf.getParam("scan_range", max_lidar_range);
			nf.getParam("safe_distance", safe_distance);

			//lidar init
			right_beam_angle_MPC = right_beam_angle-M_PI; //-M_PI/2;
			left_beam_angle_MPC = left_beam_angle-M_PI; //M_PI/2;
			ls_ang_inc = 2*M_PI/scan_beams;
			ls_str = int(round(scan_beams*right_beam_angle/(2*M_PI)));
			ls_end = int(round(scan_beams*left_beam_angle/(2*M_PI)));
			ls_len_mod = ls_end-ls_str+1;
			ls_fov = ls_len_mod*ls_ang_inc;
			angle_cen = ls_fov/2;
			ls_len_mod2 = 0;	


			//obstacle point detection
			drive_state = "normal";
			nf.getParam("angle_bl", angle_bl);
			nf.getParam("angle_al", angle_al);
			nf.getParam("angle_br", angle_br);
			nf.getParam("angle_ar", angle_ar);
			nf.getParam("n_pts_l", n_pts_l);
			nf.getParam("n_pts_r", n_pts_r);
			nf.getParam("max_lidar_range_opt", max_lidar_range_opt);
			nf.getParam("heading_beam_angle", heading_beam_angle);

			//walls
			nf.getParam("tau", tau); 
			wl0 = {0.0, -1.0}; wr0 = {0.0, 1.0};
			nf.getParam("optim_mode", optim_mode);


			//steering init
			nf.getParam("CenterOffset", CenterOffset);
			nf.getParam("wheelbase", wheelbase);
			nf.getParam("stop_distance", stop_distance);
			nf.getParam("stop_distance_decay", stop_distance_decay);
			nf.getParam("k_p", k_p);
			nf.getParam("k_d", k_d);
			nf.getParam("max_steering_angle", max_steering_angle);
			nf.getParam("vehicle_velocity", vehicle_velocity);
			nf.getParam("velocity_zero",velocity_zero);
			nf.getParam("turn_velocity", turn_velocity);
			nf.getParam("steering_angle_to_servo_gain", steering_angle_to_servo_gain);
    		nf.getParam("steering_angle_to_servo_offset", steering_angle_to_servo_offset);
			nf.getParam("max_steering_vel", max_servo_speed);
			nf.getParam("max_accel", max_accel);
			nf.getParam("max_speed", max_speed);


			vel = 0.0;

			//MPC parameters
            nf.getParam("nMPC",nMPC);
            nf.getParam("kMPC",kMPC);
			nf.getParam("d_factor_STLMPC",d_factor);
            nf.getParam("d_dot_factor_STLMPC",d_dot_factor);
			nf.getParam("delta_factor_STLMPC",delta_factor);
			nf.getParam("angle_thresh", angle_thresh);
			nf.getParam("map_thresh", map_thresh);
			nf.getParam("use_map", use_map);

			//MPC init
			default_dt=0.077;
			deltas.resize(nMPC*kMPC,0);
			thetas.resize(nMPC*kMPC,0);
			x_vehicle.resize(nMPC*kMPC,0);
			for(int i=1; i<nMPC*kMPC; i++){
				x_vehicle[i] = x_vehicle[i-1]+vel_adapt*default_dt;
				
			}
			y_vehicle.resize(nMPC*kMPC,0);
			last_delta=0;
			velocity_MPC=vehicle_velocity;
			last_servo_state.data=steering_angle_to_servo_offset;
			
			nf.getParam("yolo_rows", yolo_rows);
			nf.getParam("yolo_cols", yolo_cols);
			nf.getParam("use_neural_net", use_neural_net);
			nf.getParam("veh_det_length", veh_det_length);
			nf.getParam("veh_det_width", veh_det_width);

			meas_observability = Eigen::Matrix<double, 2, 5>::Zero();
			meas_observability(0, 0) = 1;  // (1,1) in 1-based indexing
			meas_observability(1, 1) = 1;  // (2,2) in 1-based indexing

			//timing
			nf.getParam("stop_time1", stop_time1);
			nf.getParam("stop_time2", stop_time2);
			stopped_time = 0.0;

			//camera
			nf.getParam("use_camera", use_camera);


			//imu init
			yaw0 = 0.0; dtheta = 0.0;

			//mux init
			nf.getParam("nav_mux_idx", nav_mux_idx);
			nav_active = 0;

			//cv
			ros::param::get("~min_cv_range", min_cv_range);
            ros::param::get("~max_cv_range", max_cv_range);
            ros::param::get("~cv_distance_to_lidar", cv_distance_to_lidar);
            ros::param::get("~num_cv_sample_rows", num_cv_sample_rows);
            ros::param::get("~num_cv_sample_cols",num_cv_sample_cols);

            ros::param::get("~cv_ground_angle", cv_ground_angle);
            ros::param::get("~cv_lidar_range_max_diff",cv_lidar_range_max_diff);
            ros::param::get("~camera_height",camera_height);
			ros::param::get("~camera_min",camera_min);
			ros::param::get("~camera_max", camera_max);
            ros::param::get("~cv_real_to_theo_ground_range_ratio",cv_real_to_theo_ground_range_ratio);
            ros::param::get("~cv_real_to_theo_ground_range_ratio_near_horizon",cv_real_to_theo_ground_range_ratio_near_horizon);
            ros::param::get("~cv_ground_range_decay_row",cv_ground_range_decay_row);
            ros::param::get("~cv_pitch_angle_hardcoded",cv_pitch_angle_hardcoded);

			ros::param::get("~cv_groundplane_max_height", cv_groundplane_max_height);
			ros::param::get("~cv_groundplane_max_distance", cv_groundplane_max_distance); 



			intrinsics_d_defined= false; intrinsics_c_defined= false;
        	cv_image_data_defined= false;

			//subscriptions
			lidar = nf.subscribe("/scan",1, &GapBarrier::lidar_callback, this);
			imu = nf.subscribe(imu_topic,1, &GapBarrier::imu_callback, this);
			mux = nf.subscribe(mux_topic,1, &GapBarrier::mux_callback, this);
			vesc_state_sub= nf.subscribe("/sensors/core", 1, &GapBarrier::vesc_callback, this);
			servo_sub= nf.subscribe("/sensors/servo_position_command", 1,&GapBarrier::servo_callback, this);
			// odom = nf.subscribe(odom_topic,1, &GapBarrier::odom_callback, this);
			// localize = nf.subscribe("/pose_stamped",1, &GapBarrier::localize_callback, this);
			amcl_sub = nf.subscribe("/amcl_pose", 1, &GapBarrier::amcl_callback, this);
			tf_sub = nf.subscribe("/tf", 20, &GapBarrier::tf_callback, this);
			map_sub = nf.subscribe(map_topic, 1, &GapBarrier::map_callback, this);
			yolo_sub=nf.subscribe(yolo_data_topic, 1, &GapBarrier::yolo_callback, this);
			cmdvel_sub=nf.subscribe("/cmd_vel", 1, &GapBarrier::cmdVelCallback, this);
			 

			//publications
			//lidar_pub = nf.advertise<std_msgs::Int32MultiArray>("chatter", 1000);
			marker_pub = nf.advertise<visualization_msgs::Marker>("wall_markers",2);
			mpc_marker_pub = nf.advertise<visualization_msgs::Marker>("mpc_markers",2);
			wall_marker_pub=nf.advertise<visualization_msgs::Marker>("walls",2);
			lobs=nf.advertise<visualization_msgs::Marker>("lobs",2);
			robs=nf.advertise<visualization_msgs::Marker>("robs",2);
			vehicle_detect=nf.advertise<visualization_msgs::Marker>("vehicle_detect",2);
			driver_pub = nf.advertise<ackermann_msgs::AckermannDriveStamped>(drive_topic, 1);

			if(use_camera)
			{
				cv_ranges_msg= sensor_msgs::LaserScan(); //call constructor
				cv_ranges_msg.header.frame_id= "laser";
				cv_ranges_msg.angle_increment= this->ls_ang_inc; 
				cv_ranges_msg.time_increment = 0;
				cv_ranges_msg.range_min = 0;
				cv_ranges_msg.range_max = this->max_lidar_range;
				cv_ranges_msg.angle_min = 0;
				cv_ranges_msg.angle_max = 2*M_PI;

				cv_ranges_pub=nf.advertise<sensor_msgs::LaserScan>(cv_ranges_topic,1);
				
				depth_img=nf.subscribe(depth_image_topic,1, &GapBarrier::imageDepth_callback,this);
				depth_info=nf.subscribe(depth_info_topic,1, &GapBarrier::imageDepthInfo_callback,this);
				depth_img_confidence=nf.subscribe("/camera/confidence/image_rect_raw",1, &GapBarrier::confidenceCallback, this);
				color_img=nf.subscribe(color_image_topic,1, &GapBarrier::imageColor_callback,this);
				color_info=nf.subscribe(color_info_topic,1, &GapBarrier::imageColorInfo_callback,this);
				cam_extrinsics=nf.subscribe(cam_extr_topic,1, &GapBarrier::camExtrinsics_callback,this);
			}

		}



		/// ---------------------- GENERAL HELPER FUNCTIONS ----------------------

		// void publish_lidar(std::vector<int> data2){


		// 	std_msgs::Int32MultiArray lidar_msg;
		// 	lidar_msg.data.clear();

		// 	for(int i =0; i < int(data2.size()); ++i){
		// 		lidar_msg.data.push_back(int(data2[i]));
		// 	}

		// 	lidar_pub.publish(lidar_msg);
		// }

		int equiv_sign(double qt){
			if(qt < 0) return -1;
			else if (qt == 0 ) return 0;
			else return 1;
		}


		int arg_max(std::vector<float> ranges){

			int idx = 0;

			for(int i =1; i < int(ranges.size()); ++i){
				if(ranges[idx] < ranges[i]) idx = i;
			}

			return idx;


		}


		std::string getOdom() const { return odom_topic; }
		int getRightBeam() const { return right_beam_angle;}
		std::string getLidarTopic() const { return lidarscan_topic;}


		/// ---------------------- MAIN FUNCTIONS ----------------------

		void tf_callback(const tf2_msgs::TFMessage::ConstPtr& msg){ //Update the localization transforms
			int updated=0;
			 for (const geometry_msgs::TransformStamped& transform : msg->transforms)
			{
				if (transform.header.frame_id == "odom" && transform.child_frame_id == "base_link")
				{
					odomx=transform.transform.translation.x;
					odomy=transform.transform.translation.y;
					// 		transform.transform.translation.z);
					double x=transform.transform.rotation.x;
					double y=transform.transform.rotation.y;
					double z=transform.transform.rotation.z;
					double w=transform.transform.rotation.w;
					odomtheta = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
					updated=1;

					timestamp_tf1 = transform.header.stamp;

					tf_data new_tf;
					new_tf.tf_x=odomx; new_tf.tf_y=odomy; new_tf.tf_theta=odomtheta; new_tf.tf_time=timestamp_tf1.toSec();
					past_tf.push_back(new_tf);

					if(past_tf.size()>10) past_tf.erase(past_tf.begin());

				}
				else if (transform.header.frame_id == "map" && transform.child_frame_id == "odom")
				{
					mapx=transform.transform.translation.x;
					mapy=transform.transform.translation.y;
					// 		transform.transform.translation.z);
					double x=transform.transform.rotation.x;
					double y=transform.transform.rotation.y;
					double z=transform.transform.rotation.z;
					double w=transform.transform.rotation.w;
					maptheta = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
					updated=1;
				}
				else if (transform.header.frame_id == "map" && transform.child_frame_id == "det_racecar_base_link") //Simulation detection of other vehicle
				{
					//Just for the one vehicle detection case
					robx=transform.transform.translation.x;
					roby=transform.transform.translation.y;
					// 		transform.transform.translation.z);
					double x=transform.transform.rotation.x;
					double y=transform.transform.rotation.y;
					double z=transform.transform.rotation.z;
					double w=transform.transform.rotation.w;
					robtheta = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));

					double detx=(robx-simx)*cos(simtheta)+(roby-simy)*sin(simtheta);
					double dety=-(robx-simx)*sin(simtheta)+(roby-simy)*cos(simtheta);

					tf_data new_tf;
					new_tf.tf_x=simx; new_tf.tf_y=simy; new_tf.tf_theta=simtheta; new_tf.tf_time=ros::Time::now().toSec();
					if(car_detects.size()<1){
						vehicle_detection new_det;
						new_det.bound_box={0,0,20,20}; //ymin, xmin, ymax, xmax PLACEHOLDERS
						new_det.meas={detx,dety};
						new_det.last_det=1;
						new_det.meas_tf=new_tf;
						new_det.cov_P(0,0)=0.01; new_det.cov_P(1,1)=0.01; new_det.cov_P(2,2)=std::pow(5 * M_PI / 180, 2);
						new_det.cov_P(3,3)=2; new_det.cov_P(4,4)=std::pow(5 * M_PI / 180, 2);
						new_det.proc_noise=new_det.cov_P;
		
						car_detects.push_back(new_det);
					}
					else{
						car_detects[0].bound_box={0,0,20,20}; //ymin, xmin, ymax, xmax PLACEHOLDERS
						car_detects[0].meas={detx,dety};
						car_detects[0].last_det=1; //Detected in this round
						car_detects[0].meas_tf=new_tf;
					}


				}
				else if(transform.header.frame_id == "map" && transform.child_frame_id == "base_link"){ //This is for simulation only
					vel_adapt=velocity_MPC; //In simulation, adaptive velocity is based on our input, not VESC since not connected
					
					simx=transform.transform.translation.x;
					simy=transform.transform.translation.y;
					// 		transform.transform.translation.z);
					double x=transform.transform.rotation.x;
					double y=transform.transform.rotation.y;
					double z=transform.transform.rotation.z;
					double w=transform.transform.rotation.w;
					simtheta = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
				}
			}
			if(updated){
				locx=mapx+cos(maptheta)*odomx-sin(maptheta)*odomy;
				locy=mapy+sin(maptheta)*odomx+cos(maptheta)*odomy;
				loctheta=maptheta+odomtheta;
				while (loctheta > M_PI) loctheta -= 2 * M_PI;
    			while (loctheta < -M_PI) loctheta += 2 * M_PI;
				//THIS CONVERTS BASE_LINK POSITION INTO MAP FRAME
				//Can find the inverse from base_link to map only when required in MPC function so map frame can be transformed to base_link again
			}


		}

		void cmdVelCallback(const geometry_msgs::Twist::ConstPtr& msg)
		{

			if(double(msg->linear.x)<lastcmdvel){
				lastcmdvel=std::max(std::max(double(msg->linear.x),0.0),lastcmdvel-max_accel*0.1);
			}
			else{
				lastcmdvel=std::min(std::min(double(msg->linear.x),max_speed),lastcmdvel+max_accel*0.1);
			}

			if (std::abs(lastcmdvel) > 1e-5) {
				if(double(std::atan2(wheelbase * msg->angular.z, lastcmdvel))<lastcmdsteer){
					lastcmdsteer=std::max(std::max(double(std::atan2(wheelbase * msg->angular.z, lastcmdvel)),-max_steering_angle),lastcmdsteer-max_servo_speed*0.1);
				}
				else{
					lastcmdsteer=std::min(std::min(double(std::atan2(wheelbase * msg->angular.z, lastcmdvel)),max_steering_angle),lastcmdsteer+max_servo_speed*0.1);
				}

			} else {
				lastcmdsteer=0;
			}
			

			if(local_ranges.size()>10){
				double mindist=100;
				for(int i=0;i<local_ranges.size();i++){
					if(local_ranges[i]<mindist){
						mindist=local_ranges[i];
					}

				}
			}
		}

		void map_callback(const nav_msgs::OccupancyGrid & map_msg) {
			
			if(use_map && map_saved==0){ //Upon startup, save the map once and for full run
				for (int i=0;i<map_msg.info.width*map_msg.info.height;i++){
					if(map_msg.data[i]==100){
						int add_pt=1;
						for(int j=0;j<map_pts.size();j++){
							if(pow(map_pts[j][0]-map_msg.info.origin.position.x-i%map_msg.info.width*map_msg.info.resolution,2)+pow(map_pts[j][1]-map_msg.info.origin.position.y-i/map_msg.info.width*map_msg.info.resolution,2)<map_thresh){
								add_pt=0; //If the points are too close, don't add in order to reduce computation load
								break;
							}
						}
						if(add_pt){
							map_pts.push_back({map_msg.info.origin.position.x+i%map_msg.info.width*map_msg.info.resolution,map_msg.info.origin.position.y+i/map_msg.info.width*map_msg.info.resolution});
						}
					}
				}
				map_saved=1;
			}
			


		}

		void amcl_callback(const geometry_msgs::PoseWithCovarianceStamped & amcl_msg){
			// printf("cov:%lf, %lf, %lf\n",amcl_msg.pose.covariance[0],amcl_msg.pose.covariance[7],amcl_msg.pose.covariance[35]);
			// printf("pose:%lf, %lf, %lf\n",amcl_msg.pose.pose.position.x,amcl_msg.pose.pose.position.y,2*atan2(amcl_msg.pose.pose.orientation.z, amcl_msg.pose.pose.orientation.w));
		}

		void yolo_callback(const f1tenth_simulator::YoloData & yolo_msg){ //Process all other vehicle detections for the KF
			if(!cv_image_data_defined || !intrinsics_c_defined) return; //Check if initial data exists


			if(yolo_msg.classes.size()==0) return; //No detections

			if(depth_imgs.size()<1) return;
			int depth_ind=depth_imgs.size()-1; double mindiff=100;
			for (int i=0;i<depth_imgs.size();i++){ //Find matching depth image to the rgb, based on closest reported timestamp
				if(mindiff>std::abs(depth_imgs[i]->header.stamp.toSec()-yolo_msg.time)){
					mindiff=std::abs(depth_imgs[i]->header.stamp.toSec()-yolo_msg.time);
					depth_ind=i;
				}
			}

			tf_data my_tf;

			for (int mo=past_tf.size()-1; mo>0;mo--){ //Find the tf corresponding to the closest depth image, for finding right frame in KF
				if(past_tf[mo].tf_time>depth_imgs[depth_ind]->header.stamp.toSec() && past_tf[mo-1].tf_time<depth_imgs[depth_ind]->header.stamp.toSec()){
					if(std::abs(past_tf[mo].tf_time-depth_imgs[depth_ind]->header.stamp.toSec())<std::abs(past_tf[mo-1].tf_time-depth_imgs[depth_ind]->header.stamp.toSec())){
						my_tf=past_tf[mo];
					}
					else{
						my_tf=past_tf[mo-1];
					}
					break;
				}
			}		

			cv::Mat cv_image1=(cv_bridge::toCvCopy(depth_imgs[depth_ind],depth_imgs[depth_ind]->encoding))->image;

			std::vector<float> cv_point(3);
			//Multi-vehicle tracking, distinguishing between detections
			std::vector<int> temp_vec(car_detects.size(), -1); //yolo detections of already identified detections
			std::vector<int> no_det; //yolo detections of not pre-identified detections
			std::vector<std::vector<double>> yolo_xy; //x and y measurements of depth from yolo detection
			

			for (int i=0;i<yolo_msg.classes.size();i++){
				int num=-1; float dist=100000;
				const std::string& class_name = yolo_msg.classes[i];
				if(class_name=="car"){
					
					yolo_xy.push_back(depth_calc(cv_image1, yolo_msg, i));

					for (int j=0; j<car_detects.size();j++){ //Compare midpoints of bounding boxes, find minimum
						double x_d=pow((car_detects[j].bound_box[1]+car_detects[j].bound_box[3])/2-(yolo_msg.rectangles[4*i+1]+yolo_msg.rectangles[4*i+3])/2,2);
						double y_d=pow((car_detects[j].bound_box[0]+car_detects[j].bound_box[2])/2-(yolo_msg.rectangles[4*i]+yolo_msg.rectangles[4*i+2])/2,2);
						if(sqrt(x_d+y_d)<dist && sqrt(x_d+y_d)<141 && temp_vec[j]==-1){ //Find closest match, ensure we aren't matching two yolos to the same detection and threshold dist must be below
							dist=sqrt(x_d+y_d);
							num=j;
						}
					}
					if(num!=-1){
						temp_vec[num]=i; //ith detection is mapped
					}
					else no_det.push_back(i);

					double xorigin=cos(odomtheta)*(yolo_xy[0][0])-sin(odomtheta)*yolo_xy[0][1]+odomx;
					double yorigin=sin(odomtheta)*(yolo_xy[0][0])+cos(odomtheta)*yolo_xy[0][1]+odomy;

				}
			}

			//For all existing detections, provide the x and y of the depth measurement
			for (int q=0; q<car_detects.size();q++){
				int i=temp_vec[q];
				if(temp_vec[q]!=-1 &&yolo_xy[i][0]!=0 && yolo_xy[i][1]!=0){ //Detection found
					car_detects[q].bound_box={yolo_msg.rectangles[4*i],yolo_msg.rectangles[4*i+1],yolo_msg.rectangles[4*i+2],yolo_msg.rectangles[4*i+3]}; //ymin, xmin, ymax, xmax
					car_detects[q].meas={yolo_xy[i][0],yolo_xy[i][1]};
					car_detects[q].last_det=1; //Detected in this round
					car_detects[q].meas_tf=my_tf;
				}
			}

			//For all new detections, create structs appended to the vector (deletions should take place in lidar callback)
			for (int q=0; q<no_det.size();q++){
				int i=no_det[q];
				if(yolo_xy[i][0]!=0 && yolo_xy[i][1]!=0){
					vehicle_detection new_det;
					new_det.bound_box={yolo_msg.rectangles[4*i],yolo_msg.rectangles[4*i+1],yolo_msg.rectangles[4*i+2],yolo_msg.rectangles[4*i+3]}; //ymin, xmin, ymax, xmax
					new_det.meas={yolo_xy[i][0],yolo_xy[i][1]};
					new_det.last_det=1;
					new_det.meas_tf=my_tf;
					new_det.cov_P(0,0)=0.05; new_det.cov_P(1,1)=0.05; new_det.cov_P(2,2)=std::pow(45 * M_PI / 180, 2);
					new_det.cov_P(3,3)=2; new_det.cov_P(4,4)=std::pow(45 * M_PI / 180, 2);
					new_det.proc_noise=new_det.cov_P;
	
					car_detects.push_back(new_det);
				}	
			}
		}


		std::vector<double> depth_calc(cv::Mat cv_image1, const f1tenth_simulator::YoloData & yolo_msg, int det_ind){
			std::vector<float> cv_point(3);
			std::vector<float> cv_point1(3);
			std::vector<float> col_point(3);

			//If close to a border, select an average closer to the edge to ensure the car is detected, not background
			float x_start=0.45; float x_end=0.55; float y_start=0.45; float y_end=0.55;
			if(yolo_msg.rectangles[4*det_ind+1]<10 && yolo_msg.rectangles[4*det_ind+3]<630) x_start=0.25, x_end=0.35; //Left edge
			else if(yolo_msg.rectangles[4*det_ind+1]>10 && yolo_msg.rectangles[4*det_ind+3]>630) x_start=0.65, x_end=0.75; //Right edge

			if(yolo_msg.rectangles[4*det_ind]<10 && yolo_msg.rectangles[4*det_ind+2]<470) y_start=0.25, y_end=0.35; //Top edge
			else if(yolo_msg.rectangles[4*det_ind]>10 && yolo_msg.rectangles[4*det_ind+2]>470) y_start=0.65, y_end=0.75; //Bottom edge
			
			int x_true=yolo_msg.rectangles[4*det_ind+1]*(1-x_start)+yolo_msg.rectangles[4*det_ind+3]*x_start; int y_true=yolo_msg.rectangles[4*det_ind]*(1-y_start)+yolo_msg.rectangles[4*det_ind+2]*y_start;
			int x_cv=x_true/2; int y_cv=y_true/2;
			float depth_pixel[2] = {(float) x_cv, (float) y_cv};
			float color_pixel[2] = {(float) 0, (float) 0};
			while (color_pixel[0]<=x_true || color_pixel[1]<=y_true){
				if((cv_image1.ptr<uint16_t>((int)depth_pixel[1])[(int)depth_pixel[0]])/(float)1000>0.01){
					rs2_deproject_pixel_to_point(cv_point1.data(), &intrinsics_depth, depth_pixel, (cv_image1.ptr<uint16_t>((int)depth_pixel[1])[(int)depth_pixel[0]])/(float)1000);			
					rs2_transform_point_to_point(col_point.data(),&extrinsics,cv_point1.data());
					rs2_project_point_to_pixel(color_pixel,&intrinsics_color,col_point.data());
					if(color_pixel[0]<=x_true) depth_pixel[0]+=std::max(1,(int)((x_true-color_pixel[0])/2));
					if(color_pixel[1]<=y_true) depth_pixel[1]+=std::max(1,(int)((y_true-color_pixel[1])/2));
						
				}
				else{
					if(color_pixel[0]<=x_true) depth_pixel[0]++;
					if(color_pixel[1]<=y_true) depth_pixel[1]++;
				}
			}
			
			//Average depth calculation
			int count=0;
			double av_depth=0;
			double av_x=0; double av_y=0;
			float depth_pixel_start[2]; depth_pixel_start[0]=depth_pixel[0]; depth_pixel_start[1]=depth_pixel[1];

			while(color_pixel[0]<yolo_msg.rectangles[4*det_ind+1]*(1-x_end)+yolo_msg.rectangles[4*det_ind+3]*x_end){
				
				while(color_pixel[1]<yolo_msg.rectangles[4*det_ind]*(1-y_end)+yolo_msg.rectangles[4*det_ind+2]*y_end){
					if((cv_image1.ptr<uint16_t>((int)depth_pixel[1])[(int)depth_pixel[0]])/(float)1000>0.01){
						av_depth+=(cv_image1.ptr<uint16_t>((int)depth_pixel[1])[(int)depth_pixel[0]])/(float)1000;
						rs2_deproject_pixel_to_point(cv_point.data(), &intrinsics_depth, depth_pixel, (cv_image1.ptr<uint16_t>((int)depth_pixel[1])[(int)depth_pixel[0]])/(float)1000);
						av_x=av_x+cv_point[2]+cv_distance_to_lidar; av_y=av_y-cv_point[0];
						count++;
					}
					depth_pixel[1]++;
					rs2_deproject_pixel_to_point(cv_point1.data(), &intrinsics_depth, depth_pixel, (cv_image1.ptr<uint16_t>((int)depth_pixel[1])[(int)depth_pixel[0]])/(float)1000);			
					rs2_transform_point_to_point(col_point.data(),&extrinsics,cv_point1.data());
					rs2_project_point_to_pixel(color_pixel,&intrinsics_color,col_point.data());

				}
				depth_pixel[0]++; depth_pixel[1]=depth_pixel_start[1];
				rs2_deproject_pixel_to_point(cv_point1.data(), &intrinsics_depth, depth_pixel, (cv_image1.ptr<uint16_t>((int)depth_pixel[1])[(int)depth_pixel[0]])/(float)1000);			
				rs2_transform_point_to_point(col_point.data(),&extrinsics,cv_point1.data());
				rs2_project_point_to_pixel(color_pixel,&intrinsics_color,col_point.data());
			}

			av_depth=av_depth/count;
			av_x=av_x/count; av_y=av_y/count;
			std::vector<double> det_xy; det_xy.push_back(0); det_xy.push_back(0);
			if(count>0){
				det_xy[0]=av_x; det_xy[1]=av_y;
			}
			// else printf("No points found, can't update measurement\n");

			return det_xy;

		}


		void mux_callback(const std_msgs::Int32MultiArrayConstPtr& data){nav_active = data->data[nav_mux_idx]; }

		void servo_callback(const std_msgs::Float64 & servo){
			last_servo_state=servo;
		}

		void vesc_callback(const vesc_msgs::VescStateStamped & state){
        	vel_adapt = std::max(-( state.state.speed - speed_to_erpm_offset ) / speed_to_erpm_gain,0.1);
			last_delta = ( last_servo_state.data - steering_angle_to_servo_offset) / steering_angle_to_servo_gain;
		}

		void imu_callback(const sensor_msgs::Imu::ConstPtr& data){

				tf::Quaternion myQuaternion(
				data->orientation.x,
				data->orientation.y,
				data->orientation.z,
				data->orientation.w);
			
			tf::Matrix3x3 m(myQuaternion);
			m.getRPY(imu_roll, imu_pitch, imu_yaw);

		}


		
		void imageDepth_callback( const sensor_msgs::ImageConstPtr & img)
		{
			
			timestamp_cam1=img->header.stamp;
			
			if(intrinsics_d_defined)
			{
				if(depth_imgs.size()>2) depth_imgs.erase(depth_imgs.begin());
				depth_imgs.push_back(img);
				//Unsure how copy constructor behaves, therefore manually copyied all data members
				cv_image_data.header= img->header; 
				cv_image_data.height=img->height;
				cv_image_data.width=img->width;
				cv_image_data.encoding=img->encoding;
				cv_image_data.is_bigendian=img->is_bigendian;
				cv_image_data.step=img->step;
				cv_image_data.data=img->data;
				cv_image_data_defined=true;
			}
			else
			{
				cv_image_data_defined=false;
			}

		}

		void imageDepthInfo_callback(const sensor_msgs::CameraInfoConstPtr & cameraInfo)
		{
			//intrinsics is a struct of the form:
			/*
			int           width; 
			int           height
			float         ppx;   
			float         ppy;
			float         fx;
			float         fy;   
			rs2_distortion model;
			float coeffs[5];
			*/
			if(intrinsics_d_defined){ return; }

			//std::cout << "Defining Intrinsics" <<std::endl;

            intrinsics_depth.width = cameraInfo->width;
            intrinsics_depth.height = cameraInfo->height;
            intrinsics_depth.ppx = cameraInfo->K[2];
            intrinsics_depth.ppy = cameraInfo->K[5];
            intrinsics_depth.fx = cameraInfo->K[0];
            intrinsics_depth.fy = cameraInfo->K[4];
			
            if (cameraInfo->distortion_model == "plumb_bob") 
			{
				intrinsics_depth.model = RS2_DISTORTION_BROWN_CONRADY;   
			}
               
            else if (cameraInfo->distortion_model == "equidistant")
			{
				intrinsics_depth.model = RS2_DISTORTION_KANNALA_BRANDT4;
			}
            for(int i=0; i<5; i++)
			{
				intrinsics_depth.coeffs[i]=cameraInfo->D[i];
			}
			intrinsics_d_defined=true;

			cv_rows=intrinsics_depth.height;
			cv_cols=intrinsics_depth.width;

			//define pixels that will be sampled in each row and column, spaced evenly by linspace function

			cv_sample_rows_raw= xt::linspace<int>(0, cv_rows-1, num_cv_sample_rows);
			cv_sample_cols_raw= xt::linspace<int>(0, cv_cols-1, num_cv_sample_cols);


		}
		//Realsense D435 has no confidence data
		void confidenceCallback(const sensor_msgs::ImageConstPtr & data)
		{
			/*
			cv::Mat cv_image=(cv_bridge::toCvCopy(data,data->encoding))->image; 
			auto grades= cv::bitwise_and(cv_image >> 4, cv::Scalar(0x0f));
			*/



		}

		void imageColor_callback( const sensor_msgs::ImageConstPtr & img)
		{
			
		}


		void imageColorInfo_callback(const sensor_msgs::CameraInfoConstPtr & cameraInfo)
		{
			//intrinsics is a struct of the form:
			/*
			int           width; 
			int           height
			float         ppx;   
			float         ppy;
			float         fx;
			float         fy;   
			rs2_distortion model;
			float coeffs[5];
			*/
			if(intrinsics_c_defined){ return; }

			//std::cout << "Defining Intrinsics" <<std::endl;

            intrinsics_color.width = cameraInfo->width;
            intrinsics_color.height = cameraInfo->height;
            intrinsics_color.ppx = cameraInfo->K[2];
            intrinsics_color.ppy = cameraInfo->K[5];
            intrinsics_color.fx = cameraInfo->K[0];
            intrinsics_color.fy = cameraInfo->K[4];
			
            if (cameraInfo->distortion_model == "plumb_bob") 
			{
				intrinsics_color.model = RS2_DISTORTION_BROWN_CONRADY;   
			}
               
            else if (cameraInfo->distortion_model == "equidistant")
			{
				intrinsics_color.model = RS2_DISTORTION_KANNALA_BRANDT4;
			}
            for(int i=0; i<5; i++)
			{
				intrinsics_color.coeffs[i]=cameraInfo->D[i];
			}
			intrinsics_c_defined=true;
		}


		void camExtrinsics_callback(const realsense2_camera::Extrinsics::ConstPtr &msg)
		{
			cv::Mat extrinsics1 = cv::Mat::eye(4, 4, CV_64F);
			// Extract rotation matrix
			cv::Mat rotation = cv::Mat(3, 3, CV_64F, const_cast<double *>(msg->rotation.data()));
			rotation.convertTo(rotation, CV_64F);

			// Extract translation vector
			cv::Mat translation = cv::Mat(3, 1, CV_64F, const_cast<double *>(msg->translation.data()));
			translation.convertTo(translation, CV_64F);

			// Update global extrinsics matrix
			extrinsics1.at<double>(0, 0) = rotation.at<double>(0, 0); extrinsics1.at<double>(0, 1) = rotation.at<double>(0, 1); extrinsics1.at<double>(0, 2) = rotation.at<double>(0, 2);
			extrinsics1.at<double>(1, 0) = rotation.at<double>(1, 0); extrinsics1.at<double>(1, 1) = rotation.at<double>(1, 1); extrinsics1.at<double>(1, 2) = rotation.at<double>(1, 2);
			extrinsics1.at<double>(2, 0) = rotation.at<double>(2, 0); extrinsics1.at<double>(2, 1) = rotation.at<double>(2, 1); extrinsics1.at<double>(2, 2) = rotation.at<double>(2, 2);

			extrinsics1.at<double>(0, 3) = translation.at<double>(0, 0); extrinsics1.at<double>(1, 3) = translation.at<double>(1, 0); extrinsics1.at<double>(2, 3) = translation.at<double>(2, 0);

			//Matrix to rs2:extrinsics
			if (extrinsics1.rows == 4 && extrinsics1.cols == 4) {
				// Copy rotation (3x3)
				for (int i = 0; i < 3; ++i) {
					for (int j = 0; j < 3; ++j) {
						extrinsics.rotation[i * 3 + j] = extrinsics1.at<double>(i, j);
					}
				}

				// Copy translation (3)
				for (int i = 0; i < 3; ++i) {
					extrinsics.translation[i] = extrinsics1.at<double>(i, 3);
				}
			}

		}




		plane fit_groundplane(std::vector<float3> points)
		{

            float3 sum = { 0,0,0 };
            for (auto point : points)
			{
				sum.x+=point.x;
				sum.y+=point.y;
				sum.z+=point.z;
			}

            float3 centroid = {sum.x / float(points.size()), sum.y/ float(points.size()), sum.z/ float(points.size())};

            double xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
            for (auto point : points) 
			{
                float3 temp = {point.x - centroid.x, point.y - centroid.y, point.z - centroid.z};
                xx += temp.x * temp.x;
                xy += temp.x * temp.y;
                xz += temp.x * temp.z;
                yy += temp.y * temp.y;
                yz += temp.y * temp.z;
                zz += temp.z * temp.z;
            }

            //double det_x = yy*zz - yz*yz;
            double det_y = xx*zz - xz*xz;
            //double det_z = xx*yy - xy*xy;


			//cramers rule solutions
           //double det_max = std::max({ det_x, det_y, det_z });
            //if (det_max <= 0) return{ 0, 0, 0, 0 };


			float3 dir{};

            /*if (det_max == det_x)
            {
                float a = static_cast<float>((xz*yz - xy*zz) / det_x);
                float b = static_cast<float>((xy*yz - xz*yy) / det_x);
                dir = { 1, a, b };
            }*/
            //else if (det_max == det_y)
            //{
            float a = static_cast<float>((yz*xz - xy*zz) / det_y);
            float b = static_cast<float>((xy*xz - yz*xx) / det_y);
            dir = { a, 1, b };
            //}
            /*else
            {
                float a = static_cast<float>((yz*xy - xz*yy) / det_z);
                float b = static_cast<float>((xz*xy - yz*xx) / det_z);
                dir = { a, b, 1 };
            }*/


			//normalize dir
			
			float mag= std::pow( std::pow(dir.x,2)+std::pow(dir.y,2)+std::pow(dir.z,2), 0.5 );
			//(x^2+y^2+z^2)^0.5

			dir.x=dir.x/mag, dir.y=dir.y/mag, dir.z=dir.z/mag;


			//return plane
			plane result;
			result.A = dir.x, result.B=dir.y, result.C= dir.z;
			result.D= -(dir.x*centroid.x + dir.y*centroid.y + dir.z*centroid.z);


			

			//std::cout << "Ground Plane. " << "A= " << result.A << ". B= " <<result.B << ". C= " <<result.C << ". D= " <<result.D <<std::endl;
			return result;
		}

		plane compute_groundplane(cv::Mat cv_image)
		{
			std::vector<float3> plane_points; //store all points close to the ground
			for(int i=0 ; i < (int)cv_sample_cols_raw.size() ; i++)
			{
				int col= cv_sample_cols_raw[i];

				for(int j=0; j < (int)cv_sample_rows_raw.size() ; j++)
				{
					int row=cv_sample_rows_raw[j];

					float depth= (cv_image.ptr<uint16_t>(row)[col])/(float)1000;
				
					if(depth > max_cv_range || depth < min_cv_range)
					{
						continue;
					}
					
					std::vector<float> cv_point(3); 
					float pixel[2] = {(float) col, (float) row};
					rs2_deproject_pixel_to_point(cv_point.data(), &intrinsics_depth, pixel, depth);

					//xyz points in 3D space, process and combine with lidar data
					float cv_coordx=cv_point[0];
					float cv_coordy=cv_point[1];
					float cv_coordz=cv_point[2];
					//track points to be used for plane fitting 
					float3 temp;
					temp.x=cv_coordx,temp.y=cv_coordy,temp.z=cv_coordz;
					//+y=down, -y=up

					//prefilter points 
					if(temp.y<cv_groundplane_max_height) { continue; } //too high
					else
					{
						//std::cout << "Adding Point to be part of ground plane fitting" << std::endl;
						plane_points.push_back(temp);
					}
				}
			}

			plane ground_plane=fit_groundplane(plane_points);
			return ground_plane;


		}			

		float distance_from_plane(plane groundplane, float3 point)
		{
			// project vector pointing from plane to point onto the planes normal vector

			//proj a B= proj of B onto a = (A dot B /(|A|^2)+*a
			//magnitude = |a dot b|/|a|. including sign on dot product will include direction relative to normal

			//1. Determine a point on the plane

			float3 plane_point={ 0 , 0 , 1/groundplane.C*-groundplane.D };

			//2. Compute a vector pointing from on the plane to the point in space

			float3 vector= {point.x-plane_point.x, point.y-plane_point.y, point.z-plane_point.z};

			//3. Project onto the normal vector of the plane, tracking magnitude and sign.

			float dot=groundplane.A*vector.x+groundplane.B*vector.y+groundplane.C*vector.z;
			float magnitude=std::pow(groundplane.A*groundplane.A+groundplane.B*groundplane.B+
									 groundplane.C*groundplane.C,0.5);
			
			float result= dot/magnitude;

			//std::cout <<"Result= " <<result<<std::endl;
			return result;
		}


		void augment_camera(std::vector<float> & lidar_ranges)
		{
			cv::Mat cv_image=(cv_bridge::toCvCopy(cv_image_data,cv_image_data.encoding))->image; //Encoding type is 16UC1 (depth in mm)

			plane ground= compute_groundplane(cv_image);


			

			//use to debug: Returning 1
			//bool assert=( (cv_rows==cv_image.rows) && (cv_cols==cv_image.cols) );

			//std::cout << "Augment Camera Assert = " << assert <<std::endl; 


			//1. Obtain pixel and depth
			
			for(int i=0 ; i < (int)cv_sample_cols_raw.size() ; i++)
			{
				int col= cv_sample_cols_raw[i];
				

				for(int j=0; j < (int)cv_sample_rows_raw.size() ; j++)
				{
					int row=cv_sample_rows_raw[j];

					
					
					float depth= (cv_image.ptr<uint16_t>(row)[col])/(float)1000;

					
					if(depth > max_cv_range or depth < min_cv_range)
					{
						continue;
					}
					//2 convert pixel to xyz coordinate in space using camera intrinsics, pixel coords, and depth info
					std::vector<float> cv_point(3); 
					float pixel[2] = {(float) col, (float) row};
					rs2_deproject_pixel_to_point(cv_point.data(), &intrinsics_depth, pixel, depth);

					//xyz points in 3D space, process and combine with lidar data
					float cv_coordx=cv_point[0];
					float cv_coordy=cv_point[1];
					float cv_coordz=cv_point[2];

					float3 point={cv_coordx,cv_coordy,cv_coordz};

					//ground point check

					//1. Compute distance from ground plane
					float distance= distance_from_plane(ground,point);
					//2. ignore if ground

					//distance is postive if along the same direction as plane normal (down), negative if oppsote plane normal(up)
					
					if ( distance> -cv_groundplane_max_distance || distance < -camera_max) //max distance from plane to which a point is considered ground
					{
						//postive=below plane, 
						continue; //ignore ground point
					}




					//imu_pitch=0;
					//imu_roll=0;
					


					/*
					float cv_coordy_s = -1*cv_coordx*std::sin(imu_pitch) + cv_coordy*std::cos(imu_pitch)*std::cos(imu_roll) 
					+ cv_coordz *std::cos(imu_pitch)*std::sin(imu_roll);

					
					if( cv_coordy_s > camera_min || cv_coordy_s < -camera_max)
					{
						continue;
					}
					*/


					//3. Overwrite Lidar Points with Camera Points taking into account dif frames of ref

					float lidar_coordx = (cv_coordz+cv_distance_to_lidar);
                	float lidar_coordy = -cv_coordx;
					float cv_range_temp = std::pow(std::pow(lidar_coordx,2) + std::pow(lidar_coordy,2),0.5);
					//(coordx^2+coordy^2)^0.5

					int beam_index= std::floor(scan_beams*std::atan2(lidar_coordy, lidar_coordx)/(2*M_PI));
					float lidar_range = lidar_ranges[beam_index];
					lidar_ranges[beam_index] = std::min(lidar_range, cv_range_temp);
				}
			}

			ros::Time current_time= ros::Time::now();
			cv_ranges_msg.header.stamp=current_time;
			cv_ranges_msg.ranges=lidar_ranges;

			cv_ranges_pub.publish(cv_ranges_msg);
			

		}





		std::pair <std::vector<std::vector<float>>, std::vector<float>>preprocess_lidar(std::vector<float> ranges){

			std::vector<std::vector<float>> data(ls_len_mod,std::vector<float>(2));
			std::vector<float> data2(100);

			//sets distance to zero for obstacles in safe distance, and max_lidar_range for those that are far.
			for(int i =0; i < ls_len_mod; ++i){
				if(ranges[ls_str+i] <= safe_distance) {data[i][0] = 0; data[i][1] = i*ls_ang_inc-angle_cen;}
				else if(ranges[ls_str+i] <= max_lidar_range) {data[i][0] = ranges[ls_str+i]; data[i][1] = i*ls_ang_inc-angle_cen;}
				else {data[i][0] = max_lidar_range; data[i][1] = i*ls_ang_inc-angle_cen;}
			}

			int k1 = 100; int k2 = 40;
			float s_range = 0; int index1, index2;
			
			//moving window
			for(int i =0; i < k1; ++i){
				s_range = 0;

				for(int j =0; j < k2; ++j){
					index1 = int(i*ranges.size()/k1+j);
					if(index1 >= int(ranges.size())) index1 -= ranges.size();
					
					index2 = int(i*ranges.size()/k1-j);
					if(index2 < 0) index2 += ranges.size();

					s_range += std::min(ranges[index1], (float) max_lidar_range) + std::min(ranges[index2], (float)max_lidar_range);

				}
				data2[i] = s_range;
			}

			return std::make_pair(data,data2);
			
		}

		std::pair<int, int> find_max_gap(std::vector<std::vector<float>> proc_ranges){
			int j =0; int str_indx = 0; int end_indx = 0; 
			int str_indx2 = 0; int end_indx2 = 0;
			int range_sum = 0; int range_sum_new = 0;

			/*This just finds the start and end indices of gaps (non-safe distance lidar return)
			then does a comparison to find the largest such gap.*/
			for(int i =0; i < ls_len_mod; ++i){
				if(proc_ranges[i][0] != 0){
					if (j==0){
						str_indx = i;
						range_sum_new = 0;
						j = 1;
					}
					range_sum_new += proc_ranges[i][0];
					end_indx = i;
				}
				if(j==1 && (proc_ranges[i][0] == 0 || i == ls_len_mod - 1)){
					j = 0;

					if(range_sum_new > range_sum){
						end_indx2 = end_indx;
						str_indx2 = str_indx;
						range_sum = range_sum_new;
					}
				}
			}

			return std::make_pair(str_indx2, end_indx2);
		}


		float find_best_point(int start_i, int end_i, std::vector<std::vector<float>> proc_ranges){
			float range_sum = 0;
			float best_heading =0;


			for(int i = start_i; i <= end_i; ++i){
				range_sum += proc_ranges[i][0];
				best_heading += proc_ranges[i][0]*proc_ranges[i][1];

			}

			if(range_sum != 0){
				best_heading /= range_sum;
			}

			return best_heading; 
		}


		std::vector<float> preprocess_lidar_MPC(std::vector<float> ranges, std::vector<double> lidar_angles){
			left_ind_MPC = 0; right_ind_MPC = 0;
			//sets distance to zero for obstacles in safe distance, and max_lidar_range for those that are far.
			int num_det=0;
			double safe_dist=safe_distance;
			std::vector<float> ranges1=ranges;
			while(num_det==0){
				num_det=0;
				left_ind_MPC = 0; right_ind_MPC = 0;
				ranges=ranges1;
				if(safe_dist<0.1){
					num_det=1;
					for(int i =0; i < ranges.size(); ++i){
						if(lidar_angles[i] <= right_beam_angle_MPC) right_ind_MPC +=1;
						if(lidar_angles[i] <= left_beam_angle_MPC) left_ind_MPC +=1;
					}
					left_ind_MPC +=1;
				}
				else{
					for(int i =0; i < ranges.size(); ++i){
						if(lidar_angles[i] <= right_beam_angle_MPC) right_ind_MPC +=1;
						if(lidar_angles[i] <= left_beam_angle_MPC) left_ind_MPC +=1;
						if(right_ind_MPC!=i+1 && left_ind_MPC==i+1){
							if(ranges[i] <= safe_dist) {ranges[i] = 0;}
							else if(ranges[i] > max_lidar_range) {ranges[i] = max_lidar_range; num_det++;}
							else{num_det++;}
						}
						
					}
				}
				safe_dist=safe_dist/2;
			}
			left_ind_MPC -=1;
			return ranges;
			
		}


		double find_missing_scan_gap_MPC(std::vector<double> lidar_transform_angles){
			double best_heading=0;
			double max_gap=0;
			double start_gap=0;
			for(int i=right_ind_MPC; i<=left_ind_MPC; ++i){
				if(lidar_transform_angles[i]-lidar_transform_angles[i-1]>max_gap){
					max_gap=lidar_transform_angles[i]-lidar_transform_angles[i-1];
					start_gap=lidar_transform_angles[i-1];
				}
			}
			if(lidar_transform_angles[left_ind_MPC+1]-lidar_transform_angles[left_ind_MPC]>max_gap){
					max_gap=lidar_transform_angles[left_ind_MPC+1]-lidar_transform_angles[left_ind_MPC];
					start_gap=lidar_transform_angles[left_ind_MPC];
				}
			if(max_gap>angle_thresh){ //What this does is find if the largest gap is too big to use the accurate averaging method to get heading
				best_heading=start_gap+max_gap/2;
				missing_pts=1;
			}
			else{
				best_heading=5; //Means we can use the subsequent functions instead of this to find heading
			}
			return best_heading;
		}


		std::pair<int, int> find_max_gap_MPC(std::vector<float> proc_ranges, std::vector<double> lidar_transform_angles){
			int j =0; int str_indx = 0; int end_indx = 0; 
			int str_indx2 = 0; int end_indx2 = 0;
			double range_sum = 0; double range_sum_new = 0;
			

			/*This just finds the start and end indices of gaps (non-safe distance lidar return)
			then does a comparison to find the largest such gap.*/
			for(int i =right_ind_MPC; i <= left_ind_MPC; ++i){
				
				if(proc_ranges[i] != 0){
					if (j==0){
						str_indx = i;
						range_sum_new = 0;
						j = 1;
					}
					range_sum_new += proc_ranges[i]*(lidar_transform_angles[i+1]-lidar_transform_angles[i-1])/2;
					end_indx = i;
				}
				if(j==1 && (proc_ranges[i] == 0 || i == left_ind_MPC)){
					j = 0;

					if(range_sum_new > range_sum){
						end_indx2 = end_indx;
						str_indx2 = str_indx;
						range_sum = range_sum_new;
					}
				}
			}
			return std::make_pair(str_indx2, end_indx2);
		}


		double find_best_point_MPC(int start_i, int end_i, std::vector<float> proc_ranges, std::vector<double> lidar_transform_angles){
			double best_heading = 0; //Angles aren't evenly spaced so simple midpoint suffices here, avoids complications	
			double max_range = 0; 
	
			// for(int i=start_i; i<=end_i; ++i){ //This doesnt work since many points have max_distance since no range returned
			// 	if(proc_ranges[i] > max_range){
			// 		max_range = proc_ranges[i];
			// 		best_heading = lidar_transform_angles[i];
			// 	}
				
			// }
			
			best_heading=(lidar_transform_angles[start_i]+lidar_transform_angles[end_i])/2;

			return best_heading; 
		}


		void getWalls(std::vector<std::vector<double>> obstacle_points_l, std::vector<std::vector<double>> obstacle_points_r,
		std::vector<double> wl0, std::vector<double> wr0, double alpha, std::vector<double> &wr, std::vector<double> &wl,
		std::vector<double> &wc){
			if(!optim_mode){
				//right
				quadprogpp::Matrix<double> Gr,CEr,CIr;
				quadprogpp::Vector<double> gr0,ce0r,ci0r,xr;
				int n,m,p; char ch;
				int n_obs_r = obstacle_points_r.size(); int n_obs_l = obstacle_points_l.size();


				//left
				n = 2; m = 0; p = n_obs_l;
				quadprogpp::Matrix<double> Gl, CEl, CIl;
				quadprogpp::Vector<double> gl0, ce0l, ci0l, xl;

				//left matrices
				Gl.resize(n,n);
				{
					std::istringstream is("1.0, 0.0,"
														"0.0, 1.0 ");

					for(int i=0; i < n; ++i)
						for(int j=0; j < n; ++j)
							is >> Gl[i][j] >> ch;
				}
				gl0.resize(n);
				{
					for(int i =0; i < int(wl0.size()); ++i) gl0[i] = wl0[i] * (alpha-1);
				}
				CEl.resize(n,m);
				{
					CEl[0][0] = 0.0;
					CEl[1][0] = 0.0;
				}
				ce0l.resize(m);

				CIl.resize(n,p);
				{
					for(int i=0; i < p; ++i){
						CIl[0][i] = -obstacle_points_l[i][0];
						CIl[1][i] = -obstacle_points_l[i][1]; 
					}
				}
				ci0l.resize(p);
				{
					for(int i =0; i < p; ++i) ci0l[i] = -1.0;
				}

				xl.resize(n);
				// std::stringstream ss;
				solve_quadprog(Gl, gl0, CEl, ce0l, CIl, ci0l, xl);
				wl[0] = xl[0]; wl[1] = xl[1];


				p = n_obs_r;
				//right matrices
				Gr.resize(n,n);
				{
					std::istringstream is("1.0, 0.0,"
													"0.0, 1.0 ");

					for(int i =0; i < n; ++i)
						for(int j=0; j < n; ++j)
							is >> Gr[i][j] >> ch;

				}

				gr0.resize(n);
				{
					for(int i = 0; i < int(wr0.size()); ++i) gr0[i] = wr0[i] * (alpha-1);
				}

				CEr.resize(n,m);
				{
					CEr[0][0] = 0.0;
					CEr[1][0] = 0.0;
				}
				ce0r.resize(m);

				
				CIr.resize(n,p);
				{
						for(int i =0; i < p; ++i){
							CIr[0][i] = -obstacle_points_r[i][0];
							CIr[1][i] = -obstacle_points_r[i][1];
						}
				}

				ci0r.resize(p);
				{
					for(int i =0; i < p; ++i) ci0r[i] = -1.0;
				}

				xr.resize(n);
				solve_quadprog(Gr, gr0, CEr, ce0r, CIr, ci0r, xr);
				// ss << xr[0] << " " << xr[1];
				wr[0] = xr[0]; wr[1] = xr[1]; 


			}
			else{
				quadprogpp::Matrix<double> G,CE,CI;
				quadprogpp::Vector<double> gi0, ce0, ci0, x;

				// char ch;
				int n_obs_l = obstacle_points_l.size(); int n_obs_r = obstacle_points_r.size();
				
				
				int n,m,p;
				n = 3; m = 0; p = n_obs_l + n_obs_r + 2;

				G.resize(n,n);
				{
					// std::istringstream is("1.0, 0.0, 0.0,"
					// 						"0.0, 1.0, 0.0,"
					// 						"0.0, 0.0, 0.0001");

					// for(int i =0; i < n; ++i)
					// 	for(int j =0; j < n-1; ++j)
					// 		is >> G[i][j] >> ch;


					G[0][1] = G[0][2] = G[1][0] = G[1][2] = G[2][0] = G[2][1] = 0.0;
					G[0][0] = G[1][1] = 1.0;
					G[2][2] = 0.0001;

				}
				gi0.resize(n);
				{
					for(int i =0; i < n; ++i) gi0[i] = 0.0;
				}

				CE.resize(n,m);
				{
					CE[0][0] = 0.0;
					CE[1][0] = 0.0;
					CE[2][0] = 0.0;
				}
				ce0.resize(m);

				CI.resize(n,p);
				{
					for(int i =0; i < n_obs_r; ++i){
						CI[0][i] = obstacle_points_r[i][0];
						CI[1][i] = obstacle_points_r[i][1];
						CI[2][i] = 1.0;
					}

					for(int i = n_obs_r; i < n_obs_l + n_obs_r; ++i){
						CI[0][i] = -obstacle_points_l[i-n_obs_r][0];
						CI[1][i] = -obstacle_points_l[i-n_obs_r][1];
						CI[2][i] = -1.0;
					}

					CI[0][n_obs_l+n_obs_r] = 0.0; CI[1][n_obs_l+n_obs_r] = 0.0; CI[2][n_obs_l+n_obs_r] = 1.0;
					CI[0][n_obs_l+n_obs_r+1] = 0.0; CI[1][n_obs_l+n_obs_r+1] = 0.0; CI[2][n_obs_l+n_obs_r+1] = -1.0;

				}
				ci0.resize(p);
				{
					for(int i =0; i < n_obs_r+n_obs_l; ++i){
						ci0[i] = -1.0;
					}
					
					ci0[n_obs_r+n_obs_l] = 0.9; ci0[n_obs_r+n_obs_l+1] = 0.9;
				}
				x.resize(n);

				solve_quadprog(G, gi0, CE, ce0, CI, ci0, x);



				wr[0] = (x[0]/(x[2]-1)); wr[1] = (x[1]/(x[2]-1));

				wl[0] = (x[0]/(x[2]+1)); wl[1] = (x[1]/(x[2]+1));

				wc[0] = (x[0]/(x[2])); wc[1] = (x[1]/(x[2]));


			}

		}
		

		void visualize_detections(){
			//This should occur whether or not we are in autonomous mode

			//Publish the detected vehicle(s) trajectory(s)
			vehicle_detect_path.header.frame_id = "base_link";
			vehicle_detect_path.header.stamp = ros::Time::now();
			vehicle_detect_path.type = visualization_msgs::Marker::LINE_LIST;
			vehicle_detect_path.id = 0; 
			vehicle_detect_path.action = visualization_msgs::Marker::ADD;
			vehicle_detect_path.scale.x = 0.1;
			vehicle_detect_path.color.a = 1.0;
			vehicle_detect_path.color.r = 0.6; 
			vehicle_detect_path.color.g = 0.2;
			vehicle_detect_path.color.b = 0.8;
			vehicle_detect_path.pose.orientation.w = 1;
			
			vehicle_detect_path.lifetime = ros::Duration(0.1);
			geometry_msgs::Point p7;
			vehicle_detect_path.points.clear();

			for(int i=0; i<car_detects.size();i++){ //For each detection, plot trajectory over next 3 seconds	
				if(car_detects[i].init<2) continue;
				double x_det=car_detects[i].state[0]; double y_det=car_detects[i].state[1]; double theta_det=car_detects[i].state[2];
				
				for (int traj=0;traj<40;traj++){
					p7.x = x_det;	p7.y = y_det;	p7.z = 0;
					vehicle_detect_path.points.push_back(p7);
					x_det=x_det+car_detects[i].state[3]*cos(theta_det)/10; //0.1 second increments (coarse but just for visualization)
					y_det=y_det+car_detects[i].state[3]*sin(theta_det)/10; //0.1 second increments
					theta_det=theta_det+car_detects[i].state[3]/wheelbase*tan(car_detects[i].state[4])/10; //0.1 second increments
					p7.x = x_det;	p7.y = y_det;	p7.z = 0;
					vehicle_detect_path.points.push_back(p7);
				}
			}

			vehicle_detect.publish(vehicle_detect_path);
		}



		void lidar_callback(const sensor_msgs::LaserScanConstPtr &data){

			ls_ang_inc = static_cast<double>(data->angle_increment); 
			scan_beams = int(2*M_PI/data->angle_increment);
			ls_str = int(round(scan_beams*right_beam_angle/(2*M_PI)));
			ls_end = int(round(scan_beams*left_beam_angle/(2*M_PI)));
			std::vector<std::vector<double>> save_map1;

			//TODO: ADD TIME STUFF HERE
			ros::Time t = ros::Time::now();
			current_time = t.toSec();
			double dt = current_time - prev_time;
			if(dt>1){
				dt=default_dt;
			}
			prev_time = current_time;
			
			local_ranges = data->ranges;

			//Vehicle tracking even if we aren't currently driving
			
			for (int q=car_detects.size()-1; q>=0;q--){ //Iterate backwards to handle deletions
				if(car_detects[q].last_det==0){ //No detection over last cycle
					car_detects[q].miss_fr++;
					if(car_detects[q].init==1) car_detects[q].init=0; //Two point initialization requires consecutive detections
					if(car_detects[q].miss_fr>5) car_detects.erase(car_detects.begin()+q); //Not detected over past several frames, delete detection struct
				}
				else car_detects[q].miss_fr=0; //Found, reset consecutive missed frames to 0
				
			}

			for(int q=0; q<car_detects.size();q++){ //Run the KF for every current detections here
				
				if(car_detects[q].init==0){ //Initialize
					if(car_detects[q].last_det==1){ //Only if measurement received
						car_detects[q].state[0]=car_detects[q].meas[0];
						car_detects[q].state[1]=car_detects[q].meas[1];
						car_detects[q].init=1;
					}
					car_detects[q].last_det=0;
					continue;
				}
				if(car_detects[q].init==1){ //Finish initializing
					//First transform last x and y to this new frame (also the measurements)
					double tempx=0; double tempy=0; double curmeasx=0; double curmeasy=0;

					if(simx==0){
						tempx=cos(odomtheta)*(lastx+car_detects[q].state[0]*cos(lasttheta)-car_detects[q].state[1]*sin(lasttheta)-odomx)+
							sin(odomtheta)*(lasty+car_detects[q].state[0]*sin(lasttheta)+car_detects[q].state[1]*cos(lasttheta)-odomy);
						tempy=-sin(odomtheta)*(lastx+car_detects[q].state[0]*cos(lasttheta)-car_detects[q].state[1]*sin(lasttheta)-odomx)+
							cos(odomtheta)*(lasty+car_detects[q].state[0]*sin(lasttheta)+car_detects[q].state[1]*cos(lasttheta)-odomy);

						car_detects[q].state[0]=tempx; car_detects[q].state[1]=tempy;

						curmeasx=cos(odomtheta)*(car_detects[q].meas_tf.tf_x+car_detects[q].meas[0]*cos(car_detects[q].meas_tf.tf_theta)-car_detects[q].meas[1]*sin(car_detects[q].meas_tf.tf_theta)-odomx)+
							sin(odomtheta)*(car_detects[q].meas_tf.tf_y+car_detects[q].meas[0]*sin(car_detects[q].meas_tf.tf_theta)+car_detects[q].meas[1]*cos(car_detects[q].meas_tf.tf_theta)-odomy);
						curmeasy=-sin(odomtheta)*(car_detects[q].meas_tf.tf_x+car_detects[q].meas[0]*cos(car_detects[q].meas_tf.tf_theta)-car_detects[q].meas[1]*sin(car_detects[q].meas_tf.tf_theta)-odomx)+
							cos(odomtheta)*(car_detects[q].meas_tf.tf_y+car_detects[q].meas[0]*sin(car_detects[q].meas_tf.tf_theta)+car_detects[q].meas[1]*cos(car_detects[q].meas_tf.tf_theta)-odomy);
					}
					else{
						tempx=cos(simtheta)*(lastx+car_detects[q].state[0]*cos(lasttheta)-car_detects[q].state[1]*sin(lasttheta)-simx)+
							sin(simtheta)*(lasty+car_detects[q].state[0]*sin(lasttheta)+car_detects[q].state[1]*cos(lasttheta)-simy);
						tempy=-sin(simtheta)*(lastx+car_detects[q].state[0]*cos(lasttheta)-car_detects[q].state[1]*sin(lasttheta)-simx)+
							cos(simtheta)*(lasty+car_detects[q].state[0]*sin(lasttheta)+car_detects[q].state[1]*cos(lasttheta)-simy);

						car_detects[q].state[0]=tempx; car_detects[q].state[1]=tempy;
						curmeasx=car_detects[q].meas[0];
						curmeasy=car_detects[q].meas[1];
					}

					car_detects[q].state[4]=0; //steering angle, depends on process noise
					car_detects[q].state[3]=std::min(sqrt(pow(curmeasx-car_detects[q].state[0],2)+pow(curmeasy-car_detects[q].state[1],2))/dt,0.3); //Cap initial at 3 m/s so we don't get extreme values upon initialization
					car_detects[q].state[2]=atan2(curmeasy-car_detects[q].state[1],curmeasx-car_detects[q].state[0]);
					car_detects[q].state[1]=curmeasy;
					car_detects[q].state[0]=curmeasx;

					
					car_detects[q].init=2; //Done initializing
					car_detects[q].last_det=0;
					continue;
				}

				//Now for the KF update process in the 'already initialized scenario'

				//First transform last x, y & theta to this new frame (also the measurements)
				double tempx=0; double tempy=0; double curmeasx=0; double curmeasy=0;
				
				if(simx==0){
					tempx=cos(odomtheta)*(lastx+car_detects[q].state[0]*cos(lasttheta)-car_detects[q].state[1]*sin(lasttheta)-odomx)+
						sin(odomtheta)*(lasty+car_detects[q].state[0]*sin(lasttheta)+car_detects[q].state[1]*cos(lasttheta)-odomy);
					tempy=-sin(odomtheta)*(lastx+car_detects[q].state[0]*cos(lasttheta)-car_detects[q].state[1]*sin(lasttheta)-odomx)+
						cos(odomtheta)*(lasty+car_detects[q].state[0]*sin(lasttheta)+car_detects[q].state[1]*cos(lasttheta)-odomy);

					car_detects[q].state[0]=tempx; car_detects[q].state[1]=tempy; car_detects[q].state[2]=car_detects[q].state[2]-(odomtheta-lasttheta);
					while(car_detects[q].state[2]>M_PI) car_detects[q].state[2]-=2*M_PI;
					while(car_detects[q].state[2]<-M_PI) car_detects[q].state[2]+=2*M_PI;

					curmeasx=cos(odomtheta)*(car_detects[q].meas_tf.tf_x+car_detects[q].meas[0]*cos(car_detects[q].meas_tf.tf_theta)-car_detects[q].meas[1]*sin(car_detects[q].meas_tf.tf_theta)-odomx)+
						sin(odomtheta)*(car_detects[q].meas_tf.tf_y+car_detects[q].meas[0]*sin(car_detects[q].meas_tf.tf_theta)+car_detects[q].meas[1]*cos(car_detects[q].meas_tf.tf_theta)-odomy);
					curmeasy=-sin(odomtheta)*(car_detects[q].meas_tf.tf_x+car_detects[q].meas[0]*cos(car_detects[q].meas_tf.tf_theta)-car_detects[q].meas[1]*sin(car_detects[q].meas_tf.tf_theta)-odomx)+
						cos(odomtheta)*(car_detects[q].meas_tf.tf_y+car_detects[q].meas[0]*sin(car_detects[q].meas_tf.tf_theta)+car_detects[q].meas[1]*cos(car_detects[q].meas_tf.tf_theta)-odomy);

					car_detects[q].meas[0]=curmeasx; car_detects[q].meas[1]=curmeasy;
				}
				else{
					tempx=cos(simtheta)*(lastx+car_detects[q].state[0]*cos(lasttheta)-car_detects[q].state[1]*sin(lasttheta)-simx)+
						sin(simtheta)*(lasty+car_detects[q].state[0]*sin(lasttheta)+car_detects[q].state[1]*cos(lasttheta)-simy);
					tempy=-sin(simtheta)*(lastx+car_detects[q].state[0]*cos(lasttheta)-car_detects[q].state[1]*sin(lasttheta)-simx)+
						cos(simtheta)*(lasty+car_detects[q].state[0]*sin(lasttheta)+car_detects[q].state[1]*cos(lasttheta)-simy);

					car_detects[q].state[0]=tempx; car_detects[q].state[1]=tempy; car_detects[q].state[2]=car_detects[q].state[2]-(simtheta-lasttheta);
					while(car_detects[q].state[2]>M_PI) car_detects[q].state[2]-=2*M_PI;
					while(car_detects[q].state[2]<-M_PI) car_detects[q].state[2]+=2*M_PI;
					curmeasx=car_detects[q].meas[0];
					curmeasy=car_detects[q].meas[1];
				}
				
				//1) State prediction
				Eigen::VectorXd pred_state = Eigen::VectorXd::Zero(5); //x, y, theta, vs, delta
				pred_state(0)=car_detects[q].state[0]+std::max(default_dt,dt)*car_detects[q].state[3]*cos(car_detects[q].state[2]);
				pred_state(1)=car_detects[q].state[1]+std::max(default_dt,dt)*car_detects[q].state[3]*sin(car_detects[q].state[2]);
				pred_state(2)=car_detects[q].state[2]+std::max(default_dt,dt)*car_detects[q].state[3]/wheelbase*tan(car_detects[q].state[4]);
				pred_state(3)=car_detects[q].state[3];
				pred_state(4)=car_detects[q].state[4];
				
				//2) Covariance prediction
				Eigen::Matrix<double, 5, 5> state_transition_lin = Eigen::Matrix<double, 5, 5>::Zero(); //Linearized state transition matrix
				state_transition_lin(0,0)=1; state_transition_lin(1,1)=1; state_transition_lin(2,2)=1; state_transition_lin(3,3)=1; state_transition_lin(4,4)=1;
				state_transition_lin(0,2)=-std::max(default_dt,dt)*car_detects[q].state[3]*sin(car_detects[q].state[2]);
				state_transition_lin(0,3)=std::max(default_dt,dt)*cos(car_detects[q].state[2]);
				state_transition_lin(1,2)=std::max(default_dt,dt)*car_detects[q].state[3]*cos(car_detects[q].state[2]);
				state_transition_lin(1,3)=std::max(default_dt,dt)*sin(car_detects[q].state[2]);
				state_transition_lin(2,3)=std::max(default_dt,dt)/wheelbase*tan(car_detects[q].state[4]);
				state_transition_lin(2,4)=std::max(default_dt,dt)*car_detects[q].state[3]/(wheelbase*cos(car_detects[q].state[4])*cos(car_detects[q].state[4]));

				if(car_detects[q].last_det==0){ //No detection in this cycle, don't use any measure, just predicted state and covariance
					car_detects[q].state=pred_state;
					printf("Missed Measure (%d)\n",q);
					
					car_detects[q].proc_noise(0,0)=0.05; car_detects[q].proc_noise(1,1)=0.05; car_detects[q].proc_noise(2,2)=std::pow(7.5 * M_PI / 180, 2);
					car_detects[q].proc_noise(3,3)=0.125; car_detects[q].proc_noise(4,4)=std::pow(7.5 * M_PI / 180, 2);
					car_detects[q].cov_P=state_transition_lin*car_detects[q].cov_P*state_transition_lin.transpose()+car_detects[q].proc_noise;
					continue;
				}

				//Process noise should depend on # of missed frames, speed of both our vehicle and the detected vehicle
				// if(car_detects[q].miss_fr==0){
				car_detects[q].proc_noise(0,0)=0.05; car_detects[q].proc_noise(1,1)=0.05; car_detects[q].proc_noise(2,2)=std::pow(7.5 * M_PI / 180, 2);
				car_detects[q].proc_noise(3,3)=0.125; car_detects[q].proc_noise(4,4)=std::pow(7.5 * M_PI / 180, 2);
				// }
				//Measurement noise should depend on the distance between the vehicles (maybe error of 2% of distance, increases when speeds increase)
				car_detects[q].meas_noise(0,0)=0.02*sqrt(std::pow(car_detects[q].meas[0],2)+std::pow(car_detects[q].meas[1],2));
				car_detects[q].meas_noise(1,1)=car_detects[q].meas_noise(0,0);
				
				//Also incorporate speed's effect on error
				car_detects[q].meas_noise(0,0)*=std::max(vel_adapt*car_detects[q].state[3]/0.5,1.0); car_detects[q].meas_noise(1,1)*=std::max(vel_adapt*car_detects[q].state[3]/0.5,1.0);
				car_detects[q].proc_noise(0,0)*=std::max(vel_adapt*car_detects[q].state[3]/0.5,1.0);car_detects[q].proc_noise(1,1)*=std::max(vel_adapt*car_detects[q].state[3]/0.5,1.0);
				car_detects[q].proc_noise(2,2)*=std::max(vel_adapt*car_detects[q].state[3]/0.5,1.0);car_detects[q].proc_noise(3,3)*=std::max(vel_adapt*car_detects[q].state[3]/0.5,1.0);
				car_detects[q].proc_noise(4,4)*=std::max(vel_adapt*car_detects[q].state[3]/0.5,1.0);



				car_detects[q].cov_P=state_transition_lin*car_detects[q].cov_P*state_transition_lin.transpose()+car_detects[q].proc_noise;

				//3) Innovation
				Eigen::VectorXd innovation_vec = Eigen::VectorXd::Zero(2); //Innovation matrix
				Eigen::VectorXd meas_vec = Eigen::VectorXd::Zero(2); //Measurement matrix
				meas_vec(0)=car_detects[q].meas[0]; meas_vec(1)=car_detects[q].meas[1];

				innovation_vec = meas_vec-meas_observability*pred_state;

				//4) Innovation Covariance
				Eigen::Matrix<double, 2, 2> innov_cov = Eigen::Matrix<double, 2, 2>::Zero(); //Innovation covariance matrix
				innov_cov=meas_observability*car_detects[q].cov_P*meas_observability.transpose()+car_detects[q].meas_noise;

				//5) Kalman Gain
				Eigen::Matrix<double, 5, 2> KF_gain = Eigen::Matrix<double, 5, 2>::Zero(); //Kalman Gain matrix
				KF_gain=car_detects[q].cov_P*meas_observability.transpose()*innov_cov.inverse();

				//6) Updated state estimate
				car_detects[q].state= pred_state+KF_gain*innovation_vec;
				while(car_detects[q].state[2]>M_PI) car_detects[q].state[2]-=2*M_PI;
				while(car_detects[q].state[2]<-M_PI) car_detects[q].state[2]+=2*M_PI;

				//7) Updated covariance
				Eigen::Matrix<double, 5, 5> identityMatrix = Eigen::Matrix<double, 5, 5>::Identity();
				car_detects[q].cov_P=(identityMatrix-KF_gain*meas_observability)*car_detects[q].cov_P;

				car_detects[q].last_det=0; //Reset the detection for next iteration, done at end to know in KF whether detected or not this cycle
			}

			lastx=odomx; lasty=odomy; lasttheta=odomtheta; //Keep our vehicle frame from last cycle to transform frame to new this cycle
			if(simx!=0){lastx=simx; lasty=simy; lasttheta=simtheta;}
			timestamp_tf2=timestamp_tf1; timestamp_cam2=timestamp_cam1;
			visualize_detections(); //PLot the detections in rviz regardless of if we are in autonomous mode or not

			if (!nav_active ||(use_map && !map_saved)) { //Don't start navigation until map is saved if that's what we're using
				drive_state = "normal";
				return;
			}


			std::vector<float> fused_ranges = data->ranges;
			// if(use_camera)
			// {
			// 	if(cv_image_data_defined){ augment_camera(fused_ranges); }
			// }
			

			int sec_len = int(heading_beam_angle/data->angle_increment);

			double min_distance, velocity_scale, delta_d;
			

			if(drive_state == "normal"){
				std::vector<float> fused_ranges_MPC=fused_ranges;
				std::vector<float> fused_ranges_MPC_tot0;
				std::vector<double> lidar_transform_angles_tot0;
				std::vector<float> fused_ranges_MPC_tot0s;
				std::vector<double> lidar_transform_angles_tot0s;
				std::vector<float> fused_ranges_MPC_tot0m;
				std::vector<double> lidar_transform_angles_tot0m;
				std::vector<float> fused_ranges_MPC_veh_det0;
				std::vector<double> lidar_transform_angles_veh_det0;

				std::vector<float> fused_ranges_MPC_map;
				std::vector<double> lidar_transform_angles_map; //These are the additional ranges & angles from fixed map that will be sorted, included in obs calculations
				
				callbackcount+=1;

				double track_line[2][nMPC*kMPC]; //Tracking line a & b (assume c=1) parameters for all time intervals, additional terms for passing n & k
				double theta_refs[nMPC]; //Reference angles for each time interval
				double startx, starty; //Initial points for that tracking line interval (line is likely not connected to prev. so jump)
				double xpt=0, ypt=0; //Reference point for future LIDAR calculations with same original LIDAR scan
				double theta_ref=0; //New reference theta for the LIDAR angles to be appropriately rotated so this angle is reference 0 

				std::vector<std::vector<double>> obstacle_points_l;
				std::vector<std::vector<double>> obstacle_points_r;
				double heading_angle;

				std::vector<double> lidar_transform_angles;
				for(int i=0;i<data->ranges.size();i++){
					lidar_transform_angles.push_back(i*data->angle_increment-M_PI);
				}

				double startxplot[nMPC],startyplot[nMPC],xptplot[nMPC],yptplot[nMPC];
				std::vector<double> wl = {0.0, 0.0};
				std::vector<double> wr = {0.0, 0.0};
				std::vector<double> wc = {0.0, 0.0};
				double mapped_x=locx, mapped_y=locy, mapped_theta=loctheta;

				std::vector<std::vector<double>> obstacle_points_r_MPC;

				std::vector<std::vector<double>> obstacle_points_l_MPC;

				for (int num_MPC=0;num_MPC<nMPC;num_MPC++){
					std::vector<float> fused_ranges_MPC_tot=fused_ranges_MPC;
					std::vector<double> lidar_transform_angles_tot=lidar_transform_angles; //The cumulative ranges and angles for both map (if used) and lidar

					if(use_map){ //Augment LIDAR with map obstacle points too
					
						double map_xval=mapped_x+cos(mapped_theta)*xpt-sin(mapped_theta)*ypt;
						double map_yval=mapped_y+sin(mapped_theta)*xpt+cos(mapped_theta)*ypt;
						for(int i=0;i<map_pts.size();i++){
							if(pow(map_pts[i][0]-map_xval,2)+pow(map_pts[i][1]-map_yval,2)<pow(max_lidar_range_opt,2)){
								double x_base=(map_pts[i][0]-locx)*cos(loctheta)+(map_pts[i][1]-locy)*sin(loctheta);
								double y_base=-(map_pts[i][0]-locx)*sin(loctheta)+(map_pts[i][1]-locy)*cos(loctheta);
								double x_fut=(x_base-xpt)*cos(theta_ref)+(y_base-ypt)*sin(theta_ref);
								double y_fut=-(x_base-xpt)*sin(theta_ref)+(y_base-ypt)*cos(theta_ref);
								double ang_base=atan2(y_fut,x_fut);
								if (ang_base>M_PI) ang_base-=2*M_PI;
								if (ang_base<-M_PI) ang_base+=2*M_PI;
								lidar_transform_angles_map.push_back(ang_base);
								fused_ranges_MPC_map.push_back(pow(pow(x_fut,2)+pow(y_fut,2),0.5));
							}
						}
						fused_ranges_MPC_tot.insert(fused_ranges_MPC_tot.end(), fused_ranges_MPC_map.begin(), fused_ranges_MPC_map.end());
						lidar_transform_angles_tot.insert(lidar_transform_angles_tot.end(), lidar_transform_angles_map.begin(), lidar_transform_angles_map.end());
						
						std::vector<std::pair<double, double>> vec;
						for (int i = 0; i < fused_ranges_MPC_tot.size(); ++i) {
							vec.push_back(std::make_pair(lidar_transform_angles_tot[i], fused_ranges_MPC_tot[i]));
						}

						// Step 2: Sort the vector of pairs based on the first element (myvec)
						std::sort(vec.begin(), vec.end(), [](const std::pair<double, double> &a, const std::pair<double, double> &b) {
							return a.first < b.first; // Compare the first elements of the pairs (myvec values)
						});

						// Step 3: Unpack the sorted pairs back into the original arrays
						for (int i = 0; i < fused_ranges_MPC_tot.size(); ++i) {
							lidar_transform_angles_tot[i] = vec[i].first;
							fused_ranges_MPC_tot[i] = vec[i].second;
							
						}
						
					}

					if(use_neural_net){ //Augment LIDAR with detected vehicle projected paths as well
						std::vector<float> fused_ranges_MPC_veh_det;
						std::vector<double> lidar_transform_angles_veh_det; //These are the additional ranges & angles from vehicle detections that will be sorted, included in obs calculations
						int mult_factor=1; //This way, we get 2x amount of points for detections, improves the augmentation of LIDAR data

						for (int i=0; i<car_detects.size(); i++){
							if(car_detects[i].init==2){
								int start_track=num_MPC*kMPC*mult_factor; int end_track=(num_MPC*kMPC+kMPC-1)*mult_factor;
								double track_x=car_detects[i].state[0]; double track_y=car_detects[i].state[1]; double track_theta=car_detects[i].state[2];
								//Make a box for the vehicle based on orientation and have this projected path
								//This ensures that the vehicle is more prominent in LIDAR detections and is more of a box as opposed to the mid-point
								
								for(int j=0; j<nMPC*kMPC*mult_factor; j++){
									if(j>=start_track && j<=end_track){ //Only take this line segment timeframe of the MPC this round
										int num_border=5; //Number of points along each border of the box

										double tfed_x=cos(theta_ref)*(track_x-xpt)+sin(theta_ref)*(track_y-ypt);
										double tfed_y=-sin(theta_ref)*(track_x-xpt)+cos(theta_ref)*(track_y-ypt); //tf from vehicle frame to future MPC frame
										double tfed_ang=0;

										//More complete detection if we construct a box around the current midpoint (note don't need to push back mid point then)
										double rel_theta=track_theta-theta_ref;
										std::vector<std::vector<double>> detection_corners;
										std::vector<double> top_right = {tfed_x+veh_det_length/2*cos(rel_theta)+veh_det_width/2*sin(rel_theta), tfed_y+veh_det_length/2*sin(rel_theta)-veh_det_width/2*cos(rel_theta)}; detection_corners.push_back(top_right);
										std::vector<double> top_left = {tfed_x+veh_det_length/2*cos(rel_theta)-veh_det_width/2*sin(rel_theta), tfed_y+veh_det_length/2*sin(rel_theta)+veh_det_width/2*cos(rel_theta)}; detection_corners.push_back(top_left);
										std::vector<double> bot_left = {tfed_x-veh_det_length/2*cos(rel_theta)-veh_det_width/2*sin(rel_theta), tfed_y-veh_det_length/2*sin(rel_theta)+veh_det_width/2*cos(rel_theta)}; detection_corners.push_back(bot_left);
										std::vector<double> bot_right = {tfed_x-veh_det_length/2*cos(rel_theta)+veh_det_width/2*sin(rel_theta), tfed_y-veh_det_length/2*sin(rel_theta)-veh_det_width/2*cos(rel_theta)}; detection_corners.push_back(bot_right);
										for (int q=0; q<4; q++){ //Iterate over current to next (modulus) corner
											for (int c1=0;c1<num_border;c1++){ //Add the interpolated # of points here for that border section
												double edge_x=detection_corners[q][0]+(detection_corners[(q+1)%4][0]-detection_corners[q][0])*c1/num_border;
												double edge_y=detection_corners[q][1]+(detection_corners[(q+1)%4][1]-detection_corners[q][1])*c1/num_border;
												double tfed_ang=atan2(edge_y,edge_x);
												if (tfed_ang>M_PI) tfed_ang-=2*M_PI; if (tfed_ang<-M_PI) tfed_ang+=2*M_PI;
												lidar_transform_angles_veh_det.push_back(tfed_ang);
												fused_ranges_MPC_veh_det.push_back(pow(pow(edge_x,2)+pow(edge_y,2),0.5));

												double edge_x1=edge_x*cos(theta_ref)-sin(theta_ref)*edge_y+xpt;
												double edge_y1=edge_x*sin(theta_ref)+cos(theta_ref)*edge_y+ypt;
												double tfed_ang1=atan2(edge_y1,edge_x1);
												if (tfed_ang1>M_PI) tfed_ang1-=2*M_PI; if (tfed_ang1<-M_PI) tfed_ang1+=2*M_PI;
												lidar_transform_angles_veh_det0.push_back(tfed_ang1);
												fused_ranges_MPC_veh_det0.push_back(pow(pow(edge_x1,2)+pow(edge_y1,2),0.5));

												
											}

										}

									}
									//Now, update the vehicle position for this next timestep
									track_x=track_x+car_detects[i].state[3]*cos(track_theta)*dt/mult_factor; //For the next timeframe (LIDAR callback), find position
									track_y=track_y+car_detects[i].state[3]*sin(track_theta)*dt/mult_factor;
									track_theta=track_theta+car_detects[i].state[3]/wheelbase*tan(car_detects[i].state[4])*dt/mult_factor;

								}

							}
						}

						fused_ranges_MPC_tot.insert(fused_ranges_MPC_tot.end(), fused_ranges_MPC_veh_det.begin(), fused_ranges_MPC_veh_det.end());
						lidar_transform_angles_tot.insert(lidar_transform_angles_tot.end(), lidar_transform_angles_veh_det.begin(), lidar_transform_angles_veh_det.end());
						
						std::vector<std::pair<double, double>> vec;
						for (int i = 0; i < fused_ranges_MPC_tot.size(); ++i) {
							vec.push_back(std::make_pair(lidar_transform_angles_tot[i], fused_ranges_MPC_tot[i]));
						}

						// Step 2: Sort the vector of pairs based on the first element (myvec)
						std::sort(vec.begin(), vec.end(), [](const std::pair<double, double> &a, const std::pair<double, double> &b) {
							return a.first < b.first; // Compare the first elements of the pairs (myvec values)
						});

						// Step 3: Unpack the sorted pairs back into the original arrays
						for (int i = 0; i < fused_ranges_MPC_tot.size(); ++i) {
							lidar_transform_angles_tot[i] = vec[i].first;
							fused_ranges_MPC_tot[i] = vec[i].second;
							
						}

					}

					if(num_MPC==0 && odomx!=0){
						for(int i=0; i<fused_ranges_MPC_tot.size();i++){
							if(std::isfinite(fused_ranges_MPC_tot[i]) && std::isfinite(lidar_transform_angles_tot[i])){
								double obx=fused_ranges_MPC_tot[i]*cos(lidar_transform_angles_tot[i]);
								double oby=fused_ranges_MPC_tot[i]*sin(lidar_transform_angles_tot[i]);
								double obsx=obx*cos(odomtheta)-oby*sin(odomtheta)+odomx;
								double obsy=obx*sin(odomtheta)+oby*cos(odomtheta)+odomy;
								int shortd=0;
								for(int j=0;j<save_map.size();j++){
									if(pow(obsx-save_map[j][0],2)+pow(obsy-save_map[j][1],2)<0.01){
										shortd=1;
									}
								}
								if(shortd==0){
									save_map.push_back({obsx,obsy});
									save_map1.push_back({obsx,obsy});
								}
							}
							
						}
					}

					if(num_MPC==0){
						double smallestdist=1000;
						for(int i=0;i<fused_ranges_MPC.size();i++){
							if(fused_ranges_MPC[i]<smallestdist){
								smallestdist=fused_ranges_MPC[i];
							}
						}
					}

					if(num_MPC==0){ //Take the original reference points for subsampling obstacles in NLOPT
						fused_ranges_MPC_tot0=fused_ranges_MPC_tot;
						lidar_transform_angles_tot0=lidar_transform_angles_tot;
						fused_ranges_MPC_tot0s=fused_ranges_MPC;
						lidar_transform_angles_tot0s=lidar_transform_angles;
						fused_ranges_MPC_tot0m=fused_ranges_MPC_map;
						lidar_transform_angles_tot0m=lidar_transform_angles_map;
					}


					std::vector<float> proc_ranges_MPC = preprocess_lidar_MPC(fused_ranges_MPC_tot,lidar_transform_angles_tot);
					
					int str_indx_MPC, end_indx_MPC; double heading_angle_MPC;
					heading_angle_MPC=find_missing_scan_gap_MPC(lidar_transform_angles_tot);
					heading_angle=heading_angle_MPC;
					
					if(heading_angle_MPC==5){ //Use the other method to find the heading angle (if gap is large enough, use this prior value)
						
						std::pair<int,int> max_gap_MPC = find_max_gap_MPC(proc_ranges_MPC,lidar_transform_angles_tot);
						str_indx_MPC = max_gap_MPC.first; end_indx_MPC = max_gap_MPC.second;

						heading_angle_MPC= find_best_point_MPC(str_indx_MPC, end_indx_MPC, proc_ranges_MPC,lidar_transform_angles_tot);
						heading_angle=heading_angle_MPC;
					
						float mod_angle_al_MPC = angle_al-M_PI + heading_angle_MPC;

						if(mod_angle_al_MPC > M_PI) mod_angle_al_MPC -= 2*M_PI;
						else if (mod_angle_al_MPC < -M_PI) mod_angle_al_MPC += 2*M_PI;

						float mod_angle_br_MPC = angle_br-M_PI + heading_angle_MPC;

						if(mod_angle_br_MPC > M_PI) mod_angle_br_MPC -= 2*M_PI;
						else if (mod_angle_br_MPC < -M_PI) mod_angle_br_MPC += 2*M_PI;

						float mod_angle_ar_MPC = angle_ar-M_PI + heading_angle_MPC;

						if(mod_angle_ar_MPC > M_PI) mod_angle_ar_MPC -= 2*M_PI;
						else if (mod_angle_ar_MPC < -M_PI) mod_angle_ar_MPC += 2*M_PI;

						float mod_angle_bl_MPC = angle_bl-M_PI + heading_angle_MPC;

						if(mod_angle_bl_MPC > M_PI) mod_angle_bl_MPC -= 2*M_PI;
						else if (mod_angle_bl_MPC < -M_PI) mod_angle_bl_MPC += 2*M_PI;

						int start_indx_l_MPC=0, start_indx_r_MPC=0, end_indx_l_MPC=0, end_indx_r_MPC=0;

						for (int w=0;w<fused_ranges_MPC_tot.size();w++){
						
							if(lidar_transform_angles_tot[w] <= mod_angle_br_MPC) start_indx_r_MPC +=1;
							if(lidar_transform_angles_tot[w] <= mod_angle_ar_MPC) end_indx_r_MPC +=1;
							if (lidar_transform_angles_tot[w] <= mod_angle_bl_MPC) end_indx_l_MPC +=1;
							if (lidar_transform_angles_tot[w] <= mod_angle_al_MPC) start_indx_l_MPC +=1;
						}
						end_indx_r_MPC-=1; //We overcounted by one past the end range for both left and right
						end_indx_l_MPC-=1;

						obstacle_points_l_MPC.clear();
						obstacle_points_l_MPC.push_back({0, max_lidar_range_opt});
						obstacle_points_l_MPC.push_back({1, max_lidar_range_opt}); 
						
						
						obstacle_points_r_MPC.clear();
						obstacle_points_r_MPC.push_back({0, -max_lidar_range_opt});
						obstacle_points_r_MPC.push_back({1, -max_lidar_range_opt});
						
						int num_left_pts = end_indx_l_MPC-start_indx_l_MPC+1; int num_right_pts = end_indx_r_MPC-start_indx_r_MPC+1;
						if (num_left_pts<=0) num_left_pts+=scan_beams;
						if (num_right_pts<=0) num_right_pts+=scan_beams;
						double left_step=num_left_pts, right_step=num_right_pts;
						if(num_left_pts>n_pts_l) num_left_pts = n_pts_l;
						if(num_right_pts>n_pts_r) num_right_pts = n_pts_r;
						left_step=left_step/n_pts_l, right_step=right_step/n_pts_r;
						if(left_step<1) left_step=1;
						if(right_step<1) right_step=1;


						int k_obs = 0; int obs_index;

						double x_obs, y_obs;

						for(int k = 0; k < num_left_pts; ++k){
							obs_index = (start_indx_l_MPC + (int)(k*left_step)) % fused_ranges_MPC_tot.size();

							double obs_range = static_cast<double>(fused_ranges_MPC_tot[obs_index]);
							

							if(obs_range <= max_lidar_range_opt){


								if(k_obs == 0){
									obstacle_points_l_MPC[0] = {obs_range*cos(lidar_transform_angles_tot[obs_index]), obs_range*sin(lidar_transform_angles_tot[obs_index]) };
								}
								else if (k_obs == 1){
									obstacle_points_l_MPC[1] = {obs_range*cos(lidar_transform_angles_tot[obs_index]), obs_range*sin(lidar_transform_angles_tot[obs_index]) };
								}
								else{
									x_obs = obs_range*cos(lidar_transform_angles_tot[obs_index]);
									y_obs = obs_range*sin(lidar_transform_angles_tot[obs_index]);
									
									std::vector<double> obstacles = {x_obs, y_obs};
									obstacle_points_l_MPC.push_back(obstacles);
								}
								k_obs+=1;
							}

						}
						k_obs = 0;
						
						
						for(int k = 0; k < num_right_pts; ++k){
							obs_index = (start_indx_r_MPC + (int)(k*right_step)) % fused_ranges_MPC_tot.size();
							double obs_range = static_cast<double>(fused_ranges_MPC_tot[obs_index]);

							if(obs_range <= max_lidar_range_opt) {
								if(k_obs == 0){
									obstacle_points_r_MPC[0] = {obs_range*cos(lidar_transform_angles_tot[obs_index]), obs_range*sin(lidar_transform_angles_tot[obs_index])};
								}
								else if(k_obs == 1){
									obstacle_points_r_MPC[1] = {obs_range*cos(lidar_transform_angles_tot[obs_index]),obs_range*sin(lidar_transform_angles_tot[obs_index])};
								}
								else{
									x_obs = obs_range*cos(lidar_transform_angles_tot[obs_index]);
									y_obs = obs_range*sin(lidar_transform_angles_tot[obs_index]);
									

									std::vector<double> obstacles = {x_obs, y_obs};
									obstacle_points_r_MPC.push_back(obstacles);
									
								}
								
								k_obs += 1;
							}
						}
						obstacle_points_l=obstacle_points_l_MPC, obstacle_points_r=obstacle_points_r_MPC;
					}

					
					

					double alpha = 1;
					if(num_MPC==0) alpha=1-exp(-dt/tau);
	

					std::vector<double> wl1 = {0.0, 0.0};
					std::vector<double> wr1 = {0.0, 0.0};

					if (missing_pts==0){
						getWalls(obstacle_points_l, obstacle_points_r, wl0, wr0, alpha, wr1, wl1, wc);

					}
					else{
						//New formulation for wc, if there is a large gap, just use the heading angle for tracking line
						wc[0]=-tan(heading_angle)*100;
						wc[1]=100;
						missing_pts=0;
					}

					if(num_MPC==0){ //We are only using center lines, not left and right. These get plotted in rviz
						wl[0] = wl1[0]; wl[1] = wl1[1];
						wr[0] = wr1[0]; wr[1] = wr1[1];
						wl0[0] = wl[0]; wl0[1] = wl[1];
						wr0[0] = wr[0], wr0[1] = wr[1];
					}

					//Rotate and translate the tracking line back to the original frame of reference
					std::vector<double> wc_new = wc;
					
					wc[0]=wc_new[0]*cos(theta_ref)-wc_new[1]*sin(theta_ref);
					wc[1]=wc_new[0]*sin(theta_ref)+wc_new[1]*cos(theta_ref);
					heading_angle=heading_angle+theta_ref;

					//Translation to (xpt, ypt) from (0,0)
					double temp_wc0=wc[0];
					double temp_wc1=wc[1];

					wc[0]=temp_wc0/(1-temp_wc0*xpt-temp_wc1*ypt);
					wc[1]=temp_wc1/(1-temp_wc0*xpt-temp_wc1*ypt);
					if(abs(wc[0])<0.00001){ //This prevents nan error from dividing by 0
						if(wc[0]>0) wc[0]=0.00001;
						else wc[0]=-0.00001;
					}
					if(abs(wc[1])<0.00001){
						if(wc[1]>0) wc[1]=0.00001;
						else wc[1]=-0.00001;
					}		

					
					//Find the startng point closest to the new tracking line
					double anorm=-1/wc[0],bnorm=1/wc[1],cnorm=-anorm*xpt-bnorm*ypt;
					startx=(bnorm/wc[1]-cnorm)/(anorm-wc[0]*bnorm/wc[1]);
					starty=(-1-wc[0]*startx)/wc[1];

					for (int i=0;i<kMPC;i++) //Set the tracking line parameters for this time interval
					{
						track_line[0][i+num_MPC*kMPC]=wc[0];
						track_line[1][i+num_MPC*kMPC]=wc[1];
						
					}
					

					theta_ref=-atan2(wc[0],wc[1]); //Process thetas so the difference between thetas is minimized(pick right next theta)
					

					while (theta_ref-heading_angle>M_PI) theta_ref-=2*M_PI;
					while (theta_ref-heading_angle<-M_PI) theta_ref+=2*M_PI;
					if(theta_ref-heading_angle>M_PI/2) theta_ref-=M_PI;
					if(theta_ref-heading_angle<-M_PI/2) theta_ref+=M_PI;
					while (theta_ref>M_PI) theta_ref-=2*M_PI;
					while (theta_ref<-M_PI) theta_ref+=2*M_PI;
					theta_refs[num_MPC]=theta_ref;
					xpt=startx+vel_adapt*std::max(default_dt,dt)*kMPC*cos(theta_ref); //Drive message relates to lidar callback scan topic, ~10Hz
					ypt=starty+vel_adapt*std::max(default_dt,dt)*kMPC*sin(theta_ref); //Use 13 Hz as absolute optimal but likely slower use dt
					
					xptplot[num_MPC]=xpt;
					yptplot[num_MPC]=ypt;


					startxplot[num_MPC]=startx;
					startyplot[num_MPC]=starty;
				
					if(num_MPC<nMPC-1){//Prepare LIDAR data for next iteration, no longer constantly spaced lidar angles	
						
						std::vector<float> lidar_transform = data->ranges;
						double transform_coords[2][lidar_transform.size()];
						for (int i=0;i<lidar_transform.size();i++)
						{
							transform_coords[0][i]=lidar_transform[i]*cos(-M_PI+i*2*M_PI/scan_beams);
							transform_coords[1][i]=lidar_transform[i]*sin(-M_PI+i*2*M_PI/scan_beams);
							lidar_transform[i]=sqrt(pow(xpt-transform_coords[0][i],2)+pow(ypt-transform_coords[1][i],2));
							lidar_transform_angles[i]=atan2(transform_coords[1][i]-ypt,transform_coords[0][i]-xpt)-theta_ref;
							if (lidar_transform_angles[i]>M_PI) lidar_transform_angles[i]-=2*M_PI;
							if (lidar_transform_angles[i]<-M_PI) lidar_transform_angles[i]+=2*M_PI;
							
						}
						std::vector<std::pair<double, double>> vec;
						for (int i = 0; i < lidar_transform.size(); ++i) {
							vec.push_back(std::make_pair(lidar_transform_angles[i], lidar_transform[i]));
						}

						// Step 2: Sort the vector of pairs based on the first element (myvec)
						std::sort(vec.begin(), vec.end(), [](const std::pair<double, double> &a, const std::pair<double, double> &b) {
							return a.first < b.first; // Compare the first elements of the pairs (myvec values)
						});

						// Step 3: Unpack the sorted pairs back into the original arrays
						for (int i = 0; i < lidar_transform.size(); ++i) {
							lidar_transform_angles[i] = vec[i].first;
							lidar_transform[i] = vec[i].second;
							
						}

						fused_ranges_MPC=lidar_transform; //New LIDAR data for next run through
						
						
					}

				}
				
				//PERFORM THE MPC NON-LINEAR OPTIMIZATION
				nlopt_opt opt;
				opt = nlopt_create(NLOPT_LD_SLSQP, nMPC*kMPC*4); /* algorithm and dimensionality */


				for (int i=nMPC*kMPC;i<2*nMPC*kMPC;i++){
					nlopt_set_lower_bound(opt, i, -max_steering_angle); //Bounds on max and min steering angle (delta)
					nlopt_set_upper_bound(opt, i, max_steering_angle);
				}
				

				nlopt_set_lower_bound(opt, 2*nMPC*kMPC, 0); //Set initial x, y and theta to be =0 (constraint effectively)
				nlopt_set_upper_bound(opt, 2*nMPC*kMPC, 0);
				nlopt_set_lower_bound(opt, 3*nMPC*kMPC, 0);
				nlopt_set_upper_bound(opt, 3*nMPC*kMPC, 0);
				nlopt_set_lower_bound(opt, 0, 0);
				nlopt_set_upper_bound(opt, 0, 0);

			

				nlopt_set_min_objective(opt, myfunc, &track_line);
				double tol[nMPC*kMPC-1]={1e-8};
				double tol1[2*nMPC*kMPC+1]={1e-8};
				
				
				double opt_params[4]={vel_adapt*std::max(default_dt,dt),wheelbase,std::abs(max_servo_speed*std::max(default_dt,dt)),last_delta};
				
				nlopt_add_equality_mconstraint(opt, nMPC*kMPC-1, theta_equality_con, &opt_params, tol);
				nlopt_add_equality_mconstraint(opt, nMPC*kMPC-1, x_equality_con, &opt_params, tol);
				nlopt_add_equality_mconstraint(opt, nMPC*kMPC-1, y_equality_con, &opt_params, tol);
				nlopt_add_inequality_mconstraint(opt, 2*nMPC*kMPC, delta_inequality_con, &opt_params, tol1);

			
				nlopt_set_xtol_rel(opt, 0.001); //Termination parameters
				nlopt_set_maxtime(opt, 0.05);

				double x[4*nMPC*kMPC];  /* `*`some` `initial` `guess`*` */

				//Try new attempt at initial guess
				for (int j=0;j<nMPC;j++){
					for (int i=0;i<kMPC;i++){
						if(i==0&&j==0){
							deltas[i+j*kMPC]=last_delta;
						}
						else if(i==1&&j==0){
							double thetanext=theta_refs[j];
							while(thetanext>M_PI) thetanext-=2*M_PI;
							while(thetanext<-M_PI) thetanext+=2*M_PI;
							double ex_delta=atan((thetanext)*opt_params[1]/opt_params[0]);
							if(thetanext>0){
								deltas[i+j*kMPC]=std::min(ex_delta,std::min(last_delta+opt_params[2],max_steering_angle-1e-6));
							}
							else if(thetanext<0){
								deltas[i+j*kMPC]=std::max(ex_delta,std::max(last_delta-opt_params[2],-max_steering_angle+1e-6));
							}
							else{
								deltas[i+j*kMPC]=last_delta;
							}
						}
						else{
							double thetanext=theta_refs[j];
							while(thetanext>thetas[i+j*kMPC]+M_PI) thetanext-=2*M_PI;
							while(thetanext<thetas[i+j*kMPC]-M_PI) thetanext+=2*M_PI;
							double ex_delta=atan((thetanext-thetas[i+j*kMPC])*opt_params[1]/opt_params[0]);
							if(thetanext>thetas[i+j*kMPC]){
								deltas[i+j*kMPC]=std::min(ex_delta,std::min(deltas[i+j*kMPC-1]+opt_params[2],max_steering_angle-1e-6));
							}
							else if(thetanext<thetas[i+j*kMPC]){
								deltas[i+j*kMPC]=std::max(ex_delta,std::max(deltas[i+j*kMPC-1]-opt_params[2],-max_steering_angle+1e-6));
							}
							else{
								deltas[i+j*kMPC]=deltas[i+j*kMPC-1];
							}
						}
						if(i!=kMPC-1||j!=nMPC-1){
							thetas[i+j*kMPC+1]=thetas[i+j*kMPC]+opt_params[0]/opt_params[1]*tan(deltas[i+j*kMPC]);
						}
					}
				}
				for(int i=1;i<nMPC*kMPC;i++){
					x_vehicle[i]=x_vehicle[i-1]+opt_params[0]*cos(thetas[i-1]);
					y_vehicle[i]=y_vehicle[i-1]+opt_params[0]*sin(thetas[i-1]);
				}

				for (int i=0;i<nMPC*kMPC;i++){ //STarting guess 
					x[i]=thetas[i];
					x[i+nMPC*kMPC]=deltas[i];
					x[i+2*nMPC*kMPC]=x_vehicle[i];
					x[i+3*nMPC*kMPC]=y_vehicle[i];
				}
				int successful_opt=0;

				double minf; /* `*`the` `minimum` `objective` `value,` `upon` `return`*` */
				double opttime1=ros::Time::now().toSec();
				nlopt_result optim= nlopt_optimize(opt, x, &minf); //This runs the optimization
				double opttime2=ros::Time::now().toSec();
				printf("OptTime: %lf, Evals: %d\n",opttime2-opttime1,nlopt_get_numevals(opt));


				if(isnan(minf)){
					forcestop=1;
					printf("Nan error\n");
				}
				else{
					forcestop=0;
				}

				if (optim < 0) {
					ROS_ERROR("Optimization Error %d",optim);
					printf("NLOPT Error: %s\n", nlopt_get_errmsg(opt));
				}
				else {
					successful_opt=1;
					printf("Successful Opt: %d\n",optim);
					for (int i=0;i<nMPC*kMPC;i++){
						thetas[i]=x[i];
						deltas[i]=x[i+nMPC*kMPC];
						x_vehicle[i]=x[i+2*nMPC*kMPC];
						y_vehicle[i]=x[i+3*nMPC*kMPC];
					}
					
					last_delta=deltas[1];
					
				}
				if(minf<5 && startcheck==0){
					startcheck=1;
				}

				nlopt_destroy(opt);



				double dl, dr; 
				double wr_dot, wl_dot; 
				wr_dot = wl_dot = 0;

				for(int i =0; i < 2; ++i){
					wl_dot += wl[i]*wl[i];
					wr_dot += wr[i]*wr[i];
				}

				dl = 1/sqrt(wl_dot); dr = 1/sqrt(wr_dot);

				std::vector<double> wr_h = {wr[0]*dr,wr[1]*dr}; std::vector<double> wl_h = {wl[0]*dl, wl[1]*dl};

				//Publish the optimal path via NLOPT
				marker.header.frame_id = "base_link";
				marker.header.stamp = ros::Time::now();
				marker.type = visualization_msgs::Marker::LINE_LIST;
				marker.id = 0; 
				marker.action = visualization_msgs::Marker::ADD;
				marker.scale.x = 0.1;
				marker.color.a = 1.0;
				marker.color.r = 0.5; 
				marker.color.g = 0.5;
				marker.color.b = 0.0;
				marker.pose.orientation.w = 1;
				
				marker.lifetime = ros::Duration(0.1);

				int line_len = 1;
				geometry_msgs::Point p;
				marker.points.clear();
				
				if(successful_opt==1){
					for (int i=0;i<nMPC*kMPC-1;i++){
						p.x = x_vehicle[i];	p.y = y_vehicle[i];	p.z = 0;
						marker.points.push_back(p);
						p.x = x_vehicle[i+1];	p.y = y_vehicle[i+1];	p.z = 0;
						marker.points.push_back(p);
					}
				}

				marker_pub.publish(marker);

				//Publish the MPC tracking lines
				mpc_marker.header.frame_id = "base_link";
				mpc_marker.header.stamp = ros::Time::now();
				mpc_marker.type = visualization_msgs::Marker::LINE_LIST;
				mpc_marker.id = 0; 
				mpc_marker.action = visualization_msgs::Marker::ADD;
				mpc_marker.scale.x = 0.1;
				mpc_marker.color.a = 1.0;
				mpc_marker.color.r = 0; 
				mpc_marker.color.g = 0.5;
				mpc_marker.color.b = 0.5;
				mpc_marker.pose.orientation.w = 1;
				
				mpc_marker.lifetime = ros::Duration(0.1);
				geometry_msgs::Point p1;
				mpc_marker.points.clear();

				for (int i=0;i<nMPC;i++){
					p1.x = startxplot[i];	p1.y = startyplot[i];	p1.z = 0;
					mpc_marker.points.push_back(p1);
					p1.x = xptplot[i];	p1.y = yptplot[i];	p1.z = 0;
					mpc_marker.points.push_back(p1);
				}

				mpc_marker_pub.publish(mpc_marker);


				//Publish the wall lines
				wall_marker.header.frame_id = "base_link";
				wall_marker.header.stamp = ros::Time::now();
				wall_marker.type = visualization_msgs::Marker::LINE_LIST;
				wall_marker.id = 0; 
				wall_marker.action = visualization_msgs::Marker::ADD;
				wall_marker.scale.x = 0.1;
				wall_marker.color.a = 1.0;
				wall_marker.color.r = 0.1; 
				wall_marker.color.g = 0.9;
				wall_marker.color.b = 0;
				wall_marker.pose.orientation.w = 1;
				
				wall_marker.lifetime = ros::Duration(0.1);
				geometry_msgs::Point p2;
				wall_marker.points.clear();

				p2.x = dl*(-wl_h[0]-line_len*wl_h[1]);	p2.y = dl*(-wl_h[1]+line_len*wl_h[0]);	p2.z = 0; 
				wall_marker.points.push_back(p2);

				p2.x = dl*(-wl_h[0]+line_len*wl_h[1]);	p2.y = dl*(-wl_h[1]-line_len*wl_h[0]);	p2.z = 0;
				wall_marker.points.push_back(p2);

				p2.x = dr*(-wr_h[0]-line_len*wr_h[1]);	p2.y = dr*(-wr_h[1]+line_len*wr_h[0]);	p2.z = 0;
				wall_marker.points.push_back(p2);

				p2.x = dr*(-wr_h[0]+line_len*wr_h[1]);	p2.y = dr*(-wr_h[1]-line_len*wr_h[0]);	p2.z = 0;
				wall_marker.points.push_back(p2);

				wall_marker_pub.publish(wall_marker);

				//Publish the left obstacle points
				lobs_marker.header.frame_id = "base_link";
				lobs_marker.header.stamp = ros::Time::now();
				lobs_marker.type = visualization_msgs::Marker::LINE_LIST;
				lobs_marker.id = 0; 
				lobs_marker.action = visualization_msgs::Marker::ADD;
				lobs_marker.scale.x = 0.1;
				lobs_marker.color.a = 1.0;
				lobs_marker.color.r = 0; 
				lobs_marker.color.g = 0.5;
				lobs_marker.color.b = 0.5;
				lobs_marker.pose.orientation.w = 1;
				
				lobs_marker.lifetime = ros::Duration(0.1);
				geometry_msgs::Point p3;
				lobs_marker.points.clear();
				int count=0;
				for(int i=0;i<obstacle_points_l.size();i++){
					double po1=obstacle_points_l[i][0];
					double po2=obstacle_points_l[i][1];
					p3.x = po1;	p3.y = po2;	p3.z = 0;
					lobs_marker.points.push_back(p3);	
					count++;
				}
				if(count%2==1){
					p3.x = 0;	p3.y = 0;	p3.z = 0; 
					lobs_marker.points.push_back(p3);
				}

				lobs.publish(lobs_marker);

				//Publish the right obstacle points
				robs_marker.header.frame_id = "base_link";
				robs_marker.header.stamp = ros::Time::now();
				robs_marker.type = visualization_msgs::Marker::LINE_LIST;
				robs_marker.id = 0; 
				robs_marker.action = visualization_msgs::Marker::ADD;
				robs_marker.scale.x = 0.1;
				robs_marker.color.a = 1.0;
				robs_marker.color.r = 0; 
				robs_marker.color.g = 0;
				robs_marker.color.b = 1;
				robs_marker.pose.orientation.w = 1;
				
				robs_marker.lifetime = ros::Duration(0.1);

				geometry_msgs::Point p4;
				robs_marker.points.clear();
				
				count=0;
				for(int i=0;i<obstacle_points_r.size();i++){
					double po1=obstacle_points_r[i][0];
					double po2=obstacle_points_r[i][1];
					p4.x = po1;	p4.y = po2;	p4.z = 0;
				
					robs_marker.points.push_back(p4);	
					count++;
				}
				if(count%2==1){
					p4.x = 0;	p4.y = 0;	p4.z = 0; 
					robs_marker.points.push_back(p4);
				}

				robs.publish(robs_marker);







				//Ackermann Steering
				

				
				min_distance = max_lidar_range + 100; int idx1, idx2;
				idx1 = -sec_len+int(scan_beams/2); idx2 = sec_len + int(scan_beams/2);

				// for(int i = idx1; i <= idx2; ++i){
				// 	if(fused_ranges[i] < min_distance) min_distance = fused_ranges[i];
				// }
				for(int i=0;i<fused_ranges_MPC_tot0.size();i++){ //Consider all obstacles, not just sensor data
					if(std::abs(lidar_transform_angles_tot0[i])<heading_beam_angle && fused_ranges_MPC_tot0[i]<min_distance){
						min_distance=fused_ranges_MPC_tot0[i];
					}

				}
				for(int i=0;i<fused_ranges_MPC_veh_det0.size();i++){ //Consider the full detected vehicle trajectory
					if(std::abs(lidar_transform_angles_veh_det0[i])<heading_beam_angle && fused_ranges_MPC_veh_det0[i]<min_distance){
						min_distance=fused_ranges_MPC_veh_det0[i];
					}

				}

				velocity_scale = 1 - exp(-std::max(min_distance-stop_distance,0.0)/stop_distance_decay); //ensure we only slow when appropriate, otherwise MPC behaviour dominates
				

				delta_d=deltas[1]; //Use next delta command now to allow servo to transition

				velocity_MPC = velocity_scale*vehicle_velocity; //Implement slowing if we near an obstacle
				printf("Steering Angle: %lf, Velocity: %lf\n",delta_d,velocity_MPC);
				printf("*******************\n");

			}

			ackermann_msgs::AckermannDriveStamped drive_msg; 
			drive_msg.header.stamp = ros::Time::now();
			drive_msg.header.frame_id = "base_link";
			if(startcheck==1){
				drive_msg.drive.steering_angle = delta_d; //delta_d
				if(forcestop==0){ //If the optimization fails for some reason, we get nan: stop the vehicle
					drive_msg.drive.speed = velocity_MPC; //velocity_MPC
				}
				else{
					drive_msg.drive.speed = 0;
				}
			}
			else{
				drive_msg.drive.steering_angle = 0;
				drive_msg.drive.speed = 0;
			}
			


			driver_pub.publish(drive_msg);
		}


};

int main(int argc, char **argv){
		ros::init(argc, argv, "navigation");
		GapBarrier gb;

		while(ros::ok()){

			ros::spinOnce();
		}

}


