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

#include <tf/tf.h> //Quaternions
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>


#include "ackermann_msgs/AckermannDriveStamped.h" //Ackermann Steering

#include "nav_msgs/Odometry.h" //Odometer
#include <nav_msgs/OccupancyGrid.h> //Map

#include <string>
#include <vector>


//CV includes
#include <cv_bridge/cv_bridge.h>
#include <librealsense2/rs.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include  <opencv2/core.hpp>
#include <opencv2/opencv.hpp>




//standard and external
#include <stdio.h>
#include <math.h> //cosf
#include <cmath> //M_PI, round
#include <sstream>
#include <algorithm>

#include <QuadProg++.hh>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp> 
#include <nlopt.hpp>

//C++ will auto typedef float3 data type
int nMPC=0; //Defined outside class to be used in predefined functions for nlopt MPC calculation
int kMPC=0;


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


double myfunc(unsigned n, const double *x, double *grad, void *my_func_data) //NLOPT cost function
{
	//Gradient calculated based on three parts, d part, d_dot due to p_dot for both current and then next point (obj is only nonzero partial x & y)
	double (*track_line)[nMPC*kMPC] = (double (*)[nMPC*kMPC]) my_func_data; //track_line is now the normal double array
	double funcreturn=0; //Create objective function as the sum of d and d_dot squared terms (d_dot part assumes constant w)
	int d_factor=1; //Change weighting of d vs d_dot terms
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
				grad[2*nMPC*kMPC+i]=grad[2*nMPC*kMPC+i]+2*track_line[0][i-1]*(track_line[0][i-1]*(x[2*nMPC*kMPC+i]-x[2*nMPC*kMPC+i-1])+track_line[1][i-1]*(x[3*nMPC*kMPC+i]-x[3*nMPC*kMPC+i-1]))/(pow(track_line[0][i-1],2)+pow(track_line[1][i-1],2));
				grad[3*nMPC*kMPC+i]=grad[3*nMPC*kMPC+i]+2*track_line[1][i-1]*(track_line[0][i-1]*(x[2*nMPC*kMPC+i]-x[2*nMPC*kMPC+i-1])+track_line[1][i-1]*(x[3*nMPC*kMPC+i]-x[3*nMPC*kMPC+i-1]))/(pow(track_line[0][i-1],2)+pow(track_line[1][i-1],2));
			}
			if(i<nMPC*kMPC-1){
				funcreturn=funcreturn+pow(track_line[0][i]*(x[2*nMPC*kMPC+i+1]-x[2*nMPC*kMPC+i])+track_line[1][i]*(x[3*nMPC*kMPC+i+1]-x[3*nMPC*kMPC+i]),2)/(pow(track_line[0][i],2)+pow(track_line[1][i],2));
				if(grad){
					grad[2*nMPC*kMPC+i]=grad[2*nMPC*kMPC+i]-2*track_line[0][i]*(track_line[0][i]*(x[2*nMPC*kMPC+i+1]-x[2*nMPC*kMPC+i])+track_line[1][i]*(x[3*nMPC*kMPC+i+1]-x[3*nMPC*kMPC+i]))/(pow(track_line[0][i],2)+pow(track_line[1][i],2));
					grad[3*nMPC*kMPC+i]=grad[3*nMPC*kMPC+i]-2*track_line[1][i]*(track_line[0][i]*(x[2*nMPC*kMPC+i+1]-x[2*nMPC*kMPC+i])+track_line[1][i]*(x[3*nMPC*kMPC+i+1]-x[3*nMPC*kMPC+i]))/(pow(track_line[0][i],2)+pow(track_line[1][i],2));
				}
			}
			funcreturn=funcreturn+pow(x[nMPC*kMPC+i],2); //The scaling factor of this term may need to be param, depends on speed (tuning)
			if(grad){
				grad[i]=0; //Gradients wrt theta = 0
				grad[nMPC*kMPC+i]=2*x[nMPC*kMPC+i];
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
	double *opt_params = (double (*))f_data; //[0]-> velocity/sample; [1]-> wheelabse (l) parameter
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
	double *opt_params = (double (*))f_data; //[0]-> velocity/sample; [1]-> wheelabse (l) parameter
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
	double *opt_params = (double (*))f_data; //[0]-> velocity/sample; [1]-> wheelabse (l) parameter; [2]-> max change in delta
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
	result[2*nMPC*kMPC-1]=x[nMPC*kMPC]-opt_params[3]; //opt_params[3] is the last delta (from previous iteration)
	result[2*nMPC*kMPC]=-x[nMPC*kMPC]+opt_params[3]; //Can't change servo instantly so delta[0] is fixed
	if(grad){
		grad[(2*nMPC*kMPC-1)*n+nMPC*kMPC]=1;
		grad[(2*nMPC*kMPC)*n+nMPC*kMPC]=-1;
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
		// ros::Subscriber odom;
		// ros::Subscriber localize;
		ros::Subscriber amcl_sub;
		ros::Subscriber tf_sub;
		ros::Subscriber map_sub;

		//More CV data members, used if use_camera is true
		ros::Subscriber depth_img;
		ros::Subscriber depth_info;
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
		ros::Publisher driver_pub;
		ros::Publisher cv_ranges_pub;


		
		//topics
		std::string depth_image_topic, depth_info_topic, cv_ranges_topic, depth_index_topic, 
		depth_points_topic,lidarscan_topic, drive_topic, odom_topic, mux_topic, imu_topic, map_topic;

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

		double steering_angle_to_servo_gain;
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

		double lastx=0, lasty=0, lasttheta=0;

		//odom and map transforms for map localization and tf of occupancy grid points
		double mapx=0, mapy=0, maptheta=0;
		double odomx=0, odomy=0, odomtheta=0;
		double locx=0, locy=0, loctheta=0;
		double simx=0, simy=0, simtheta=0;

		//MPC MAP localization parameters
		std::vector<std::vector<double>> map_pts;
		int map_saved=0;
		double map_thresh;
		int use_map=0; //Whether we use the pre-defined map as part of MPC

		//imu
		double imu_roll, imu_pitch, imu_yaw;


		//mux
		int nav_mux_idx; int nav_active; 

		//odom
		double yaw;


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


		rs2_intrinsics intrinsics;
		bool intrinsics_defined;
		sensor_msgs::Image cv_image_data;
		bool cv_image_data_defined;

		//ground plane parameters
		float cv_groundplane_max_height; 
		float cv_groundplane_max_distance; 

		

	public:
		
		GapBarrier(){

			nf = ros::NodeHandle("~");
			// topics	
			nf.getParam("depth_image_topic", depth_image_topic);
			nf.getParam("depth_info_topic", depth_info_topic);
			nf.getParam("cv_ranges_topic", cv_ranges_topic);
			nf.getParam("depth_index_topic", depth_index_topic);
			nf.getParam("depth_points_topic", depth_points_topic);
			nf.getParam("scan_topic", lidarscan_topic);
			nf.getParam("nav_drive_topic", drive_topic);
			nf.getParam("odom_topic", odom_topic);
			nf.getParam("mux_topic", mux_topic);
			nf.getParam("imu_topic", imu_topic);
			nf.getParam("map_topic", map_topic);



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
			nf.getParam("max_steering_vel", max_servo_speed);


			vel = 0.0;

			//MPC parameters
            nf.getParam("nMPC",nMPC);
            nf.getParam("kMPC",kMPC);
			nf.getParam("angle_thresh", angle_thresh);
			nf.getParam("map_thresh", map_thresh);
			nf.getParam("use_map", use_map);

			//MPC init
			default_dt=0.077;
			deltas.resize(nMPC*kMPC,0);
			thetas.resize(nMPC*kMPC,0);
			x_vehicle.resize(nMPC*kMPC,0);
			for(int i=1; i<nMPC*kMPC; i++){
				x_vehicle[i] = x_vehicle[i-1]+vehicle_velocity*default_dt;
				
			}
			y_vehicle.resize(nMPC*kMPC,0);
			last_delta=0;
			velocity_MPC=vehicle_velocity;
			



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



			intrinsics_defined= false;
        	cv_image_data_defined= false;

			//subscriptions
			lidar = nf.subscribe("/scan",1, &GapBarrier::lidar_callback, this);
			imu = nf.subscribe(imu_topic,1, &GapBarrier::imu_callback, this);
			mux = nf.subscribe(mux_topic,1, &GapBarrier::mux_callback, this);
			// odom = nf.subscribe(odom_topic,1, &GapBarrier::odom_callback, this);
			// localize = nf.subscribe("/pose_stamped",1, &GapBarrier::localize_callback, this);
			amcl_sub = nf.subscribe("/amcl_pose", 1, &GapBarrier::amcl_callback, this);
			tf_sub = nf.subscribe("/tf", 20, &GapBarrier::tf_callback, this);
			map_sub = nf.subscribe(map_topic, 1, &GapBarrier::map_callback, this);
			

			//publications
			//lidar_pub = nf.advertise<std_msgs::Int32MultiArray>("chatter", 1000);
			marker_pub = nf.advertise<visualization_msgs::Marker>("wall_markers",2);
			mpc_marker_pub = nf.advertise<visualization_msgs::Marker>("mpc_markers",2);
			wall_marker_pub=nf.advertise<visualization_msgs::Marker>("walls",2);
			lobs=nf.advertise<visualization_msgs::Marker>("lobs",2);
			robs=nf.advertise<visualization_msgs::Marker>("robs",2);
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
				// if(transform.header.frame_id == "map" && transform.child_frame_id == "base_link"){ //This is for simulation only
				// 	simx=transform.transform.translation.x;
				// 	simy=transform.transform.translation.y;
				// 	// 		transform.transform.translation.z);
				// 	double x=transform.transform.rotation.x;
				// 	double y=transform.transform.rotation.y;
				// 	double z=transform.transform.rotation.z;
				// 	double w=transform.transform.rotation.w;
				// 	simtheta = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
				// 	// printf("%lf, %lf, %lf, %lf\n",simx,simy,simtheta,ros::Time::now().toSec());
				// }
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
			printf("cov:%lf, %lf, %lf\n",amcl_msg.pose.covariance[0],amcl_msg.pose.covariance[7],amcl_msg.pose.covariance[35]);
			printf("pose:%lf, %lf, %lf\n",amcl_msg.pose.pose.position.x,amcl_msg.pose.pose.position.y,2*atan2(amcl_msg.pose.pose.orientation.z, amcl_msg.pose.pose.orientation.w));
			
		}

		// void odom_callback(const nav_msgs::OdometryConstPtr& odom_msg){
		// 	// FILE *file1 = fopen("/home/gjsk/topics1.txt", "a");
		// 	// ros::Time t19 = ros::Time::now();
		// 	// fprintf(file1,"ODOM: %lf\n",t19.toSec());
		// 	// fclose(file1);
		// 	vel = odom_msg->twist.twist.linear.x;
		// 	yaw = 2*atan2(odom_msg->pose.pose.orientation.z, odom_msg->pose.pose.orientation.w);
			
		// 	lastx=odom_msg->pose.pose.position.x;
		// 	lasty=odom_msg->pose.pose.position.y;
		// 	lasttheta=yaw;
		// 	// printf("odom:%lf, %lf, %lf\n",lastx,lasty,lasttheta);
		// 	if(abs(odom_msg->pose.pose.position.x)>0.1 && abs(odom_msg->pose.pose.position.y)>0.1){
		// 		// Open file in append mode
		// 		// FILE *file = fopen("/home/gjsk/output.txt", "a");
		// 		// if (file == NULL) {
		// 		// 	ROS_ERROR("Could not open file for writing.");
		// 		// 	return;
		// 		// }

		// 		// Write some odom data to the file (e.g., position data)
		// 		// fprintf(file,"%lf, %lf, %lf, %lf\n",odom_msg->pose.pose.position.x,odom_msg->pose.pose.position.y,yaw,t19.toSec());

		// 		// Close the file
		// 		// fclose(file);
		// 	}
		// 	// locx=odom_msg->pose.pose.position.x;
		// 	// locy=odom_msg->pose.pose.position.y;
		// 	// loctheta=yaw;



		// 	// ROS_INFO("%.3f", vel);
		// }

		void mux_callback(const std_msgs::Int32MultiArrayConstPtr& data){nav_active = data->data[nav_mux_idx]; }

		void imu_callback(const sensor_msgs::Imu::ConstPtr& data){

				tf::Quaternion myQuaternion(
				data->orientation.x,
				data->orientation.y,
				data->orientation.z,
				data->orientation.w);
			
			tf::Matrix3x3 m(myQuaternion);
			m.getRPY(imu_roll, imu_pitch, imu_yaw);
			// ROS_INFO("ROLL: %.3f, PITCH: %.3f, YAW: %.3f", imu_roll, imu_pitch, imu_yaw);

		}


		
		void imageDepth_callback( const sensor_msgs::ImageConstPtr & img)
		{
			if(intrinsics_defined)
			{
				//Unsure how copy consttuctor behaves, therefore manually copyied all data members
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
			if(intrinsics_defined){ return; }

			//std::cout << "Defining Intrinsics" <<std::endl;

            intrinsics.width = cameraInfo->width;
            intrinsics.height = cameraInfo->height;
            intrinsics.ppx = cameraInfo->K[2];
            intrinsics.ppy = cameraInfo->K[5];
            intrinsics.fx = cameraInfo->K[0];
            intrinsics.fy = cameraInfo->K[4];
			
            if (cameraInfo->distortion_model == "plumb_bob") 
			{
				intrinsics.model = RS2_DISTORTION_BROWN_CONRADY;   
			}
               
            else if (cameraInfo->distortion_model == "equidistant")
			{
				intrinsics.model = RS2_DISTORTION_KANNALA_BRANDT4;
			}
            for(int i=0; i<5; i++)
			{
				intrinsics.coeffs[i]=cameraInfo->D[i];
			}
			intrinsics_defined=true;

			cv_rows=intrinsics.height;
			cv_cols=intrinsics.width;

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
					rs2_deproject_pixel_to_point(cv_point.data(), &intrinsics, pixel, depth);

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
			cv::Mat cv_image=(cv_bridge::toCvCopy(cv_image_data,cv_image_data.encoding))->image; //Encoding type is 16UC1

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
					rs2_deproject_pixel_to_point(cv_point.data(), &intrinsics, pixel, depth);

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

					float lidar_coordx = -(cv_coordz+cv_distance_to_lidar);
                	float lidar_coordy = cv_coordx;
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
			for(int i =0; i < ranges.size(); ++i){
				if(lidar_angles[i] <= right_beam_angle_MPC) right_ind_MPC +=1;
				if(lidar_angles[i] <= left_beam_angle_MPC) left_ind_MPC +=1;
				if(right_ind_MPC!=i+1 && left_ind_MPC==i+1){
					if(ranges[i] <= safe_distance) {ranges[i] = 0;}
					else if(ranges[i] > max_lidar_range) {ranges[i] = max_lidar_range;}
				}
				
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
			double best_heading = 0; //Angles aren't evenly spaced so simple max distance suffices here, avoids complications	
			double max_range = 0; 
	
			for(int i=start_i; i<=end_i; ++i){
				if(proc_ranges[i] > max_range){
					max_range = proc_ranges[i];
					best_heading = lidar_transform_angles[i];
				}
				
			}
			
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
				// ss << xl[0] << " " << xl[1];
				// for(int j =0; j < int(obstacle_points_l.size()); ++j){
				// 	ss << obstacle_points_l[j][0] << " " << obstacle_points_l[j][1];
				// }
				// std_msgs::String msg; msg.data = ss.str();
				// ROS_INFO("%s", msg.data.c_str());
				// msg.data.clear();

				// ROS_INFO("%f, %f", wl[0], wl[1]);


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

				// ROS_INFO("%f, %f", wl[0], wl[1]);

				// std::stringstream ss; 

				// for(int i =0; i < n; ++i){
				// 	for(int j=0; j < n; ++j)
				// 	ss << Gr[i][j] << " ";
				// }

				// std_msgs::String msg; msg.data = ss.str();
				// ROS_INFO("%s", msg.data.c_str());
				// msg.data.clear();


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

				// std::stringstream ss;

				// ss << wl[0] << " " << wl[1];


				// for(int i =0; i < n; ++i)
				// 	for(int j =0; j < n; ++j) ss << G[i][j] << " ";


				// ss << x[0]/(x[2]-1) << " " << x[1]/(x[2]-1);
				// ss << solve_quadprog(G, gi0, CE, ce0, CI, ci0, x);

				// std_msgs::String msg; msg.data = ss.str();
				// ROS_INFO("%s", msg.data.c_str());
				// msg.data.clear();

			}

		}



		void lidar_callback(const sensor_msgs::LaserScanConstPtr &data){
			
			
			ls_ang_inc = static_cast<double>(data->angle_increment); 
			scan_beams = int(2*M_PI/data->angle_increment);
			ls_str = int(round(scan_beams*right_beam_angle/(2*M_PI)));
			ls_end = int(round(scan_beams*left_beam_angle/(2*M_PI)));



			if (!nav_active ||(use_map && !map_saved)) { //Don't start navigation until map is saved if that's what we're using
				drive_state = "normal";
				return;
			}
			
			 


			//pre-processing
			// std::vector<double> double_data; double value;
			// for(int i =0; i < int(data->ranges.size()); ++i){
			// 	value = static_cast<double>(data->ranges[i]);
			// 	double_data.push_back(value);
			// }
			// std::transform(data->ranges.begin(), data->ranges.end(), std::back_inserter(double_data),[](float value)
			// {return static_cast<double>(value); });


			std::vector<float> fused_ranges = data->ranges;
			// if(use_camera)
			// {
			// 	if(cv_image_data_defined){ augment_camera(fused_ranges); }
			// }

	
			// publish_lidar(mod_ranges);

			// std::stringstream ss; 
			// for(int i =0; i < ls_len_mod ;++i){
			// 	for(int j =0; j < 2; ++j){
			// 		ss << proc_ranges[i][j] << " ";
			// 	}
			// }
			// std_msgs::String msg; msg.data = ss.str();
			// ROS_INFO("%s", msg.data.c_str());


			//TODO: ADD TIME STUFF HERE
			ros::Time t = ros::Time::now();
			current_time = t.toSec();
			double dt = current_time - prev_time;
			if(dt>1){
				dt=default_dt;
			}
			prev_time = current_time;
			

			int sec_len = int(heading_beam_angle/data->angle_increment);

			double min_distance, velocity_scale, delta_d;
			

			if(drive_state == "normal"){
				std::vector<float> fused_ranges_MPC=fused_ranges;


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

				for (int num_MPC=0;num_MPC<nMPC;num_MPC++){
					std::vector<float> fused_ranges_MPC_map;
					std::vector<double> lidar_transform_angles_map; //These are the additional ranges & angles from fixed map that will be sorted, included in obs calculations
					std::vector<float> fused_ranges_MPC_tot=fused_ranges_MPC;
					std::vector<double> lidar_transform_angles_tot=lidar_transform_angles; //The cumulative ranges and angles for both map (if used) and lidar

					if(use_map){ //Augment LIDAR with map obstacle points too
						double map_xval=mapped_x+cos(mapped_theta)*xpt-sin(mapped_theta)*ypt;
						double map_yval=mapped_y+sin(mapped_theta)*xpt+cos(mapped_theta)*ypt;
						for(int i=0;i<map_pts.size();i++){
							if(pow(map_pts[i][0]-map_xval,2)+pow(map_pts[i][1]-map_yval,2)<pow(max_lidar_range_opt,2)){
								double x_base=(map_pts[i][0]-locx)*cos(loctheta)+(map_pts[i][1]-locy)*sin(loctheta)-xpt;
								double y_base=-(map_pts[i][0]-locx)*sin(loctheta)+(map_pts[i][1]-locy)*cos(loctheta)-ypt;
								double ang_base=atan2(y_base,x_base)-theta_ref;
								if (ang_base>M_PI) ang_base-=2*M_PI;
								if (ang_base<-M_PI) ang_base+=2*M_PI;
								lidar_transform_angles_map.push_back(ang_base);
								fused_ranges_MPC_map.push_back(pow(pow(x_base,2)+pow(y_base,2),0.5));
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

						std::vector<std::vector<double>> obstacle_points_l_MPC;
						obstacle_points_l_MPC.push_back({0, max_lidar_range_opt});
						obstacle_points_l_MPC.push_back({1, max_lidar_range_opt}); 
						
						
						std::vector<std::vector<double>> obstacle_points_r_MPC;
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
					xpt=startx+vehicle_velocity*std::max(default_dt,dt)*kMPC*cos(theta_ref); //Drive message relates to lidar callback scan topic, ~10Hz
					ypt=starty+vehicle_velocity*std::max(default_dt,dt)*kMPC*sin(theta_ref); //Use 13 Hz as absolute optimal but likely slower use dt
					
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
				
				
				double opt_params[4]={vehicle_velocity*std::max(default_dt,dt),wheelbase,std::abs(max_servo_speed*std::max(default_dt,dt)),last_delta};
				
				nlopt_add_equality_mconstraint(opt, nMPC*kMPC-1, theta_equality_con, &opt_params, tol);
				nlopt_add_equality_mconstraint(opt, nMPC*kMPC-1, x_equality_con, &opt_params, tol);
				nlopt_add_equality_mconstraint(opt, nMPC*kMPC-1, y_equality_con, &opt_params, tol);
				nlopt_add_inequality_mconstraint(opt, 2*nMPC*kMPC+1, delta_inequality_con, &opt_params, tol1);

			
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
							if(theta_refs[j]>0){
								deltas[i+j*kMPC]=std::min(last_delta+opt_params[2],max_steering_angle);
							}
							else if(theta_refs[j]<0){
								deltas[i+j*kMPC]=std::max(last_delta-opt_params[2],-max_steering_angle);
							}
							else{
								deltas[i+j*kMPC]=last_delta;
							}
						}
						else{
							if(theta_refs[j]>thetas[i+j*kMPC]){
								deltas[i+j*kMPC]=std::min(deltas[i+j*kMPC-1]+opt_params[2],max_steering_angle);
							}
							else if(theta_refs[j]<thetas[i+j*kMPC]){
								deltas[i+j*kMPC]=std::max(deltas[i+j*kMPC-1]-opt_params[2],-max_steering_angle);
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
				nlopt_result optim= nlopt_optimize(opt, x, &minf); //This runs the optimization

				if(isnan(minf)){
					forcestop=1;
				}
				else{
					forcestop=0;
				}

				if (optim < 0) {
					ROS_INFO("Optimization Error");
				}
				else {
					successful_opt=1;
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

				for(int i = idx1; i <= idx2; ++i){
					if(fused_ranges[i] < min_distance) min_distance = fused_ranges[i];
				}

				velocity_scale = 1 - exp(-std::max(min_distance-stop_distance,0.0)/stop_distance_decay); //2 factor ensures we only slow when appropriate, otherwise MPC behaviour dominates
				
				// ROS_INFO("%.3f", velocity);

				delta_d=deltas[1]; //Use next delta command now to allow servo to transition

				velocity_MPC = velocity_scale*vehicle_velocity; //Implement slowing if we near an obstacle


			}

			ackermann_msgs::AckermannDriveStamped drive_msg; 
			drive_msg.header.stamp = ros::Time::now();
			drive_msg.header.frame_id = "base_link";
			if(startcheck==1){
				drive_msg.drive.steering_angle = delta_d; 
				if(forcestop==0){ //If the optimization fails for some reason, we get nan: stop the vehicle
					drive_msg.drive.speed = velocity_MPC;
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
			// std_msgs::String msg;
			// std::stringstream ss;
			// ss  << gb.preprocess_lidar(); 
			// msg.data = ss.str();

			// ROS_INFO("%s", msg.data.c_str());

			// chatter_pub.publish(msg);
			ros::spinOnce();
		}

}


