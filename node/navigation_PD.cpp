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

#include "sensor_msgs/Image.h" // ros image

#include <tf/tf.h> //Quaternions
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>


#include "ackermann_msgs/AckermannDriveStamped.h" //Ackermann Steering

#include "nav_msgs/Odometry.h" //Odometer

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
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp> 




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

class GapBarrier 
{
	private:
		ros::NodeHandle nf;


		//Subscriptions
		ros::Subscriber lidar;
		ros::Subscriber image, info, confidence; 
		ros::Subscriber imu;
		ros::Subscriber mux;
		ros::Subscriber odom;
		
		//More CV data members, used if use_camera is true
		ros::Subscriber depth_img;
		ros::Subscriber depth_info;
		ros::Subscriber depth_img_confidence;
		sensor_msgs::LaserScan cv_ranges_msg;
		int cv_rows, cv_cols;
		xt::xarray<int> cv_sample_rows_raw;
		xt::xarray<int> cv_sample_cols_raw;
		ros::Subscriber vesc_state_sub;
		ros::Subscriber servo_sub;
		ros::Subscriber tf_sub;

		


		
		//Publications
		ros::Publisher lidar_pub;
		ros::Publisher marker_pub;
		ros::Publisher driver_pub;
		ros::Publisher cv_ranges_pub;
		

		
		//topics
		std::string depth_image_topic, depth_info_topic, cv_ranges_topic, depth_index_topic, 
		depth_points_topic,lidarscan_topic, drive_topic, odom_topic, mux_topic, imu_topic;

		//time
		double current_time = ros::Time::now().toSec();
		double prev_time = current_time;
		double time_ref = 0.0; 
		double heading_beam_angle;
		double velocity=0;

		//lidar-preprocessing
		int scan_beams; double right_beam_angle, left_beam_angle;
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

		//imu
		double imu_roll, imu_pitch, imu_yaw;
		double delta_d=0;


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
		double mapx=0, mapy=0, maptheta=0;
		double odomx=0, odomy=0, odomtheta=0;
		double locx=0, locy=0, loctheta=0;
		double simx=0, simy=0, simtheta=0;
		ros::Time timestamp_tf1; ros::Time timestamp_tf2;
		double vel_adapt=0;
		double last_delta=0;
		double speed_to_erpm_gain, speed_to_erpm_offset;
		double steering_angle_to_servo_gain, steering_angle_to_servo_offset;
		std_msgs::Float64 last_servo_state;


		rs2_intrinsics intrinsics;
		bool intrinsics_defined;
		sensor_msgs::Image cv_image_data;
		bool cv_image_data_defined;

		//ground plane parameters
		float cv_groundplane_max_height; 
		float cv_groundplane_max_distance; 
		

	
		

		

	public:
		
		GapBarrier()
		{

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



			//lidar params
			nf.getParam("scan_beams", scan_beams);
			nf.getParam("right_beam_angle", right_beam_angle);
			nf.getParam("left_beam_angle", left_beam_angle);
			nf.getParam("scan_range", max_lidar_range);
			nf.getParam("safe_distance", safe_distance);

			//lidar init
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

			nf.getParam("speed_to_erpm_gain", speed_to_erpm_gain);
			nf.getParam("speed_to_erpm_offset", speed_to_erpm_offset);

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


			vel = 0.0;

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
			odom = nf.subscribe(odom_topic,1, &GapBarrier::odom_callback, this);
			vesc_state_sub= nf.subscribe("/sensors/core", 1, &GapBarrier::vesc_callback, this);
			servo_sub= nf.subscribe("/sensors/servo_position_command", 1,&GapBarrier::servo_callback, this);
			tf_sub = nf.subscribe("/tf", 20, &GapBarrier::tf_callback, this);




			//publications
			//lidar_pub = nf.advertise<std_msgs::Int32MultiArray>("chatter", 1000);
			marker_pub = nf.advertise<visualization_msgs::Marker>("wall_markers",2);
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


		void odom_callback(const nav_msgs::OdometryConstPtr& odom_msg){
			vel = odom_msg->twist.twist.linear.x;
			yaw = 2*atan2(odom_msg->pose.pose.orientation.z, odom_msg->pose.pose.orientation.w);
		}

		void mux_callback(const std_msgs::Int32MultiArrayConstPtr& data){nav_active = data->data[nav_mux_idx]; }

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
			if(intrinsics_defined)
			{
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




		std::pair < std::vector<std::vector<float>>, std::vector<float> >preprocess_lidar(std::vector<float> ranges)
		{

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
				else if(transform.header.frame_id == "map" && transform.child_frame_id == "base_link"){ //This is for simulation only
					vel_adapt=velocity; //In simulation, adaptive velocity is based on our input, not VESC since not connected
					
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

		void servo_callback(const std_msgs::Float64 & servo){
			last_servo_state=servo;
		}

		void vesc_callback(const vesc_msgs::VescStateStamped & state){
        	vel_adapt = std::max(-( state.state.speed - speed_to_erpm_offset ) / speed_to_erpm_gain,0.1);
			last_delta = ( last_servo_state.data - steering_angle_to_servo_offset) / steering_angle_to_servo_gain;
		}

		void getWalls(std::vector<std::vector<double>> obstacle_points_l, std::vector<std::vector<double>> obstacle_points_r,
		std::vector<double> wl0, std::vector<double> wr0, double alpha, std::vector<double> &wr, std::vector<double> &wl){
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


			}

		}







		void lidar_callback(const sensor_msgs::LaserScanConstPtr &data){
			ls_ang_inc = static_cast<double>(data->angle_increment); 
			scan_beams = int(2*M_PI/data->angle_increment);
			ls_str = int(round(scan_beams*right_beam_angle/(2*M_PI)));
			ls_end = int(round(scan_beams*left_beam_angle/(2*M_PI)));

			
			if (!nav_active) {
				drive_state = "normal";
				return;
			}


			std::vector<float> fused_ranges = data->ranges;

			// if(use_camera)
			// {
			// 	if(cv_image_data_defined){ augment_camera(fused_ranges); }
			// }

			std::pair<std::vector<std::vector<float>>, std::vector<float>> lidar_preprocess = preprocess_lidar(fused_ranges);

			std::vector<std::vector<float>> proc_ranges = lidar_preprocess.first;
			std::vector<float> mod_ranges = lidar_preprocess.second;
			
			ros::Time t = ros::Time::now();
			current_time = t.toSec();
			double dt = current_time - prev_time;
			prev_time = current_time;


			int sec_len = int(heading_beam_angle/data->angle_increment);

			double min_distance, velocity_scale;
			

			if(drive_state == "normal"){

					
				min_distance = max_lidar_range + 100;
				for(int i = 0; i < fused_ranges.size(); ++i){
					if(fused_ranges[i] < min_distance){
						min_distance = fused_ranges[i];
					}
				}


				int str_indx, end_indx, index_l, index_r, start_indx_l, start_indx_r; double heading_angle;
				std::pair<int,int> max_gap = find_max_gap(proc_ranges);
				str_indx = max_gap.first; end_indx = max_gap.second;



				heading_angle= find_best_point(str_indx, end_indx, proc_ranges);


				index_l = int(round((angle_bl-angle_al)/(data->angle_increment*n_pts_l)));
				index_r = int(round((angle_ar-angle_br)/(data->angle_increment*n_pts_r)));

				float mod_angle_al = angle_al + heading_angle;

				if(mod_angle_al > 2*M_PI) mod_angle_al -= 2*M_PI;
				else if (mod_angle_al < 0) mod_angle_al += 2*M_PI;

				float mod_angle_br = angle_br + heading_angle;

				if(mod_angle_br > 2*M_PI) mod_angle_br -= 2*M_PI;
				else if (mod_angle_br < 0) mod_angle_br += 2*M_PI;

				start_indx_l = int(round(mod_angle_al/data->angle_increment));
				start_indx_r = int(round(mod_angle_br/data->angle_increment));

				std::vector<std::vector<double>> obstacle_points_l;
				obstacle_points_l.push_back({0, max_lidar_range_opt});
				obstacle_points_l.push_back({1, max_lidar_range_opt}); 

				std::vector<std::vector<double>> obstacle_points_r;
				obstacle_points_r.push_back({0, -max_lidar_range_opt});
				obstacle_points_r.push_back({1, -max_lidar_range_opt});


				int k_obs = 0; int obs_index;

				double x_obs, y_obs;

				for(int k = 0; k < n_pts_l; ++k){
					obs_index = (start_indx_l + k*index_l) % scan_beams;

					double obs_range = static_cast<double>(fused_ranges[obs_index]);
					

					if(obs_range <= max_lidar_range_opt){


						if(k_obs == 0){
							obstacle_points_l[0] = {-obs_range*cos(mod_angle_al+k*index_l*data->angle_increment), -obs_range*sin(mod_angle_al+k*index_l*data->angle_increment) };
						}
						else if (k_obs == 1){
							obstacle_points_l[1] = {-obs_range*cos(mod_angle_al+k*index_l*data->angle_increment), -obs_range*sin(mod_angle_al+k*index_l*data->angle_increment) };
						}
						else{
							x_obs = -obs_range*cos(mod_angle_al+k*index_l*data->angle_increment);
							y_obs = -obs_range*sin(mod_angle_al+k*index_l*data->angle_increment);

							std::vector<double> obstacles = {x_obs, y_obs};
							obstacle_points_l.push_back(obstacles);
						}

						k_obs+=1;
					}

				}
				k_obs = 0;

				for(int k = 0; k < n_pts_r; ++k){
					obs_index = (start_indx_r+k*index_r) % scan_beams;
					double obs_range = static_cast<double>(fused_ranges[obs_index]);

					if(obs_range <= max_lidar_range_opt) {
						if(k_obs == 0){
							obstacle_points_r[0] = {-obs_range*cos(mod_angle_br+k*index_r*data->angle_increment), -obs_range*sin(mod_angle_br+k*index_r*data->angle_increment)};
						}
						else if(k_obs == 1){
							obstacle_points_r[1] = {-obs_range*cos(mod_angle_br+k*index_r*data->angle_increment),-obs_range*sin(mod_angle_br+k*index_r*data->angle_increment)};
						}
						else{
							x_obs = -obs_range*cos(mod_angle_br+k*index_r*data->angle_increment);
							y_obs = -obs_range*sin(mod_angle_br+k*index_r*data->angle_increment);

							std::vector<double> obstacles = {x_obs, y_obs};
							obstacle_points_r.push_back(obstacles);
	
						}

						k_obs += 1;
					}
				}

				


				double alpha = 1-exp(-dt/tau);

				std::vector<double> wl = {0.0, 0.0};
				std::vector<double> wr = {0.0, 0.0};


				getWalls(obstacle_points_l, obstacle_points_r, wl0, wr0, alpha, wr, wl);

				
				wl0[0] = wl[0]; wl0[1] = wl[1];
				wr0[0] = wr[0], wr0[1] = wr[1];

				double dl, dr; 
				double wr_dot, wl_dot; 
				wr_dot = wl_dot = 0;

				for(int i =0; i < 2; ++i){
					wl_dot += wl[i]*wl[i];
					wr_dot += wr[i]*wr[i];
				}

				dl = 1/sqrt(wl_dot); dr = 1/sqrt(wr_dot);

				std::vector<double> wr_h = {wr[0]*dr,wr[1]*dr}; std::vector<double> wl_h = {wl[0]*dl, wl[1]*dl};

				
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

				p.x = dl*(-wl_h[0]-line_len*wl_h[1]);	p.y = dl*(-wl_h[1]+line_len*wl_h[0]);	p.z = 0; 
				marker.points.push_back(p);

				p.x = dl*(-wl_h[0]+line_len*wl_h[1]);	p.y = dl*(-wl_h[1]-line_len*wl_h[0]);	p.z = 0;
				marker.points.push_back(p);

				p.x = dr*(-wr_h[0]-line_len*wr_h[1]);	p.y = dr*(-wr_h[1]+line_len*wr_h[0]);	p.z = 0;
				marker.points.push_back(p);

				p.x = dr*(-wr_h[0]+line_len*wr_h[1]);	p.y = dr*(-wr_h[1]-line_len*wr_h[0]);	p.z = 0;
				marker.points.push_back(p);

				p.x = 0; p.y = 0; p.z = 0;
				marker.points.push_back(p);

				p.x = line_len*cosf(heading_angle);	p.y = line_len*sinf(heading_angle);	p.z = 0;
				marker.points.push_back(p);

				marker_pub.publish(marker);


				//Ackermann Steering
				if(vel >= 0.01 || vel <= -0.01){
					double d_tilde, d_tilde_dot;
					d_tilde = dl - dr - CenterOffset;
					d_tilde_dot = vel*(wl_h[0]-wr_h[0]);
					delta_d = atan(wheelbase*(k_p*d_tilde + k_d*d_tilde_dot))/((std::pow(vel,2))*(-wl_h[1]+wr_h[1]));
				}
				else delta_d = 0.0;


				if(delta_d >= max_steering_angle) delta_d = max_steering_angle;
				else if (delta_d <= -max_steering_angle) delta_d = -max_steering_angle;

				min_distance = max_lidar_range + 100; int idx1, idx2;
				idx1 = -sec_len+int(scan_beams/2); idx2 = sec_len + int(scan_beams/2);

				for(int i = idx1; i <= idx2; ++i){
					if(fused_ranges[i] < min_distance){
						min_distance = fused_ranges[i];
					}
				}

				velocity_scale = 1 - exp(-std::max(min_distance-stop_distance,0.0)/stop_distance_decay);
				velocity = velocity_scale*vehicle_velocity;


			}

			
			ackermann_msgs::AckermannDriveStamped drive_msg; 
			drive_msg.header.stamp = ros::Time::now();
			drive_msg.header.frame_id = "base_link";
			drive_msg.drive.steering_angle = delta_d;
			drive_msg.drive.speed = velocity;

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


