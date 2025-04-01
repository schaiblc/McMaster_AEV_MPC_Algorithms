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
#include <string.h>
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
	double* raw_data = static_cast<double*>(my_func_data); //First extract as 1D array to get the column count (first value passed)
	int cols = static_cast<int>(raw_data[0]); 

	double xopt[2][cols]; //2D array of obstacle locations appended to some certain variables
	for (int i = 0; i < cols; i++) {
        xopt[0][i] = raw_data[2*i];
        xopt[1][i] = raw_data[2*i + 1];
    }

	int bez_ctrl_pts=xopt[1][0]; //Order of the Bezier Curve
	int bez_curv_pts=xopt[0][1]; //Discretized points on our curve
	double bez_alpha=xopt[1][1]; //Shaping of the exponential decay for further points
	double x1=xopt[0][2]; //These are fixed by initial conditions and thus aren't variables in optimization
	double y2=xopt[1][2];
	double x2_lead=xopt[0][3];
	double x3_lead=xopt[0][4];
	double y3_lead=xopt[1][4];
	double x4_lead=xopt[0][5];
	double y4_lead=xopt[1][5];
	double pursuit_weight=xopt[0][6];
	int leader_detect=xopt[1][6];

	double base_pursue_w=1; //How to scale each term before applying the time-variant transistion scaling via pursuit_weight
	
	std::vector<std::vector<double>> bez_curv;
	//Optimization variables:
	//[0] -> x2
	//[1] -> x3
	//[2] -> y3
	//[3] -> x4
	//[4] -> y4

	//Create the discretized Bezier Curve
	for(int i=0; i<bez_curv_pts; i++){
		double t=double(i)/double(bez_curv_pts-1);
		double bez_x=4*pow(1-t,3)*t*x1+6*pow(1-t,2)*pow(t,2)*x[0]+4*(1-t)*pow(t,3)*x[1]+pow(t,4)*x[3];
		double bez_y=6*pow(1-t,2)*pow(t,2)*y2+4*(1-t)*pow(t,3)*x[2]+pow(t,4)*x[4]; //y1=0
		bez_curv.push_back({bez_x,bez_y});
	}

	double funcreturn=0.0;
	if(grad){
		for(int i=0;i<n;i++){
			grad[i]=0.0;
		}
	}

	for(int i=0;i<bez_curv_pts;i++){
		double t=double(i)/double(bez_curv_pts-1);
		double px2=6*pow(1-t,2)*pow(t,2);
		double px3=4*(1-t)*pow(t,3);
		double py3=px3;
		double px4=pow(t,4);
		double py4=px4;
		for(int j=0;j<cols-3;j++){
			double dist2= pow(bez_curv[i][0]-xopt[0][j+7],2)+pow(bez_curv[i][1]-xopt[1][j+7],2); //Squared distance
			// printf("D2: %e, %e\n",(1/dist2)*exp(-bez_alpha*dist2),funcreturn);
			
			// if(bez_alpha*dist2>100) continue; //Numerical error since too small so skip this obs
			
			funcreturn=funcreturn+(1-pursuit_weight)*(1.0/dist2)*exp(-bez_alpha*dist2); //Sum of reciprocal squared distances, exponentially decaying weight
			//Next, find grad for each of five variables
			if(grad){
	/* x2 */ 	grad[0]=grad[0]-(1-pursuit_weight)*2*(bez_curv[i][0]-xopt[0][j+7])*(bez_alpha/dist2+1.0/pow(dist2,2))*exp(-bez_alpha*dist2)*px2;
	/* x3 */ 	grad[1]=grad[1]-(1-pursuit_weight)*2*(bez_curv[i][0]-xopt[0][j+7])*(bez_alpha/dist2+1.0/pow(dist2,2))*exp(-bez_alpha*dist2)*px3;
	/* y3 */ 	grad[2]=grad[2]-(1-pursuit_weight)*2*(bez_curv[i][1]-xopt[1][j+7])*(bez_alpha/dist2+1.0/pow(dist2,2))*exp(-bez_alpha*dist2)*py3;
	/* x4 */ 	grad[3]=grad[3]-(1-pursuit_weight)*2*(bez_curv[i][0]-xopt[0][j+7])*(bez_alpha/dist2+1.0/pow(dist2,2))*exp(-bez_alpha*dist2)*px4;
	/* y4 */ 	grad[4]=grad[4]-(1-pursuit_weight)*2*(bez_curv[i][1]-xopt[1][j+7])*(bez_alpha/dist2+1.0/pow(dist2,2))*exp(-bez_alpha*dist2)*py4;
			}

		}
		
	}

	//Now, the pursuit terms for minimizing distance to the leader's bezier control points
	if(leader_detect==1){
		//For (x2,y2) where y2 is fixed
		funcreturn=funcreturn+base_pursue_w*pursuit_weight*pow(x[0]-x2_lead,2);
		grad[0]=grad[0]+base_pursue_w*pursuit_weight*2*(x[0]-x2_lead);
		
		//For (x3,y3)
		funcreturn=funcreturn+base_pursue_w*pursuit_weight*(pow(x[1]-x3_lead,2)+pow(x[2]-y3_lead,2));
		grad[1]=grad[1]+base_pursue_w*pursuit_weight*2*(x[1]-x3_lead);
		grad[2]=grad[2]+base_pursue_w*pursuit_weight*2*(x[2]-y3_lead);
		
		//For (x4,y4)
		funcreturn=funcreturn+base_pursue_w*pursuit_weight*(pow(x[3]-x4_lead,2)+pow(x[4]-y4_lead,2));
		grad[3]=grad[3]+base_pursue_w*pursuit_weight*2*(x[3]-x4_lead);
		grad[4]=grad[4]+base_pursue_w*pursuit_weight*2*(x[4]-y4_lead);
		
	}

	return funcreturn;
}


void bezier_inequality_con(unsigned m, double *result, unsigned n, const double* x, double* grad, void* my_func_data){ //Bezier Curve inequalities
	
	double* raw_data = static_cast<double*>(my_func_data); //First extract as 1D array to get the column count (first value passed)
	int cols = static_cast<int>(raw_data[0]); 
	double xopt[2][cols]; //2D array of obstacle locations appended to some certain variables
	for (int i = 0; i < cols; i++) {
        xopt[0][i] = raw_data[2*i];
        xopt[1][i] = raw_data[2*i + 1];
    }

	int bez_ctrl_pts=xopt[1][0]; //Order of the Bezier Curve
	int bez_curv_pts=xopt[0][1]; //Discretized points on our curve
	int bez_beta=xopt[1][1]; //Large value to use softmin function which is differentiable (different from alpha used in myfunc)
	double x1=xopt[0][2]; //These are fixed by initial conditions and thus aren't variables in optimization
	double y2=xopt[1][2];
	double max_v=xopt[0][3]; //Highest allowed velocity
	double min_v=xopt[1][3]; //Minimum allowed velocity (if set lower, our vehicle just stops)
	double max_dv=xopt[0][4]; //Max allowable change in m/s^2 from motor (acceleration)
	double max_delta=xopt[1][4]; //Max allowable steering angle
	double max_ddelta=xopt[0][5]; //Max change in steering angle
	double t_end=xopt[1][5]; //Temporal scaling of the Bezier Curve
	double wheelbase=xopt[0][6]; //Physical constant property of vehicle
	double bez_min_dist=xopt[1][6]; //Our constraint on minimum distance to an obstacle
	std::vector<std::vector<double>> bez_curv;
	//Optimization variables:
	//[0] -> x2
	//[1] -> x3
	//[2] -> y3
	//[3] -> x4
	//[4] -> y4
	int nanflag=0;
	//Create the discretized Bezier Curve
	for(int i=0; i<bez_curv_pts; i++){
		double t=double(i)/double(bez_curv_pts);
		double bez_x=4*pow(1-t,3)*t*x1+6*pow(1-t,2)*pow(t,2)*x[0]+4*(1-t)*pow(t,3)*x[1]+pow(t,4)*x[3];
		double bez_y=6*pow(1-t,2)*pow(t,2)*y2+4*(1-t)*pow(t,3)*x[2]+pow(t,4)*x[4]; //y1=0
		bez_curv.push_back({bez_x,bez_y});
	}

	if(result){
		for(int i=0;i<=(bez_curv_pts-1)*9+8;i++){
			result[i]=0;
		}
	}

	if(grad){
		for(int i=0;i<n*m;i++){
			grad[i]=0;
		}
	}

	for(int i=0;i<bez_curv_pts; i++){
		double t=double(i)/double(bez_curv_pts);
		double x_dot=4*x1*(-4*pow(t,3)+9*pow(t,2)-6*t+1)+6*x[0]*(4*pow(t,3)-6*pow(t,2)+2*t)+4*x[1]*(-4*pow(t,3)+3*pow(t,2))+4*x[3]*pow(t,3);
		double x_ddot=4*x1*(-12*pow(t,2)+18*t-6)+6*x[0]*(12*pow(t,2)-12*t+2)+4*x[1]*(-12*pow(t,2)+6*t)+12*x[3]*pow(t,2);
		double x_dddot=4*x1*(-24*t+18)+6*x[0]*(24*t-12)+4*x[1]*(-24*t+6)+24*x[3]*t;

		double y_dot=6*y2*(4*pow(t,3)-6*pow(t,2)+2*t)+4*x[2]*(-4*pow(t,3)+3*pow(t,2))+4*x[4]*pow(t,3);
		double y_ddot=6*y2*(12*pow(t,2)-12*t+2)+4*x[2]*(-12*pow(t,2)+6*t)+12*x[4]*pow(t,2);
		double y_dddot=6*y2*(24*t-12)+4*x[2]*(-24*t+6)+24*x[4]*t;

		double curv=(x_dot*y_ddot-y_dot*x_ddot)/(pow(pow(x_dot,2)+pow(y_dot,2),1.5));
		double curv_dot=((x_dot*y_dddot-y_dot*x_dddot)*(pow(x_dot,2)+pow(y_dot,2))-3*(x_dot*x_ddot+y_dot*y_ddot)*(x_dot*y_ddot-y_dot*x_ddot))/pow((pow(x_dot,2)+pow(y_dot,2)),2.5);

		double px2=6*pow(1-t,2)*pow(t,2);double px3=4*(1-t)*pow(t,3);double py3=px3;double px4=pow(t,4);double py4=px4; //Partials of original x & y
		double pdx2=6*(4*pow(t,3)-6*pow(t,2)+2*t);double pdx3=4*(-4*pow(t,3)+3*pow(t,2));double pdy3=pdx3;double pdx4=4*pow(t,3);double pdy4=pdx4; //X_dot & y_dot
		double pddx2=6*(12*pow(t,2)-12*t+2);double pddx3=4*(-12*pow(t,2)+6*t);double pddy3=pddx3;double pddx4=12*pow(t,2);double pddy4=pddx4; //x_ddot & y_ddot
		double pdddx2=6*(24*t-12);double pdddx3=4*(-24*t+6);double pdddy3=pdddx3;double pdddx4=24*t;double pdddy4=pdddx4; //x_dddot & y_dddot
		
		double pcurvx2=((y_ddot*pdx2-y_dot*pddx2)*(pow(x_dot,2)+pow(y_dot,2))-3*x_dot*pdx2*(x_dot*y_ddot-y_dot*x_ddot))/pow(pow(x_dot,2)+pow(y_dot,2),2.5);
		double pcurvx3=((y_ddot*pdx3-y_dot*pddx3)*(pow(x_dot,2)+pow(y_dot,2))-3*x_dot*pdx3*(x_dot*y_ddot-y_dot*x_ddot))/pow(pow(x_dot,2)+pow(y_dot,2),2.5);
		double pcurvy3=((x_dot*pddy3-x_ddot*pdy3)*(pow(x_dot,2)+pow(y_dot,2))-3*y_dot*pdy3*(x_dot*y_ddot-y_dot*x_ddot))/pow(pow(x_dot,2)+pow(y_dot,2),2.5);
		double pcurvx4=((y_ddot*pdx4-y_dot*pddx4)*(pow(x_dot,2)+pow(y_dot,2))-3*x_dot*pdx4*(x_dot*y_ddot-y_dot*x_ddot))/pow(pow(x_dot,2)+pow(y_dot,2),2.5);
		double pcurvy4=((x_dot*pddy4-x_ddot*pdy4)*(pow(x_dot,2)+pow(y_dot,2))-3*y_dot*pdy4*(x_dot*y_ddot-y_dot*x_ddot))/pow(pow(x_dot,2)+pow(y_dot,2),2.5);
		
		double num_pdcurvx2=(y_dddot*pdx2-y_dot*pdddx2)*(pow(x_dot,2)+pow(y_dot,2))+2*x_dot*pdx2*(x_dot*y_dddot-y_dot*x_dddot)-3*((x_dot*pddx2+x_ddot*pdx2)*(x_dot*y_ddot-y_dot*x_ddot)+(y_ddot*pdx2-y_dot*pddx2)*(x_dot*x_ddot+y_dot*y_ddot));
		double num_pdcurvx3=(y_dddot*pdx3-y_dot*pdddx3)*(pow(x_dot,2)+pow(y_dot,2))+2*x_dot*pdx3*(x_dot*y_dddot-y_dot*x_dddot)-3*((x_dot*pddx3+x_ddot*pdx3)*(x_dot*y_ddot-y_dot*x_ddot)+(y_ddot*pdx3-y_dot*pddx3)*(x_dot*x_ddot+y_dot*y_ddot));
		double num_pdcurvy3=(x_dot*pdddy3-x_dddot*pdy3)*(pow(x_dot,2)+pow(y_dot,2))+2*y_dot*pdy3*(x_dot*y_dddot-y_dot*x_dddot)-3*((y_dot*pddy3+y_ddot*pdy3)*(x_dot*y_ddot-y_dot*x_ddot)+(x_dot*pddy3-x_ddot*pdy3)*(x_dot*x_ddot+y_dot*y_ddot));
		double num_pdcurvx4=(y_dddot*pdx4-y_dot*pdddx4)*(pow(x_dot,2)+pow(y_dot,2))+2*x_dot*pdx4*(x_dot*y_dddot-y_dot*x_dddot)-3*((x_dot*pddx4+x_ddot*pdx4)*(x_dot*y_ddot-y_dot*x_ddot)+(y_ddot*pdx4-y_dot*pddx4)*(x_dot*x_ddot+y_dot*y_ddot));
		double num_pdcurvy4=(x_dot*pdddy4-x_dddot*pdy4)*(pow(x_dot,2)+pow(y_dot,2))+2*y_dot*pdy4*(x_dot*y_dddot-y_dot*x_dddot)-3*((y_dot*pddy4+y_ddot*pdy4)*(x_dot*y_ddot-y_dot*x_ddot)+(x_dot*pddy4-x_ddot*pdy4)*(x_dot*x_ddot+y_dot*y_ddot));

		double pdcurvx2=(num_pdcurvx2*pow(pow(x_dot,2)+pow(y_dot,2),2.5)-((x_dot*y_dddot-y_dot*x_dddot)*(pow(x_dot,2)+pow(y_dot,2))-3*(x_dot*x_ddot+y_dot*y_ddot)*(x_dot*y_ddot-y_dot*x_ddot))*5*pow(pow(x_dot,2)+pow(y_dot,2),1.5)*x_dot*pdx2)/pow(pow(x_dot,2)+pow(y_dot,2),5);
		double pdcurvx3=(num_pdcurvx3*pow(pow(x_dot,2)+pow(y_dot,2),2.5)-((x_dot*y_dddot-y_dot*x_dddot)*(pow(x_dot,2)+pow(y_dot,2))-3*(x_dot*x_ddot+y_dot*y_ddot)*(x_dot*y_ddot-y_dot*x_ddot))*5*pow(pow(x_dot,2)+pow(y_dot,2),1.5)*x_dot*pdx3)/pow(pow(x_dot,2)+pow(y_dot,2),5);
		double pdcurvy3=(num_pdcurvy3*pow(pow(x_dot,2)+pow(y_dot,2),2.5)-((x_dot*y_dddot-y_dot*x_dddot)*(pow(x_dot,2)+pow(y_dot,2))-3*(x_dot*x_ddot+y_dot*y_ddot)*(x_dot*y_ddot-y_dot*x_ddot))*5*pow(pow(x_dot,2)+pow(y_dot,2),1.5)*y_dot*pdy3)/pow(pow(x_dot,2)+pow(y_dot,2),5);
		double pdcurvx4=(num_pdcurvx4*pow(pow(x_dot,2)+pow(y_dot,2),2.5)-((x_dot*y_dddot-y_dot*x_dddot)*(pow(x_dot,2)+pow(y_dot,2))-3*(x_dot*x_ddot+y_dot*y_ddot)*(x_dot*y_ddot-y_dot*x_ddot))*5*pow(pow(x_dot,2)+pow(y_dot,2),1.5)*x_dot*pdx4)/pow(pow(x_dot,2)+pow(y_dot,2),5);
		double pdcurvy4=(num_pdcurvy4*pow(pow(x_dot,2)+pow(y_dot,2),2.5)-((x_dot*y_dddot-y_dot*x_dddot)*(pow(x_dot,2)+pow(y_dot,2))-3*(x_dot*x_ddot+y_dot*y_ddot)*(x_dot*y_ddot-y_dot*x_ddot))*5*pow(pow(x_dot,2)+pow(y_dot,2),1.5)*y_dot*pdy4)/pow(pow(x_dot,2)+pow(y_dot,2),5);


		//Max and min velocity
		result[9*i]=pow(x_dot,2)+pow(y_dot,2)-pow(max_v,2)*pow(t_end,2);
		result[9*i+1]=-pow(x_dot,2)-pow(y_dot,2)+pow(min_v,2)*pow(t_end,2);

		if(grad){
	/*M_x2*/grad[9*i*n]=2*x_dot*pdx2;
	/*M_x3*/grad[9*i*n+1]=2*x_dot*pdx3;
	/*M_y3*/grad[9*i*n+2]=2*y_dot*pdy3;
	/*M_x4*/grad[9*i*n+3]=2*x_dot*pdx4;
	/*M_y4*/grad[9*i*n+4]=2*y_dot*pdy4;

	/*m_x2*/grad[(9*i+1)*n]=-2*x_dot*pdx2;
	/*m_x3*/grad[(9*i+1)*n+1]=-2*x_dot*pdx3;
	/*m_y3*/grad[(9*i+1)*n+2]=-2*y_dot*pdy3;
	/*m_x4*/grad[(9*i+1)*n+3]=-2*x_dot*pdx4;
	/*m_y4*/grad[(9*i+1)*n+4]=-2*y_dot*pdy4;
		}

		//Max change (+ & -) in velocity
		result[9*i+2]=(pow(x_dot*x_ddot+y_dot*y_ddot,2)/(pow(x_dot,2)+pow(y_dot,2)))-pow(max_dv,2)*pow(t_end,4);
		result[9*i+3]=-(pow(x_dot*x_ddot+y_dot*y_ddot,2)/(pow(x_dot,2)+pow(y_dot,2)))-pow(max_dv,2)*pow(t_end,4);
		if(isnan(result[9*i+2])){
			printf("NAN %lf, %lf, %lf, %lf, %lf, %lf\n",t,x[0],x[1],x[2],x[3],x[4]);
		}

		if(grad){
	/*+_x2*/grad[(9*i+2)*n]=2*(x_dot*x_ddot+y_dot*y_ddot)*((pdx2*x_ddot+pddx2*x_dot)*(pow(x_dot,2)+pow(y_dot,2))-x_dot*pdx2*(x_dot*x_ddot+y_dot*y_ddot))/pow(pow(x_dot,2)+pow(y_dot,2),2);
	/*+_x3*/grad[(9*i+2)*n+1]=2*(x_dot*x_ddot+y_dot*y_ddot)*((pdx3*x_ddot+pddx3*x_dot)*(pow(x_dot,2)+pow(y_dot,2))-x_dot*pdx3*(x_dot*x_ddot+y_dot*y_ddot))/pow(pow(x_dot,2)+pow(y_dot,2),2);
	/*+_y3*/grad[(9*i+2)*n+2]=2*(x_dot*x_ddot+y_dot*y_ddot)*((pdy3*y_ddot+pddy3*y_dot)*(pow(x_dot,2)+pow(y_dot,2))-y_dot*pdy3*(x_dot*x_ddot+y_dot*y_ddot))/pow(pow(x_dot,2)+pow(y_dot,2),2);
	/*+_x4*/grad[(9*i+2)*n+3]=2*(x_dot*x_ddot+y_dot*y_ddot)*((pdx4*x_ddot+pddx4*x_dot)*(pow(x_dot,2)+pow(y_dot,2))-x_dot*pdx4*(x_dot*x_ddot+y_dot*y_ddot))/pow(pow(x_dot,2)+pow(y_dot,2),2);
	/*+_y4*/grad[(9*i+2)*n+4]=2*(x_dot*x_ddot+y_dot*y_ddot)*((pdy4*y_ddot+pddy4*y_dot)*(pow(x_dot,2)+pow(y_dot,2))-y_dot*pdy4*(x_dot*x_ddot+y_dot*y_ddot))/pow(pow(x_dot,2)+pow(y_dot,2),2);

	/*-_x2*/grad[(9*i+3)*n]=-(2*(x_dot*x_ddot+y_dot*y_ddot)*((pdx2*x_ddot+pddx2*x_dot)*(pow(x_dot,2)+pow(y_dot,2))-x_dot*pdx2*(x_dot*x_ddot+y_dot*y_ddot))/pow(pow(x_dot,2)+pow(y_dot,2),2));
	/*-_x3*/grad[(9*i+3)*n+1]=-(2*(x_dot*x_ddot+y_dot*y_ddot)*((pdx3*x_ddot+pddx3*x_dot)*(pow(x_dot,2)+pow(y_dot,2))-x_dot*pdx3*(x_dot*x_ddot+y_dot*y_ddot))/pow(pow(x_dot,2)+pow(y_dot,2),2));
	/*-_y3*/grad[(9*i+3)*n+2]=-(2*(x_dot*x_ddot+y_dot*y_ddot)*((pdy3*y_ddot+pddy3*y_dot)*(pow(x_dot,2)+pow(y_dot,2))-y_dot*pdy3*(x_dot*x_ddot+y_dot*y_ddot))/pow(pow(x_dot,2)+pow(y_dot,2),2));
	/*-_x4*/grad[(9*i+3)*n+3]=-(2*(x_dot*x_ddot+y_dot*y_ddot)*((pdx4*x_ddot+pddx4*x_dot)*(pow(x_dot,2)+pow(y_dot,2))-x_dot*pdx4*(x_dot*x_ddot+y_dot*y_ddot))/pow(pow(x_dot,2)+pow(y_dot,2),2));
	/*-_y4*/grad[(9*i+3)*n+4]=-(2*(x_dot*x_ddot+y_dot*y_ddot)*((pdy4*y_ddot+pddy4*y_dot)*(pow(x_dot,2)+pow(y_dot,2))-y_dot*pdy4*(x_dot*x_ddot+y_dot*y_ddot))/pow(pow(x_dot,2)+pow(y_dot,2),2));
		}

		//Max (+ & -) curvature
		result[9*i+4]=wheelbase*curv-tan(max_delta);
		result[9*i+5]=-wheelbase*curv-tan(max_delta);

		if(grad){
	/*+_x2*/grad[(9*i+4)*n]=wheelbase*pcurvx2;
	/*+_x3*/grad[(9*i+4)*n+1]=wheelbase*pcurvx3;
	/*+_y3*/grad[(9*i+4)*n+2]=wheelbase*pcurvy3;
	/*+_x4*/grad[(9*i+4)*n+3]=wheelbase*pcurvx4;
	/*+_y4*/grad[(9*i+4)*n+4]=wheelbase*pcurvy4;

	/*-_x2*/grad[(9*i+5)*n]=-wheelbase*pcurvx2;
	/*-_x3*/grad[(9*i+5)*n+1]=-wheelbase*pcurvx3;
	/*-_y3*/grad[(9*i+5)*n+2]=-wheelbase*pcurvy3;
	/*-_x4*/grad[(9*i+5)*n+3]=-wheelbase*pcurvx4;
	/*-_y4*/grad[(9*i+5)*n+4]=-wheelbase*pcurvy4;
		}

		//Max change (+ & -) in curvature
		result[9*i+6]=wheelbase*curv_dot/(1+pow(wheelbase*curv,2))-max_ddelta*t_end;
		result[9*i+7]=-wheelbase*curv_dot/(1+pow(wheelbase*curv,2))-max_ddelta*t_end;

		if(grad){
	/*+_x2*/grad[(9*i+6)*n]=wheelbase/(1+pow(wheelbase*curv,2))*pdcurvx2-2*pow(wheelbase,3)*curv*curv_dot/pow(1+pow(wheelbase*curv,2),2)*pcurvx2;
	/*+_x3*/grad[(9*i+6)*n+1]=wheelbase/(1+pow(wheelbase*curv,2))*pdcurvx3-2*pow(wheelbase,3)*curv*curv_dot/pow(1+pow(wheelbase*curv,2),2)*pcurvx3;
	/*+_y3*/grad[(9*i+6)*n+2]=wheelbase/(1+pow(wheelbase*curv,2))*pdcurvy3-2*pow(wheelbase,3)*curv*curv_dot/pow(1+pow(wheelbase*curv,2),2)*pcurvy3;
	/*+_x4*/grad[(9*i+6)*n+3]=wheelbase/(1+pow(wheelbase*curv,2))*pdcurvx4-2*pow(wheelbase,3)*curv*curv_dot/pow(1+pow(wheelbase*curv,2),2)*pcurvx4;
	/*+_y4*/grad[(9*i+6)*n+4]=wheelbase/(1+pow(wheelbase*curv,2))*pdcurvy4-2*pow(wheelbase,3)*curv*curv_dot/pow(1+pow(wheelbase*curv,2),2)*pcurvy4;

	/*-_x2*/grad[(9*i+7)*n]=-wheelbase/(1+pow(wheelbase*curv,2))*pdcurvx2+2*pow(wheelbase,3)*curv*curv_dot/pow(1+pow(wheelbase*curv,2),2)*pcurvx2;
	/*-_x3*/grad[(9*i+7)*n+1]=-wheelbase/(1+pow(wheelbase*curv,2))*pdcurvx3+2*pow(wheelbase,3)*curv*curv_dot/pow(1+pow(wheelbase*curv,2),2)*pcurvx3;
	/*-_y3*/grad[(9*i+7)*n+2]=-wheelbase/(1+pow(wheelbase*curv,2))*pdcurvy3+2*pow(wheelbase,3)*curv*curv_dot/pow(1+pow(wheelbase*curv,2),2)*pcurvy3;
	/*-_x4*/grad[(9*i+7)*n+3]=-wheelbase/(1+pow(wheelbase*curv,2))*pdcurvx4+2*pow(wheelbase,3)*curv*curv_dot/pow(1+pow(wheelbase*curv,2),2)*pcurvx4;
	/*-_y4*/grad[(9*i+7)*n+4]=-wheelbase/(1+pow(wheelbase*curv,2))*pdcurvy4+2*pow(wheelbase,3)*curv*curv_dot/pow(1+pow(wheelbase*curv,2),2)*pcurvy4;
		}

		//Smooth minimum obstacle distance
		for(int j=0;j<cols-7;j++){
			double dist1= pow(pow(bez_curv[i][0]-xopt[0][j+7],2)+pow(bez_curv[i][1]-xopt[1][j+7],2),0.5); //Euclidean distance
			result[9*i+8]=result[9*i+8]+exp(-1.0*double(bez_beta)*dist1);
			if(result[9*i+8]==0){
				printf("ZERO %d, %d, %e, %lf, %lf, %lf, %lf\n",i,j,result[9*i+8], bez_curv[i][0],xopt[0][j+7],bez_curv[i][1],xopt[1][j+7]);
			}

			if(grad){
		/*x2*/	grad[(9*i+8)*n]=grad[(9*i+8)*n]+exp(-1*double(bez_beta)*dist1)*(bez_curv[i][0]-xopt[0][j+7])/dist1*px2;
		/*x3*/	grad[(9*i+8)*n+1]=grad[(9*i+8)*n+1]+exp(-1*double(bez_beta)*dist1)*(bez_curv[i][0]-xopt[0][j+7])/dist1*px3;
		/*y3*/	grad[(9*i+8)*n+2]=grad[(9*i+8)*n+2]+exp(-1*double(bez_beta)*dist1)*(bez_curv[i][1]-xopt[1][j+7])/dist1*py3;
		/*x4*/	grad[(9*i+8)*n+3]=grad[(9*i+8)*n+3]+exp(-1*double(bez_beta)*dist1)*(bez_curv[i][0]-xopt[0][j+7])/dist1*px4;
		/*y4*/	grad[(9*i+8)*n+4]=grad[(9*i+8)*n+4]+exp(-1*double(bez_beta)*dist1)*(bez_curv[i][1]-xopt[1][j+7])/dist1*py4;
			}
		}
		if(grad){
			double myval=grad[(9*i+8)*n];
			grad[(9*i+8)*n]=-grad[(9*i+8)*n]/result[9*i+8];
			grad[(9*i+8)*n+1]=-grad[(9*i+8)*n+1]/result[9*i+8];
			grad[(9*i+8)*n+2]=-grad[(9*i+8)*n+2]/result[9*i+8];
			grad[(9*i+8)*n+3]=-grad[(9*i+8)*n+3]/result[9*i+8];
			grad[(9*i+8)*n+4]=-grad[(9*i+8)*n+4]/result[9*i+8];
			if(nanflag==0 && std::isnan(grad[(9*i+8)*n])){
				nanflag=1;
				printf("NAN %e, %e\n",myval,result[9*i+8]);
			}
		}

		double myval=result[9*i+8];
		result[9*i+8]=1.0/double(bez_beta)*log(result[9*i+8])+bez_min_dist;
		if(std::isnan(result[9*i+8])){
			printf("NAN %lf\n",myval);
		}
		
	}

	for(int i=0;i<=(bez_curv_pts-1)*9+8;i++){
		if(std::isnan(result[i])){
			printf("Result nan %d\n",i);
		}
	}
	if(grad){
		for(int i=0;i<=(9*(bez_curv_pts-1)+8)*n+4;i++){
			if(std::isnan(grad[i])){
				printf("Grad nan %d\n",i);
			}
		}
	}


	//Manual checking of analytical gradients
	// std::vector<std::vector<double>> bez_curv1;
	// double hval=1e-8;
	// //Create the discretized Bezier Curve
	// for(int i=0; i<bez_curv_pts; i++){
	// 	double t=double(i)/double(bez_curv_pts-1);
	// 	double bez_x=4*pow(1-t,3)*t*x1+6*pow(1-t,2)*pow(t,2)*x[0]+4*(1-t)*pow(t,3)*x[1]+pow(t,4)*(hval+x[3]);
	// 	double bez_y=6*pow(1-t,2)*pow(t,2)*y2+4*(1-t)*pow(t,3)*x[2]+pow(t,4)*x[4]; //y1=0
	// 	bez_curv1.push_back({bez_x,bez_y});
	// }
	// double result1[9*(bez_curv_pts-1)+9];
	// if(result1){
	// 	for(int i=0;i<=(bez_curv_pts-1)*9+8;i++){
	// 		result1[i]=0;
	// 	}
	// }
	

	// if(grad){
	// 	for(int i=0;i<bez_curv_pts; i++){
	// 		double t=double(i)/double(bez_curv_pts-1);
	// 		double x_dot=4*x1*(-4*pow(t,3)+9*pow(t,2)-6*t+1)+6*x[0]*(4*pow(t,3)-6*pow(t,2)+2*t)+4*x[1]*(-4*pow(t,3)+3*pow(t,2))+4*(hval+x[3])*pow(t,3);
	// 		double x_ddot=4*x1*(-12*pow(t,2)+18*t-6)+6*x[0]*(12*pow(t,2)-12*t+2)+4*x[1]*(-12*pow(t,2)+6*t)+12*(hval+x[3])*pow(t,2);
	// 		double x_dddot=4*x1*(-24*t+18)+6*x[0]*(24*t-12)+4*x[1]*(-24*t+6)+24*(hval+x[3])*t;

	// 		double y_dot=6*y2*(4*pow(t,3)-6*pow(t,2)+2*t)+4*x[2]*(-4*pow(t,3)+3*pow(t,2))+4*x[4]*pow(t,3);
	// 		double y_ddot=6*y2*(12*pow(t,2)-12*t+2)+4*x[2]*(-12*pow(t,2)+6*t)+12*x[4]*pow(t,2);
	// 		double y_dddot=6*y2*(24*t-12)+4*x[2]*(-24*t+6)+24*x[4]*t;

	// 		double curv=(x_dot*y_ddot-y_dot*x_ddot)/(pow(pow(x_dot,2)+pow(y_dot,2),1.5));
	// 		double curv_dot=((x_dot*y_dddot-y_dot*x_dddot)*(pow(x_dot,2)+pow(y_dot,2))-3*(x_dot*x_ddot+y_dot*y_ddot)*(x_dot*y_ddot-y_dot*x_ddot))/pow((pow(x_dot,2)+pow(y_dot,2)),2.5);

	// 		double px2=6*pow(1-t,2)*pow(t,2);double px3=4*(1-t)*pow(t,3);double py3=px3;double px4=pow(t,4);double py4=px4; //Partials of original x & y
	// 		double pdx2=6*(4*pow(t,3)-6*pow(t,2)+2*t);double pdx3=4*(-4*pow(t,3)+3*pow(t,2));double pdy3=pdx3;double pdx4=4*pow(t,3);double pdy4=pdx4; //X_dot & y_dot
	// 		double pddx2=6*(12*pow(t,2)-12*t+2);double pddx3=4*(-12*pow(t,2)+6*t);double pddy3=pddx3;double pddx4=12*pow(t,2);double pddy4=pddx4; //x_ddot & y_ddot
	// 		double pdddx2=6*(24*t-12);double pdddx3=4*(-24*t+6);double pdddy3=pdddx3;double pdddx4=24*t;double pdddy4=pdddx4; //x_dddot & y_dddot
			
	// 		double pcurvx2=((y_ddot*pdx2-y_dot*pddx2)*(pow(x_dot,2)+pow(y_dot,2))-3*x_dot*pdx2*(x_dot*y_ddot-y_dot*x_ddot))/pow(pow(x_dot,2)+pow(y_dot,2),2.5);
	// 		double pcurvx3=((y_ddot*pdx3-y_dot*pddx3)*(pow(x_dot,2)+pow(y_dot,2))-3*x_dot*pdx3*(x_dot*y_ddot-y_dot*x_ddot))/pow(pow(x_dot,2)+pow(y_dot,2),2.5);
	// 		double pcurvy3=((x_dot*pddy3-x_ddot*pdy3)*(pow(x_dot,2)+pow(y_dot,2))-3*y_dot*pdy3*(x_dot*y_ddot-y_dot*x_ddot))/pow(pow(x_dot,2)+pow(y_dot,2),2.5);
	// 		double pcurvx4=((y_ddot*pdx4-y_dot*pddx4)*(pow(x_dot,2)+pow(y_dot,2))-3*x_dot*pdx4*(x_dot*y_ddot-y_dot*x_ddot))/pow(pow(x_dot,2)+pow(y_dot,2),2.5);
	// 		double pcurvy4=((x_dot*pddy4-x_ddot*pdy4)*(pow(x_dot,2)+pow(y_dot,2))-3*y_dot*pdy4*(x_dot*y_ddot-y_dot*x_ddot))/pow(pow(x_dot,2)+pow(y_dot,2),2.5);
			
	// 		double num_pdcurvx2=(y_dddot*pdx2-y_dot*pdddx2)*(pow(x_dot,2)+pow(y_dot,2))+2*x_dot*pdx2*(x_dot*y_dddot-y_dot*x_dddot)-3*((x_dot*pddx2+x_ddot*pdx2)*(x_dot*y_ddot-y_dot*x_ddot)+(y_ddot*pdx2-y_dot*pddx2)*(x_dot*x_ddot+y_dot*y_ddot));
	// 		double num_pdcurvx3=(y_dddot*pdx3-y_dot*pdddx3)*(pow(x_dot,2)+pow(y_dot,2))+2*x_dot*pdx3*(x_dot*y_dddot-y_dot*x_dddot)-3*((x_dot*pddx3+x_ddot*pdx3)*(x_dot*y_ddot-y_dot*x_ddot)+(y_ddot*pdx3-y_dot*pddx3)*(x_dot*x_ddot+y_dot*y_ddot));
	// 		double num_pdcurvy3=(x_dot*pdddy3-x_dddot*pdy3)*(pow(x_dot,2)+pow(y_dot,2))+2*y_dot*pdy3*(x_dot*y_dddot-y_dot*x_dddot)-3*((y_dot*pddy3+y_ddot*pdy3)*(x_dot*y_ddot-y_dot*x_ddot)+(x_dot*pddy3-x_ddot*pdy3)*(x_dot*x_ddot+y_dot*y_ddot));
	// 		double num_pdcurvx4=(y_dddot*pdx4-y_dot*pdddx4)*(pow(x_dot,2)+pow(y_dot,2))+2*x_dot*pdx4*(x_dot*y_dddot-y_dot*x_dddot)-3*((x_dot*pddx4+x_ddot*pdx4)*(x_dot*y_ddot-y_dot*x_ddot)+(y_ddot*pdx4-y_dot*pddx4)*(x_dot*x_ddot+y_dot*y_ddot));
	// 		double num_pdcurvy4=(x_dot*pdddy4-x_dddot*pdy4)*(pow(x_dot,2)+pow(y_dot,2))+2*y_dot*pdy4*(x_dot*y_dddot-y_dot*x_dddot)-3*((y_dot*pddy4+y_ddot*pdy4)*(x_dot*y_ddot-y_dot*x_ddot)+(x_dot*pddy4-x_ddot*pdy4)*(x_dot*x_ddot+y_dot*y_ddot));

	// 		double pdcurvx2=(num_pdcurvx2*pow(pow(x_dot,2)+pow(y_dot,2),2.5)-((x_dot*y_dddot-y_dot*x_dddot)*(pow(x_dot,2)+pow(y_dot,2))-3*(x_dot*x_ddot+y_dot*y_ddot)*(x_dot*y_ddot-y_dot*x_ddot))*5*pow(pow(x_dot,2)+pow(y_dot,2),1.5)*x_dot*pdx2)/pow(pow(x_dot,2)+pow(y_dot,2),5);
	// 		double pdcurvx3=(num_pdcurvx3*pow(pow(x_dot,2)+pow(y_dot,2),2.5)-((x_dot*y_dddot-y_dot*x_dddot)*(pow(x_dot,2)+pow(y_dot,2))-3*(x_dot*x_ddot+y_dot*y_ddot)*(x_dot*y_ddot-y_dot*x_ddot))*5*pow(pow(x_dot,2)+pow(y_dot,2),1.5)*x_dot*pdx3)/pow(pow(x_dot,2)+pow(y_dot,2),5);
	// 		double pdcurvy3=(num_pdcurvy3*pow(pow(x_dot,2)+pow(y_dot,2),2.5)-((x_dot*y_dddot-y_dot*x_dddot)*(pow(x_dot,2)+pow(y_dot,2))-3*(x_dot*x_ddot+y_dot*y_ddot)*(x_dot*y_ddot-y_dot*x_ddot))*5*pow(pow(x_dot,2)+pow(y_dot,2),1.5)*y_dot*pdy3)/pow(pow(x_dot,2)+pow(y_dot,2),5);
	// 		double pdcurvx4=(num_pdcurvx4*pow(pow(x_dot,2)+pow(y_dot,2),2.5)-((x_dot*y_dddot-y_dot*x_dddot)*(pow(x_dot,2)+pow(y_dot,2))-3*(x_dot*x_ddot+y_dot*y_ddot)*(x_dot*y_ddot-y_dot*x_ddot))*5*pow(pow(x_dot,2)+pow(y_dot,2),1.5)*x_dot*pdx4)/pow(pow(x_dot,2)+pow(y_dot,2),5);
	// 		double pdcurvy4=(num_pdcurvy4*pow(pow(x_dot,2)+pow(y_dot,2),2.5)-((x_dot*y_dddot-y_dot*x_dddot)*(pow(x_dot,2)+pow(y_dot,2))-3*(x_dot*x_ddot+y_dot*y_ddot)*(x_dot*y_ddot-y_dot*x_ddot))*5*pow(pow(x_dot,2)+pow(y_dot,2),1.5)*y_dot*pdy4)/pow(pow(x_dot,2)+pow(y_dot,2),5);


	// 		//Max and min velocity
	// 		result1[9*i]=pow(x_dot,2)+pow(y_dot,2)-pow(max_v,2)*pow(t_end,2);
	// 		printf("%d, %d, Grad: %lf, Diff: %lf\n",9*i,i,grad[(9*i)*n+3],(result1[9*i]-result[9*i])/hval);
	// 		result1[9*i+1]=-pow(x_dot,2)-pow(y_dot,2)+pow(min_v,2)*pow(t_end,2);
	// 		printf("%d, %d, Grad: %lf, Diff: %lf\n",9*i+1,i,grad[(9*i+1)*n+3],(result1[9*i+1]-result[9*i+1])/hval);

	// 		//Max change (+ & -) in velocity
	// 		result1[9*i+2]=(pow(x_dot*x_ddot+y_dot*y_ddot,2)/(pow(x_dot,2)+pow(y_dot,2)))-pow(max_dv,2)*pow(t_end,4);
	// 		printf("%d, %d, Grad: %lf, Diff: %lf\n",9*i+2,i,grad[(9*i+2)*n+3],(result1[9*i+2]-result[9*i+2])/hval);
	// 		result1[9*i+3]=-(pow(x_dot*x_ddot+y_dot*y_ddot,2)/(pow(x_dot,2)+pow(y_dot,2)))-pow(max_dv,2)*pow(t_end,4);
	// 		printf("%d, %d, Grad: %lf, Diff: %lf\n",9*i+3,i,grad[(9*i+3)*n+3],(result1[9*i+3]-result[9*i+3])/hval);

	// 		//Max (+ & -) curvature
	// 		result1[9*i+4]=wheelbase*curv-tan(max_delta);
	// 		printf("%d, %d, Grad: %lf, Diff: %lf\n",9*i+4,i,grad[(9*i+4)*n+3],(result1[9*i+4]-result[9*i+4])/hval);
	// 		result1[9*i+5]=-wheelbase*curv-tan(max_delta);
	// 		printf("%d, %d, Grad: %lf, Diff: %lf\n",9*i+5,i,grad[(9*i+5)*n+3],(result1[9*i+5]-result[9*i+5])/hval);

	// 		//Max change (+ & -) in curvature
	// 		result1[9*i+6]=wheelbase*curv_dot/(1+pow(wheelbase*curv,2))-max_ddelta*t_end;
	// 		printf("%d, %d, Grad: %lf, Diff: %lf\n",9*i+6,i,grad[(9*i+6)*n+3],(result1[9*i+6]-result[9*i+6])/hval);
	// 		result1[9*i+7]=-wheelbase*curv_dot/(1+pow(wheelbase*curv,2))-max_ddelta*t_end;
	// 		printf("%d, %d, Grad: %lf, Diff: %lf\n",9*i+7,i,grad[(9*i+7)*n+3],(result1[9*i+7]-result[9*i+7])/hval);

	// 		//Smooth minimum obstacle distance
	// 		for(int j=0;j<cols-7;j++){
	// 			double dist1= pow(pow(bez_curv1[i][0]-xopt[0][j+7],2)+pow(bez_curv1[i][1]-xopt[1][j+7],2),0.5); //Euclidean distance
	// 			result1[9*i+8]=result1[9*i+8]+exp(-1*double(bez_beta)*dist1);

	// 		}
			

	// 		result1[9*i+8]=1.0/double(bez_beta)*log(result1[9*i+8])+bez_min_dist;
	// 		printf("%d, %d, Grad: %lf, Diff: %lf\n",9*i+8,i,grad[(9*i+8)*n+3],(result1[9*i+8]-result[9*i+8])/hval);
			
	// 	}
	// }

}


void pursuit_inequality_con(unsigned m, double *result, unsigned n, const double* x, double* grad, void* my_func_data){ //Minimum leader distance inequality
	double* raw_data = static_cast<double*>(my_func_data); //First extract as 1D array to get the column count (first value passed)
	int cols = 7; //Constant amount of variables passed, only using leader, pursuit trajectories, no obstacles
	double xopt[2][cols]; //2D array of tracking lines appended to some certain variables
	for (int i = 0; i < cols; i++) {
		xopt[0][i] = raw_data[2*i];
		xopt[1][i] = raw_data[2*i + 1];
	    }

	double pursue_x=xopt[0][0]; //Initial leader x pos, brought back by the distance we want to track behind
	double pursue_y=xopt[1][0]; //Initial leader y pos, brought back by the distance we want to track behind
	double lead_theta=xopt[0][1]; //Leader's initial theta
	double lead_vel=xopt[1][1]; //Leader's constant velocity over traj
	double radius=xopt[0][2]; //Radius of circle arc of leader, + or - to account for delta + or -
	double pursuit_weight=xopt[1][2]; //Weighting of pursuit term vs middle line MPC
	int bez_curv_pts=static_cast<int>(xopt[0][3]); //Whether leader is detected or not
	double bez_t_end=xopt[1][3]; //Time of the pursuit trajectory, used to find the bez_curv_pts for the arc length of the leader's trajectory, with known vel_lead
	double min_pursue=xopt[0][4]; //Minimum distance allowed between pursuit and leader vehicle
	double veh_det_length=xopt[1][4]; //Leader vehicle length, for constructing box around center point of vehicle
	double veh_det_width=xopt[0][5]; //Leader vehicle width, for constructing box around center point of vehicle
	double bez_beta=xopt[1][5]; //Soft min smoothing constant
	double x1=xopt[0][6]; //Fixed x1 point for bezier control points
	double y2=xopt[1][6]; //Fixed y2 point for bezier control points


	if(result){
		for(int i=0;i<m;i++){
			result[i]=0;
		}
	}

	if(grad){
		for(int i=0;i<n*m;i++){
			grad[i]=0;
		}
	}

	//THIS PART IS BASED ON VELOCITY_OURSUIT, NEEDS TO BE FIXED AND UPDATED FOR BEZIER IMPLEMENTATION
	//*********************************************************************************************

	//Find the distances between the sample bezier pts of our pursuit, leader and find gradients with partial of x,y wrt control points as above

	//Find softmin dist between follower, leader vehicles over all time steps in the trajectory & for the four corners of the vehicle box
	//Each point along trajectory has four points, the gradient wrt x_i & y_i only involves numerator terms at that time step. Others wrt x_i, y_i = 0 so ignore
	double sum_dist=0;
	double curdist=0;
	for(int i=0;i<bez_curv_pts;i++){ //Just one, big softmin distance constraint
		double t=double(i)/double(bez_curv_pts);
		double px2=6*pow(1-t,2)*pow(t,2); double px3=4*(1-t)*pow(t,3); double py3=px3; double px4=pow(t,4); double py4=px4; //Partials of bezier x,y wrt ctrl pts

		double bez_x=4*pow(1-t,3)*t*x1+6*pow(1-t,2)*pow(t,2)*x[0]+4*(1-t)*pow(t,3)*x[1]+pow(t,4)*x[3];
		double bez_y=6*pow(1-t,2)*pow(t,2)*y2+4*(1-t)*pow(t,3)*x[2]+pow(t,4)*x[4]; //y1=0
		
		double xl_og=radius*sin(lead_vel*bez_t_end*t/radius);
		double yl_og=radius*(1-cos(lead_vel*bez_t_end*t/radius)); //xlead, ylead before accoutning for theta rot
		double xv_og=lead_vel*cos(lead_vel*bez_t_end*t/radius);
		double yv_og=lead_vel*sin(lead_vel*bez_t_end*t/radius); //x_vel, y_vel components before accounting for theta rot

		double x_lead=pursue_x+xl_og*cos(lead_theta)-yl_og*sin(lead_theta);
		double y_lead=pursue_y+xl_og*sin(lead_theta)+yl_og*cos(lead_theta); //future leader pos in base_link frame (wrt our pursuit vehicle)
		double x_lead_d=xv_og*cos(lead_theta)-yv_og*sin(lead_theta);
		double y_lead_d=xv_og*sin(lead_theta)+yv_og*cos(lead_theta); //future leader vel in base_link fram (wrt our pursuit vehicle)
		
		double thet_lead=atan2(y_lead_d,x_lead_d); //Can use x_vel, y_vel to get the theta and thus the orientation of the vehicle for the four box corners
		double lead_ptsx[4]; double lead_ptsy[4];
		lead_ptsx[0]=x_lead+veh_det_length/2*cos(thet_lead)+veh_det_width/2*sin(thet_lead); lead_ptsy[0]=y_lead+veh_det_length/2*sin(thet_lead)-veh_det_width/2*cos(thet_lead);
		lead_ptsx[1]=x_lead+veh_det_length/2*cos(thet_lead)-veh_det_width/2*sin(thet_lead); lead_ptsy[1]=y_lead+veh_det_length/2*sin(thet_lead)+veh_det_width/2*cos(thet_lead);
		lead_ptsx[2]=x_lead-veh_det_length/2*cos(thet_lead)-veh_det_width/2*sin(thet_lead); lead_ptsy[2]=y_lead-veh_det_length/2*sin(thet_lead)+veh_det_width/2*cos(thet_lead);
		lead_ptsx[3]=x_lead-veh_det_length/2*cos(thet_lead)+veh_det_width/2*sin(thet_lead); lead_ptsy[3]=y_lead-veh_det_length/2*sin(thet_lead)-veh_det_width/2*cos(thet_lead);

		//Sum distance for this point, find the numerator at least for this gradient wrt x_i & y_i
		for(int j=0;j<4;j++){
			curdist=sqrt(pow(bez_x-lead_ptsx[j],2)+pow(bez_y-lead_ptsy[j],2));
			sum_dist+=exp(-vel_beta*curdist);
			grad[0]=exp(-vel_beta*curdist)*(bez_x-lead_ptsx[j])/curdist*px2; //x2
			grad[1]=exp(-vel_beta*curdist)*(bez_x-lead_ptsx[j])/curdist*px3; //x3
			grad[2]=exp(-vel_beta*curdist)*(bez_y-lead_ptsy[j])/curdist*py3; //y3
			grad[3]=exp(-vel_beta*curdist)*(bez_x-lead_ptsx[j])/curdist*px4; //x4
			grad[4]=exp(-vel_beta*curdist)*(bez_y-lead_ptsy[j])/curdist*py4; //y4
		}	
	}
	for(int i=0;i<nMPC*kMPC;i++){
		grad[0]=-grad[0]/sum_dist; //Divide numerator by summed denominator to get the appropriate grad for x_i, y_i
		grad[1]=-grad[1]/sum_dist;
		grad[2]=-grad[2]/sum_dist;
		grad[3]=-grad[3]/sum_dist;
		grad[4]=-grad[4]/sum_dist;
	}
	result[0]=1.0/double(vel_beta)*log(sum_dist)+min_pursue; //Final softmin for the distances along trajectory, either violates or doesn't vs min_pursue

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
		ros::Publisher bez_mark;
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
		double safe_distance_adapt;

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
		visualization_msgs::Marker bez;
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
		double max_speed=0;
		double min_speed=0;
		double max_accel=0; //Max decel is set to equal

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

		//Leader-follower MPC pursuit params
		double pursuit_weight=0;
		int leader_detect=0;
		double MPC_dist=0;
		double pursuit_dist=0;
		double transit_rate=0;
		double min_pursue=0;
		double min_delta=0;
		double pursuit_x=0;
		double pursuit_y=0;

		double speed_to_erpm_gain, speed_to_erpm_offset;
		double steering_angle_to_servo_gain, steering_angle_to_servo_offset;
		std_msgs::Float64 last_servo_state;
		double vel_adapt=1;

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

		Eigen::Matrix<double, 2, 5> meas_observability; //Measurement observability matrix, used in KF & same for all

		double lastx=0, lasty=0, lasttheta=0;

		int use_neural_net=0; //Whether one of the neural networks is being used for vehicle detection

		double veh_det_length=0;
		double veh_det_width=0;

		//Bezier MPC Parameters
		int bez_ctrl_pts=0;
		int bez_curv_pts=0;
		double bez_alpha=0;
		int bez_beta=0;
		double bez_min_dist=0;
		double bez_t_end=0;
		double obs_sep=0;
		double max_obs=0;

		ros::Time timestamp_tf1; ros::Time timestamp_tf2;
		ros::Time timestamp_cam1; ros::Time timestamp_cam2;

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
			safe_distance_adapt=safe_distance;

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
			nf.getParam("max_speed",max_speed);
			nf.getParam("min_speed",min_speed);
			nf.getParam("max_accel",max_accel); //Max decel set to equal

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

			//Bezier MPC Parameters
			nf.getParam("bez_ctrl_pts", bez_ctrl_pts);
			nf.getParam("bez_curv_pts", bez_curv_pts);
			nf.getParam("bez_alpha", bez_alpha);
			nf.getParam("bez_beta", bez_beta);
			nf.getParam("bez_min_dist", bez_min_dist);
			nf.getParam("bez_t_end", bez_t_end);
			nf.getParam("obs_sep", obs_sep);
			nf.getParam("max_obs", max_obs);

			//Leader-follower MPC pursuit
			nf.getParam("MPC_dist", MPC_dist);
			nf.getParam("pursuit_dist", pursuit_dist);
			nf.getParam("transit_rate", transit_rate);
			nf.getParam("min_pursue", min_pursue);
			nf.getParam("min_delta", min_delta);
			nf.getParam("pursuit_x", pursuit_x);
			nf.getParam("pursuit_y", pursuit_y);

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
			

			//publications
			//lidar_pub = nf.advertise<std_msgs::Int32MultiArray>("chatter", 1000);
			marker_pub = nf.advertise<visualization_msgs::Marker>("wall_markers",2);
			mpc_marker_pub = nf.advertise<visualization_msgs::Marker>("mpc_markers",2);
			wall_marker_pub=nf.advertise<visualization_msgs::Marker>("walls",2);
			lobs=nf.advertise<visualization_msgs::Marker>("lobs",2);
			robs=nf.advertise<visualization_msgs::Marker>("robs",2);
			bez_mark=nf.advertise<visualization_msgs::Marker>("bez",2);
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
					double robx=transform.transform.translation.x;
					double roby=transform.transform.translation.y;
					// 		transform.transform.translation.z);
					double x=transform.transform.rotation.x;
					double y=transform.transform.rotation.y;
					double z=transform.transform.rotation.z;
					double w=transform.transform.rotation.w;
					double robtheta = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));

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
					simx=transform.transform.translation.x;
					simy=transform.transform.translation.y;
					// 		transform.transform.translation.z);
					double x=transform.transform.rotation.x;
					double y=transform.transform.rotation.y;
					double z=transform.transform.rotation.z;
					double w=transform.transform.rotation.w;
					simtheta = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
					// printf("%lf, %lf, %lf, %lf\n",simx,simy,simtheta,ros::Time::now().toSec());
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
			// printf("YOLO: %lf, %lf\n",ros::Time::now().toSec(),cv_image_data.header.stamp.toSec());
			

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

					printf("%d, %lf, %lf (%lf, %lf, %lf)\n",i,xorigin,yorigin, odomx,odomy,odomtheta);

				}
			}

			//For all existing detections, provide the x and y of the depth measurement
			for (int q=0; q<car_detects.size();q++){
				// printf("ID: %d, yolo: %d\n",q,temp_vec[q]);
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
				// printf("ID: %d, yolo: %d NEW\n",q,no_det[q]);
				int i=no_det[q];
				if(yolo_xy[i][0]!=0 && yolo_xy[i][1]!=0){
					vehicle_detection new_det;
					new_det.bound_box={yolo_msg.rectangles[4*i],yolo_msg.rectangles[4*i+1],yolo_msg.rectangles[4*i+2],yolo_msg.rectangles[4*i+3]}; //ymin, xmin, ymax, xmax
					new_det.meas={yolo_xy[i][0],yolo_xy[i][1]};
					new_det.last_det=1;
					new_det.meas_tf=my_tf;
					new_det.cov_P(0,0)=0.01; new_det.cov_P(1,1)=0.01; new_det.cov_P(2,2)=std::pow(5 * M_PI / 180, 2);
					new_det.cov_P(3,3)=2; new_det.cov_P(4,4)=std::pow(5 * M_PI / 180, 2);
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
					// printf("DEPPIXEL: %lf, %lf COLPIXEL: %lf, %lf\n",depth_pixel[0],depth_pixel[1],color_pixel[0],color_pixel[1]);
					if(color_pixel[0]<=x_true) depth_pixel[0]+=std::max(1,(int)((x_true-color_pixel[0])/2));
					if(color_pixel[1]<=y_true) depth_pixel[1]+=std::max(1,(int)((y_true-color_pixel[1])/2));
						
				}
				else{
					// printf("0, increment %lf, %lf\n",depth_pixel[0],depth_pixel[1]);
					if(color_pixel[0]<=x_true) depth_pixel[0]++;
					if(color_pixel[1]<=y_true) depth_pixel[1]++;
				}
			}
			// printf("Depth of corner: %lf\n",(cv_image1.ptr<uint16_t>((int)depth_pixel[1])[(int)depth_pixel[0]])/(float)1000);
			// printf("\n");
			// return;
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
				// printf("x: %lf, y: %lf, depth: %lf\n",av_x,av_y,av_depth);
				det_xy[0]=av_x; det_xy[1]=av_y;
			}
			// else printf("No points found, can't update measurement\n");

			return det_xy;

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

		void servo_callback(const std_msgs::Float64 & servo){
			last_servo_state=servo;
		}

		void vesc_callback(const vesc_msgs::VescStateStamped & state){
        	vel_adapt = std::max(-( state.state.speed - speed_to_erpm_offset ) / speed_to_erpm_gain,0.1);
			last_delta = ( last_servo_state.data - steering_angle_to_servo_offset) / steering_angle_to_servo_gain;
			// vel_adapt= vehicle_velocity; //Remove this
		}

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
			
			ros::Time t19 = ros::Time::now();
			// printf("%lf\n",t19.toSec());
			timestamp_cam1=img->header.stamp;
			// printf("DEPTH: %lf, %lf\n",t19.toSec(),timestamp_cam1.toSec());
			// ROS_INFO("https://github.com/IntelRealSense/librealsense/issues/2455\n"); //google search: aligned depth to color frequency
			//or this: https://github.com/IntelRealSense/librealsense/issues/5583
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
				cv_image_data.data=std::vector<uint8_t>(img->data.begin(), img->data.end());
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

			float3 plane_point={ 0 , 0 , 1.0/groundplane.C*-groundplane.D };

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
				if(ranges[ls_str+i] <= safe_distance_adapt) {data[i][0] = 0; data[i][1] = i*ls_ang_inc-angle_cen;}
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
			double safe_dist=safe_distance_adapt;
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
					// printf("Forecast #%d: %lf, %lf\n",traj,x_det,y_det);
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

			//TODO: ADD TIME STUFF HERE
			ros::Time ttt = ros::Time::now();
			current_time = ttt.toSec();
			// printf("LIDAR: %lf\n",current_time);
			double dt = current_time - prev_time;
			if(dt>1){
				dt=default_dt;
			}
			prev_time = current_time;

			leader_detect=0;


			//Vehicle tracking even if we aren't currently driving
			
			// std::cout << "Matrix 1:" << std::endl << mat1 << std::endl; //use eigen for Kalman Filter calculations, there may be conflict errors
			//for example inverse if needed conflicts with QuadProg: under QuadProg.h include but above Eigen, have line:
			//#undef inverse // Remove the conflicting macro
			for (int q=car_detects.size()-1; q>=0;q--){ //Iterate backwards to handle deletions
				// printf("%d\n",car_detects[q].miss_fr);
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

					// printf("Last KF state (og frame): %lf, %lf\n",car_detects[q].state[0],car_detects[q].state[1]);
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

					// printf("Last KF state (new frame): %lf, %lf\n",tempx,tempy);
					// printf("Meas (og) %lf, %lf TF'd -> %lf, %lf\n",car_detects[q].meas[0],car_detects[q].meas[1],curmeasx,curmeasy);
					// printf("Last KF tf: %lf, %lf, %lf\n", lastx, lasty, lasttheta);
					// printf("Measure TF: %lf, %lf, %lf\n",car_detects[q].meas_tf.tf_x,car_detects[q].meas_tf.tf_y,car_detects[q].meas_tf.tf_theta);
					// printf("New KF (live) tf: %lf, %lf, %lf\n",odomx,odomy,odomtheta);
					// printf("Cur Time: %lf, Closest TF to Meas Time: %lf\n",ros::Time::now().toSec(),car_detects[q].meas_tf.tf_time);

					// printf("%d: %lf, %lf, %lf, %lf, %lf\n\n",q,car_detects[q].state[0],car_detects[q].state[1],car_detects[q].state[2],car_detects[q].state[3],car_detects[q].state[4]);
					
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
					car_detects[q].proc_noise(0,0)+=0.01; car_detects[q].proc_noise(1,1)+=0.01; //Increase the model's uncertainty via process noise
					car_detects[q].proc_noise(2,2)+=std::pow(5 * M_PI / 180, 2); car_detects[q].proc_noise(3,3)+=0.01;
					car_detects[q].proc_noise(4,4)+=std::pow(5 * M_PI / 180, 2);
					car_detects[q].cov_P=state_transition_lin*car_detects[q].cov_P*state_transition_lin.transpose()+car_detects[q].proc_noise;
					continue;
				}

				//Process noise should depend on # of missed frames, speed of both our vehicle and the detected vehicle
				if(car_detects[q].miss_fr==0){
					car_detects[q].proc_noise(0,0)=0.01; car_detects[q].proc_noise(1,1)=0.01; car_detects[q].proc_noise(2,2)=std::pow(5 * M_PI / 180, 2);
					car_detects[q].proc_noise(3,3)=0.01; car_detects[q].proc_noise(4,4)=std::pow(5 * M_PI / 180, 2);
				}
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

				//Expand once base functionality with changing parameters depending on conditions
				//Also, different mechanism if no measurement for this cycle

				// std::cout << "Proc Noise:\n" << car_detects[q].proc_noise << std::endl;
				// std::cout << "Meas Noise:\n" << car_detects[q].meas_noise << std::endl;
				// printf("Measurement: %lf, %lf\n",car_detects[q].meas[0],car_detects[q].meas[1]);
				// std::cout << "State:\n" << car_detects[q].state << std::endl;
				// std::cout << "Cov:\n" << car_detects[q].cov_P << std::endl;
				// printf("Seems that once measurements get missed, can run into cases of states exploding\n");

				// printf("%d: %lf, %lf, %lf, %lf, %lf\n",q,car_detects[q].state[0],car_detects[q].state[1],car_detects[q].state[2],car_detects[q].state[3],car_detects[q].state[4]);
				if(car_detects[q].state[3]>0.3){
					std::cout << "State:\n" << car_detects[q].state << std::endl;
					std::cout << "Cov:\n" << car_detects[q].cov_P << std::endl;
					std::cout << "Pred State:\n" << pred_state << std::endl;
					std::cout << "Measure:\n" << meas_vec << std::endl;
					printf("%d\n",car_detects[q].miss_fr);
				}

				//Now, for the leader-follower MPC pursuit, determine if there is only one detection and it is being reasonably tracked
				if(leader_detect==0){
					if(car_detects[q].state[3]<3 && std::abs(car_detects[q].state[2])<M_PI/2){ //Reasonable velocity and oriented ahead, not pointing at us
						if(std::abs(car_detects[q].state[4])<max_steering_angle){ //Detected delta has to be reasonable, within physical limits of vehicle
							if(sqrt(pow(car_detects[q].state[0],2)+pow(car_detects[q].state[1],2))<max_lidar_range_opt){ //Within 5 m of us, more important for simulator environment
								if(tan(car_detects[q].state[4])/wheelbase*bez_t_end*car_detects[q].state[3]<2*M_PI){ //Doesn't complete full rotation of 2*PI over the future trajectory
									leader_detect=1; //Leader detected
								}
							}
						}
					}        
				}
				else{
					leader_detect=-1; //More than 1 vehicle detected, don't pursue
				}

				car_detects[q].last_det=0; //Reset the detection for next iteration, done at end to know in KF whether detected or not this cycle
			}

			lastx=odomx; lasty=odomy; lasttheta=odomtheta; //Keep our vehicle frame from last cycle to transform frame to new this cycle
			if(simx!=0){lastx=simx; lasty=simy; lasttheta=simtheta;}
			timestamp_tf2=timestamp_tf1; timestamp_cam2=timestamp_cam1;
			visualize_detections(); //PLot the detections in rviz regardless of if we are in autonomous mode or not
			// printf("**********************\n\n");

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


			
			

			int sec_len = int(heading_beam_angle/data->angle_increment);

			double min_distance, velocity_scale, delta_d;
			

			if(drive_state == "normal"){
				std::vector<float> fused_ranges_MPC=fused_ranges;
				std::vector<float> fused_ranges_MPC_tot0;
				std::vector<double> lidar_transform_angles_tot0;


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

				//Discard LIDAR scans close to the leader so that our LIDAR doesn't incorporate leader detections. These are handled separately
				if(leader_detect==1){
					for(int i=0; i<fused_ranges_MPC.size();i++){
						double lidx=fused_ranges_MPC[i]*cos(lidar_transform_angles[i]);
						double lidy=fused_ranges_MPC[i]*sin(lidar_transform_angles[i]);
						
						if(sqrt(pow(car_detects[0].state[0]-lidx,2)+pow(car_detects[0].state[1]-lidy,2))<veh_det_length){
							fused_ranges_MPC[i]=max_lidar_range_opt*2; //Set range very large, effectively ignore
						}
					}
				}

				for (int num_MPC=0;num_MPC<bez_ctrl_pts-3;num_MPC++){
					std::vector<float> fused_ranges_MPC_tot=fused_ranges_MPC;
					std::vector<double> lidar_transform_angles_tot=lidar_transform_angles; //The cumulative ranges and angles for both map (if used) and lidar

					if(use_map){ //Augment LIDAR with map obstacle points too
						std::vector<float> fused_ranges_MPC_map;
						std::vector<double> lidar_transform_angles_map; //These are the additional ranges & angles from fixed map that will be sorted, included in obs calculations
					
						double map_xval=mapped_x+cos(mapped_theta)*xpt-sin(mapped_theta)*ypt;
						double map_yval=mapped_y+sin(mapped_theta)*xpt+cos(mapped_theta)*ypt;
						for(int i=0;i<map_pts.size();i++){
							if(pow(map_pts[i][0]-map_xval,2)+pow(map_pts[i][1]-map_yval,2)<pow(max_lidar_range_opt,2)){
								double x_base=(map_pts[i][0]-locx)*cos(loctheta)-(map_pts[i][1]-locy)*sin(loctheta);
								double y_base=(map_pts[i][0]-locx)*sin(loctheta)+(map_pts[i][1]-locy)*cos(loctheta);
								double x_fut=(x_base-xpt)*cos(theta_ref)-(y_base-ypt)*sin(theta_ref);
								double y_fut=(x_base-xpt)*sin(theta_ref)+(y_base-ypt)*cos(theta_ref);
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

					if(leader_detect<1) leader_detect=0;
					
					if(use_neural_net && leader_detect==0){ //Augment LIDAR with detected vehicle projected paths as well, only if mutliple detects, not following
						std::vector<float> fused_ranges_MPC_veh_det;
						std::vector<double> lidar_transform_angles_veh_det; //These are the additional ranges & angles from vehicle detections that will be sorted, included in obs calculations
						int mult_factor=1; //This way, we get 2x amount of points for detections, improves the augmentation of LIDAR data

						for (int i=0; i<car_detects.size(); i++){
							if(car_detects[i].init==2){
								int start_track=0; int end_track=((nMPC-1)*kMPC+kMPC-1)*mult_factor; //Use all of the trajectpry in Bezier case
								double track_x=car_detects[i].state[0]; double track_y=car_detects[i].state[1]; double track_theta=car_detects[i].state[2];
								//Make a box for the vehicle based on orientation and have this projected path
								//This ensures that the vehicle is more prominent in LIDAR detections and is more of a box as opposed to the mid-point
								
								for(int j=0; j<nMPC*kMPC*mult_factor; j++){
									if(j>=start_track && j<=end_track){ //Only take this line segment timeframe of the MPC this round
										int num_border=5; //Number of points along each border of the box

										double tfed_x=cos(theta_ref)*(track_x-xpt)+sin(theta_ref)*(track_y-ypt);
										double tfed_y=-sin(theta_ref)*(track_x-xpt)+cos(theta_ref)*(track_y-ypt); //tf from vehicle frame to future MPC frame
										double tfed_ang=0;
										// double tfed_ang=atan2(tfed_y,tfed_x);
										// if (tfed_ang>M_PI) tfed_ang-=2*M_PI;
										// if (tfed_ang<-M_PI) tfed_ang+=2*M_PI;
								
										// lidar_transform_angles_veh_det.push_back(tfed_ang);
										// fused_ranges_MPC_veh_det.push_back(pow(pow(tfed_x,2)+pow(tfed_y,2),0.5));

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
												if (tfed_ang>M_PI) tfed_ang-=2*M_PI;
												if (tfed_ang<-M_PI) tfed_ang+=2*M_PI;
												lidar_transform_angles_veh_det.push_back(tfed_ang);
												fused_ranges_MPC_veh_det.push_back(pow(pow(edge_x,2)+pow(edge_y,2),0.5));
												// printf("%d, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n",num_MPC,theta_ref, xpt, ypt, track_x, track_y, track_theta, tfed_x, tfed_y);
												// printf(" %d, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n",num_MPC,top_right[0],top_right[1],top_left[0],top_left[1],bot_left[0],bot_left[1],bot_right[0],bot_right[1]);
												
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

					if(num_MPC==0){ //Take the original reference points for subsampling obstacles in NLOPT
						fused_ranges_MPC_tot0=fused_ranges_MPC_tot;
						lidar_transform_angles_tot0=lidar_transform_angles_tot;
					}

					std::vector<float> proc_ranges_MPC = preprocess_lidar_MPC(fused_ranges_MPC_tot,lidar_transform_angles_tot);
					
					int str_indx_MPC, end_indx_MPC; double heading_angle_MPC;
					heading_angle_MPC=find_missing_scan_gap_MPC(lidar_transform_angles_tot);
					heading_angle=heading_angle_MPC;
					
					if(heading_angle_MPC==5 || num_MPC==0){ //Use the other method to find the heading angle (if gap is large enough, use this prior value)
						
						std::pair<int,int> max_gap_MPC = find_max_gap_MPC(proc_ranges_MPC,lidar_transform_angles_tot);
						str_indx_MPC = max_gap_MPC.first; end_indx_MPC = max_gap_MPC.second;

						heading_angle_MPC= find_best_point_MPC(str_indx_MPC, end_indx_MPC, proc_ranges_MPC,lidar_transform_angles_tot);
						heading_angle=heading_angle_MPC;
						
						printf("HEADING %d, %lf\n",num_MPC,heading_angle);
						
						if(num_MPC==0){
					
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
					}

					//Rotate and translate the tracking line back to the original frame of reference
					
					heading_angle=heading_angle+theta_ref;
		
					//Find the startng point closest to the new tracking line
					

					theta_ref=heading_angle; //Process thetas so the difference between thetas is minimized(pick right next theta)
					

					while (theta_ref-heading_angle>M_PI) theta_ref-=2*M_PI;
					while (theta_ref-heading_angle<-M_PI) theta_ref+=2*M_PI;
					if(theta_ref-heading_angle>M_PI/2) theta_ref-=M_PI;
					if(theta_ref-heading_angle<-M_PI/2) theta_ref+=M_PI;
					while (theta_ref>M_PI) theta_ref-=2*M_PI;
					while (theta_ref<-M_PI) theta_ref+=2*M_PI;
					theta_refs[num_MPC]=theta_ref;
					//For first, drive 3/4 of the way, second drive the last 1/4 since first three points are fixed
					if(num_MPC==0){
						xpt=xpt+std::max(vel_adapt,min_speed)*bez_t_end*3.0/(bez_ctrl_pts-1)*cos(theta_ref); //Drive message relates to lidar callback scan topic, ~10Hz
						ypt=ypt+std::max(vel_adapt,min_speed)*bez_t_end*3.0/(bez_ctrl_pts-1)*sin(theta_ref); //Use 13 Hz as absolute optimal but likely slower use dt
					}
					else{
						xpt=xpt+std::max(vel_adapt,min_speed)*bez_t_end/(bez_ctrl_pts-1)*cos(theta_ref); //Drive message relates to lidar callback scan topic, ~10Hz
						ypt=ypt+std::max(vel_adapt,min_speed)*bez_t_end/(bez_ctrl_pts-1)*sin(theta_ref); //Use 13 Hz as absolute optimal but likely slower use dt
					}
					
					
					xptplot[num_MPC]=xpt;
					yptplot[num_MPC]=ypt;
					printf("THETA %d, %lf\n",num_MPC,theta_ref);

				
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

				//Select a subsample of all obstacles
				std::vector<std::vector<double>> bez_obs;
				std::vector<std::vector<double>> sub_bez_obs;
				for(int i=0;i<fused_ranges_MPC_tot0.size();i++){
					bez_obs.push_back({fused_ranges_MPC_tot0[i]*cos(lidar_transform_angles_tot0[i]),fused_ranges_MPC_tot0[i]*sin(lidar_transform_angles_tot0[i])});
				}

				double obs_sep1=obs_sep;
				int num_obs=100000;

				while(num_obs>max_obs){
					sub_bez_obs.clear();
					for (const auto& obstacle : bez_obs) {
						// Accessing x and y coordinates
						double x_ob = obstacle[0]; // x coordinate
						double y_ob = obstacle[1]; // y coordinate
						double min_dist=100;

						for (const auto& obstacle1 : sub_bez_obs) {
							double mydist=pow(pow(obstacle1[0]-x_ob,2)+pow(obstacle1[1]-y_ob,2),0.5);
							if(mydist<min_dist){
								min_dist=mydist;
							}
						}
						if(min_dist>obs_sep1 && pow(pow(x_ob,2)+pow(y_ob,2),0.5)<max_lidar_range_opt){
							// FILE *file1 = fopen("/home/gjsk/catkin_ws/bezier.txt", "a");
							// fprintf(file1,"%lf, %lf\n",obstacle[0],obstacle[1]);
							// fclose(file1);
							
							sub_bez_obs.push_back({obstacle[0],obstacle[1]});
						}
						
					}
					num_obs=sub_bez_obs.size();
					obs_sep1=obs_sep1*1.1;
				}
				// FILE *file1 = fopen("/home/gjsk/catkin_ws/bezier.txt", "a");
				// fprintf(file1,"*******************\n");
				// fclose(file1);

				//Find the desired bezier control points for the leader trajectory
				double bez_pts[2][5];
				double turn_ang=0;
				if(car_detects[0].state[4]<0){
					turn_ang=std::min(-min_delta,car_detects[0].state[4]);
				}
				else turn_ang=std::max(min_delta,car_detects[0].state[4]);
				double radius=std::abs(wheelbase/tan(turn_ang)); //Can't have div by 0 or inf radius
				if(leader_detect==1){
					double x_fin=radius*sin(bez_t_end*car_detects[0].state[3]/radius);
					double y_fin=radius*(1.0-cos(bez_t_end*car_detects[0].state[3]/radius));
					double ang_gap=bez_t_end*car_detects[0].state[3]/radius;
					if(car_detects[0].state[4]<0){
						ang_gap*=-1;
					}
					double uval_num=3*cos(ang_gap/2)*sin(ang_gap/2)-2*sin(ang_gap/2)+4*sqrt(2)*pow(sin(ang_gap/4),3);
					double uval=uval_num/(2*pow(cos(ang_gap/2),2));
					double rval=8/3-5/3*cos(ang_gap/2)-4/3*uval*sin(ang_gap/2);
						
					bez_pts[0][0]=1; bez_pts[1][0]=0;
					bez_pts[0][1]=1; bez_pts[1][1]=uval;
					bez_pts[0][2]=rval*cos(ang_gap/2); bez_pts[1][2]=rval*sin(ang_gap/2);
					bez_pts[0][3]=cos(ang_gap)+uval*sin(ang_gap); bez_pts[1][3]=sin(ang_gap)-uval*cos(ang_gap);
					bez_pts[0][4]=cos(ang_gap); bez_pts[1][4]=sin(ang_gap);

					//Multiply by R and subtract R from x coords to scale and reset to 0,0 base point
					bez_pts[0][0]=bez_pts[0][0]*radius-radius; bez_pts[1][0]=bez_pts[1][0]*radius;
					bez_pts[0][1]=bez_pts[0][1]*radius-radius; bez_pts[1][1]=bez_pts[1][1]*radius;
					bez_pts[0][2]=bez_pts[0][2]*radius-radius; bez_pts[1][2]=bez_pts[1][2]*radius;
					bez_pts[0][3]=bez_pts[0][3]*radius-radius; bez_pts[1][3]=bez_pts[1][3]*radius;
					bez_pts[0][4]=bez_pts[0][4]*radius-radius; bez_pts[1][4]=bez_pts[1][4]*radius;

					//Now, apply transform from leader to pursuer frame
					double copy[2][5];
					memcpy(copy, bez_pts, sizeof(bez_pts));  // Copy entire array
					double theta_rot=car_detects[0].state[2];
					if(car_detects[0].state[4]<0){ //Circular arc going the opposite direction of curvature
						theta_rot+=M_PI;
					}
					bez_pts[0][0]=car_detects[0].state[0]+copy[0][0]*cos(theta_rot)-copy[1][0]*sin(theta_rot);
					bez_pts[1][0]=car_detects[0].state[1]+copy[0][0]*sin(theta_rot)+copy[1][0]*cos(theta_rot);
					
					bez_pts[0][1]=car_detects[0].state[0]+copy[0][1]*cos(theta_rot)-copy[1][1]*sin(theta_rot);
					bez_pts[1][1]=car_detects[0].state[1]+copy[0][1]*sin(theta_rot)+copy[1][1]*cos(theta_rot);

					bez_pts[0][2]=car_detects[0].state[0]+copy[0][2]*cos(theta_rot)-copy[1][2]*sin(theta_rot);
					bez_pts[1][2]=car_detects[0].state[1]+copy[0][2]*sin(theta_rot)+copy[1][2]*cos(theta_rot);

					bez_pts[0][3]=car_detects[0].state[0]+copy[0][3]*cos(theta_rot)-copy[1][3]*sin(theta_rot);
					bez_pts[1][3]=car_detects[0].state[1]+copy[0][3]*sin(theta_rot)+copy[1][3]*cos(theta_rot);

					bez_pts[0][4]=car_detects[0].state[0]+copy[0][4]*cos(theta_rot)-copy[1][4]*sin(theta_rot);
					bez_pts[1][4]=car_detects[0].state[1]+copy[0][4]*sin(theta_rot)+copy[1][4]*cos(theta_rot);
					
				}
				else{ //No leader detected, just set to 0 placeholders
					bez_pts[0][0]=0; bez_pts[1][0]=0;
					bez_pts[0][1]=0; bez_pts[1][1]=0;
					bez_pts[0][2]=0; bez_pts[1][2]=0;
					bez_pts[0][3]=0; bez_pts[1][3]=0;
					bez_pts[0][4]=0; bez_pts[1][4]=0;
				}
				
				
				//PERFORM THE MPC NON-LINEAR OPTIMIZATION
				nlopt_opt opt;
				opt = nlopt_create(NLOPT_LD_SLSQP, bez_ctrl_pts*2-5); /* algorithm and dimensionality */
				//[0] -> x2
				//[1] -> x3
				//[2] -> y3
				//[3] -> x4
				//[4] -> y4

				for (int i=0;i<bez_ctrl_pts*2-5;i++){
					nlopt_set_lower_bound(opt, i, -max_lidar_range); //Bounds on max and min control point coordinates
					nlopt_set_upper_bound(opt, i, max_lidar_range);
				}

				double bez_x1=std::max(vel_adapt,min_speed)/4*bez_t_end; //Set the fixed point values here
				double bez_y2=4.0/3.0*pow(bez_x1,2)*tan(last_delta)/wheelbase; //last_delta based on optimization in sim, actual value returned by vesc in exp

				std::vector<double> opt_params1;
				std::vector<double> opt_params2;
				std::vector<double> opt_paramsp;
				
				opt_params1.push_back(num_obs+7); opt_params1.push_back(bez_ctrl_pts); opt_params1.push_back(bez_curv_pts); opt_params1.push_back(bez_alpha); opt_params1.push_back(bez_x1); opt_params1.push_back(bez_y2);
				//Leader reference control pts (only the last 3 pts since others are fixed for optimization)
				opt_params1.push_back(bez_pts[0][2]); opt_params1.push_back(bez_pts[1][2]); opt_params1.push_back(bez_pts[0][3]); opt_params1.push_back(bez_pts[1][3]); opt_params1.push_back(bez_pts[0][4]); opt_params1.push_back(bez_pts[1][4]);
				if(leader_detect==1){ //Leader detected
					opt_params1.push_back(pursuit_weight); 
				}
				else{ //No leader detected, pursuit weight is 0 for optimization purposes
					opt_params1.push_back(0);	
				}
				opt_params1.push_back(leader_detect); //For weighing objective terms

				opt_params2.push_back(num_obs+7); opt_params2.push_back(bez_ctrl_pts); opt_params2.push_back(bez_curv_pts); opt_params2.push_back(bez_beta); opt_params2.push_back(bez_x1); opt_params2.push_back(bez_y2);
				opt_params2.push_back(max_speed); opt_params2.push_back(min_speed); opt_params2.push_back(max_accel); opt_params2.push_back(max_steering_angle); opt_params2.push_back(std::abs(max_servo_speed*std::max(default_dt,dt)));
				opt_params2.push_back(bez_t_end); opt_params2.push_back(wheelbase); opt_params2.push_back(bez_min_dist);

				for (int i=0; i<num_obs;i++){ //Add all subsampled obstacles to the parameters
					opt_params1.push_back(sub_bez_obs[i][0]);
					opt_params1.push_back(sub_bez_obs[i][1]);
					opt_params2.push_back(sub_bez_obs[i][0]);
					opt_params2.push_back(sub_bez_obs[i][1]);
				}
				// printf("1: %d, %d 2: %d, %d\n",opt_params1.size(),2*(num_obs+7),opt_params2.size(),2*(num_obs+7));

				opt_paramsp.push_back(car_detects[0].state[0]-pursuit_x); //Our target to follow trajectory in X
				opt_paramsp.push_back(car_detects[0].state[1]-pursuit_y); //And Y
				opt_paramsp.push_back(car_detects[0].state[2]); //Leader's theta
				opt_paramsp.push_back(car_detects[0].state[3]); //Leader's constant velocity over traj
				double delta_lead=0;
				if(car_detects[0].state[4]<0) delta_lead=std::min(-min_delta,car_detects[0].state[4]); //Limit delta to prevent div by 0
				else delta_lead=std::max(min_delta,car_detects[0].state[4]);
				opt_paramsp.push_back(wheelbase/tan(delta_lead)); //Leader's circle arc (+ or -) radius, may need to use abs so we can use sign in some spots, abs radius other
				opt_paramsp.push_back(pursuit_weight);
				opt_paramsp.push_back(bez_curv_pts);
				opt_paramsp.push_back(bez_t_end); //Time of the pursuit trajectory, used to find the bez_curv_pts for the arc length of the leader's trajectory, with known vel_lead
				opt_paramsp.push_back(min_pursue); //Minimum distance allowed between pursuit and leader vehicle
				opt_paramsp.push_back(veh_det_length); //Leader vehicle length, for constructing box around center point of vehicle
				opt_paramsp.push_back(veh_det_width); //Leader vehicle width, for constructing box around center point of vehicle
				opt_paramsp.push_back(bez_beta);
				opt_paramsp.push_back(bez_x1);
				opt_paramsp.push_back(bez_y2);
				
				



				nlopt_set_min_objective(opt, myfunc, opt_params1.data());
				double tol[9*bez_curv_pts]={1e-8};
				double tolp[1]={1e-8};
				
				nlopt_add_inequality_mconstraint(opt, 9*bez_curv_pts, bezier_inequality_con, opt_params2.data(), tol);
				if(leader_detect==1){
					nlopt_add_inequality_mconstraint(opt, 1, pursuit_inequality_con, opt_paramsp.data(), tolp);
				}
			
				nlopt_set_xtol_rel(opt, 0.001); //Termination parameters
				nlopt_set_maxtime(opt, 0.05);

				double x[bez_ctrl_pts*2-5];  /* `*`some` `initial` `guess`*` */

				//Try new attempt at initial guess
				x[0]=xptplot[0]*2.0/3.0; //x2
				x[1]=xptplot[0]; //x3
				x[2]=yptplot[0]; //y3
				x[3]=xptplot[1]; //x4
				x[4]=yptplot[1]; //y4
				printf("%lf, %lf, %lf, %lf, %lf StartPt\n",x[0],x[1],x[2],x[3],x[4]);

				int successful_opt=0;

				double minf; /* `*`the` `minimum` `objective` `value,` `upon` `return`*` */
				ros::Time ott1 = ros::Time::now();
				double opt_time1 = ott1.toSec();
				nlopt_result optim= nlopt_optimize(opt, x, &minf); //This runs the optimization
				ros::Time ott2 = ros::Time::now();
				double opt_time2 = ott2.toSec();
				printf("OptTime: %lf, Evals: %d\n",opt_time2-opt_time1,nlopt_get_numevals(opt));

				double bez_x2=0;
				double bez_x3=0;
				double bez_y3=0;
				double bez_x4=0;
				double bez_y4=0;

				if(isnan(minf)){
					forcestop=1;
				}
				else{
					forcestop=0;
				}

				double min_dist1=100; //Min dist along trajectory, used for update pursuit weighting

				if (optim < 0) {
					safe_distance_adapt=safe_distance_adapt/2;
					printf("Optimization Error, %d, %lf, %lf, %lf, %lf, %lf\n",optim,x[0],x[1],x[2],x[3],x[4]);
					printf("NLOPT Error: %s\n", nlopt_get_errmsg(opt));
				}
				else {
					successful_opt=1;
					printf("Successful Opt: %d\n",optim);
					safe_distance_adapt=safe_distance;
					//Save the control points here
					printf("%lf, %lf, %lf, %lf, %lf SOLVED\n",x[0],x[1],x[2],x[3],x[4]);
					bez_x2=x[0];
					bez_x3=x[1];
					bez_y3=x[2];
					bez_x4=x[3];
					bez_y4=x[4];
					//Integrate over curve until next sample to obtain v, delta
					double dist_trav=0;
					double dist_des=std::max(vel_adapt,min_speed)*std::max(default_dt,dt);
					double t_des=0;
					double last_x=0; double last_y=0;
					for(int i=1; i<bez_curv_pts*100;i++){
						double t=double(i)/double(bez_curv_pts*100-1);
						double new_x=4*pow(1-t,3)*t*bez_x1+6*pow(1-t,2)*pow(t,2)*bez_x2+4*(1-t)*pow(t,3)*bez_x3+pow(t,4)*bez_x4;
						double new_y=6*pow(1-t,2)*pow(t,2)*bez_y2+4*(1-t)*pow(t,3)*bez_y3+pow(t,4)*bez_y4; //y1=0
						
						dist_trav=dist_trav+pow(pow(new_x-last_x,2)+pow(new_y-last_y,2),0.5);
						last_x=new_x; last_y=new_y;

						if(dist_trav>dist_des){
							t_des=t;
							break;
						}
					}
					
					double t= t_des;
					double x_dot=4*bez_x1*(-4*pow(t,3)+9*pow(t,2)-6*t+1)+6*x[0]*(4*pow(t,3)-6*pow(t,2)+2*t)+4*x[1]*(-4*pow(t,3)+3*pow(t,2))+4*x[3]*pow(t,3);
					double x_ddot=4*bez_x1*(-12*pow(t,2)+18*t-6)+6*x[0]*(12*pow(t,2)-12*t+2)+4*x[1]*(-12*pow(t,2)+6*t)+12*x[3]*pow(t,2);
					double x_dddot=4*bez_x1*(-24*t+18)+6*x[0]*(24*t-12)+4*x[1]*(-24*t+6)+24*x[3]*t;

					double y_dot=6*bez_y2*(4*pow(t,3)-6*pow(t,2)+2*t)+4*x[2]*(-4*pow(t,3)+3*pow(t,2))+4*x[4]*pow(t,3);
					double y_ddot=6*bez_y2*(12*pow(t,2)-12*t+2)+4*x[2]*(-12*pow(t,2)+6*t)+12*x[4]*pow(t,2);
					double y_dddot=6*bez_y2*(24*t-12)+4*x[2]*(-24*t+6)+24*x[4]*t;

					double curv=(x_dot*y_ddot-y_dot*x_ddot)/(pow(pow(x_dot,2)+pow(y_dot,2),1.5));
					double curv_dot=((x_dot*y_dddot-y_dot*x_dddot)*(pow(x_dot,2)+pow(y_dot,2))-3*(x_dot*x_ddot+y_dot*y_ddot)*(x_dot*y_ddot-y_dot*x_ddot))/pow((pow(x_dot,2)+pow(y_dot,2)),2.5);

					last_delta=atan2(curv*wheelbase,1);
					vel_adapt=std::min(pow(pow(x_dot,2)+pow(y_dot,2),0.5)/bez_t_end,max_speed);


					//Find minimum distance
					for(int i=1; i<bez_curv_pts;i++){
						t=double(i)/double(bez_curv_pts);
						double new_x=4*pow(1-t,3)*t*bez_x1+6*pow(1-t,2)*pow(t,2)*bez_x2+4*(1-t)*pow(t,3)*bez_x3+pow(t,4)*bez_x4;
						double new_y=6*pow(1-t,2)*pow(t,2)*bez_y2+4*(1-t)*pow(t,3)*bez_y3+pow(t,4)*bez_y4; //y1=0
						for(int j=0; j<num_obs;j++){
							double my_dist=pow(pow(new_x-sub_bez_obs[j][0],2)+pow(new_y-sub_bez_obs[j][1],2),0.5);
							if(my_dist<min_dist1){
								min_dist1=my_dist;
							}
						}
					}
					if(min_dist1<bez_min_dist){
						printf("Unsafe MinDist: %lf\n",min_dist1);
						vel_adapt=0;
					}
					

					// printf("%lf, %lf %lf, %lf, %lf Delta Vel\n",last_delta,vel_adapt,curv*wheelbase,dist_des,t);
					// printf("(%lf,0), (%lf,%lf), (%lf,%lf), (%lf,%lf)",bez_x1,bez_x2,bez_y2,bez_x3,bez_y3,bez_x4,bez_y4);
					
				}
				startcheck=1;

				nlopt_destroy(opt);

				//Update the min_dist to weight middle-line MPC vs. pursuit of leader
				//***************************************************************** */
				if(successful_opt==1){
					double aval=2*transit_rate/(pursuit_dist-MPC_dist);
					double bval=-transit_rate-2*transit_rate*MPC_dist/(pursuit_dist-MPC_dist);
					double update_weight=aval*min_dist1+bval; //Linear eqn for update weight
					update_weight=std::min(std::max(-transit_rate,update_weight),transit_rate); //Clip at max transition rate
					
					pursuit_weight=pursuit_weight+update_weight; //Update prusuit weight term
					pursuit_weight=std::min(std::max(0.0,pursuit_weight),1.0); //Clip at 0, 1
					printf("Pursuit Weight %lf, %lf, %lf\n",pursuit_weight,min_dist1, update_weight);
				}
				//***************************************************************** */


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



				//Publish the bezier points
				bez.header.frame_id = "base_link";
				bez.header.stamp = ros::Time::now();
				bez.type = visualization_msgs::Marker::POINTS;
				bez.id = 0; 
				bez.ns = "points";
				bez.action = visualization_msgs::Marker::ADD;
				bez.scale.x = 0.1;
				bez.color.a = 1.0;
				bez.color.r = 0.6; 
				bez.color.g = 0.2;
				bez.color.b = 0.5;
				bez.pose.orientation.w = 1;

				bez.scale.x = 0.1;  // Size of points
				bez.scale.y = 0.1;
				
				bez.lifetime = ros::Duration(0.1);
				geometry_msgs::Point p5;
				bez.points.clear();
				for (const auto& obstacle1 : sub_bez_obs) {
					p5.x = obstacle1[0]; p5.y = obstacle1[1]; p5.z = 0;
					bez.points.push_back(p5);
				}
				// p5.x = xptplot[0]; p5.y = yptplot[0]; p5.z = 0;
				// bez.points.push_back(p5);
				// p5.x = xptplot[1]; p5.y = yptplot[1]; p5.z = 0;
				// bez.points.push_back(p5);

				// printf("%lf, %lf, %lf, %lf, %lf, %lf, %lf\n",bez_x1,bez_y2,bez_x2,bez_y3,bez_x3,bez_x4,bez_y4);
				for(int i=0; i<bez_curv_pts*10; i++){
					double t=double(i)/double(bez_curv_pts*10-1);
					double bez_x=4*pow(1-t,3)*t*bez_x1+6*pow(1-t,2)*pow(t,2)*bez_x2+4*(1-t)*pow(t,3)*bez_x3+pow(t,4)*bez_x4;
					double bez_y=6*pow(1-t,2)*pow(t,2)*bez_y2+4*(1-t)*pow(t,3)*bez_y3+pow(t,4)*bez_y4; //y1=0
					p5.x = bez_x; p5.y = bez_y; p5.z = 0;
					// printf("%lf, %lf\n",bez_x,bez_y);
					bez.points.push_back(p5);
				}

				bez_mark.publish(bez);

				//Implement steering angle based on opt now, use last_delta returned as next delta_init for simulation (overwritten for experiment)



				//Ackermann Steering
				

				
				min_distance = max_lidar_range + 100; int idx1, idx2;
				idx1 = -sec_len+int(scan_beams/2); idx2 = sec_len + int(scan_beams/2);

				for(int i = idx1; i <= idx2; ++i){
					if(fused_ranges[i] < min_distance) min_distance = fused_ranges[i];
				}

				velocity_scale = 1 - exp(-std::max(min_distance-stop_distance,0.0)/stop_distance_decay); //factor ensures we only slow when appropriate, otherwise MPC behaviour dominates
				
				// ROS_INFO("%.3f", velocity);

				delta_d=last_delta; //Use next delta command now to allow servo to transition

				velocity_MPC = velocity_scale*vel_adapt; //Implement slowing if we near an obstacle

				// velocity_MPC=std::max(vel_adapt,min_speed); //REMOVE

				vel_adapt=velocity_MPC;
				printf("SLOW %lf%% %lf m/s %lf m\n",velocity_scale*100,velocity_MPC,min_distance);

			}

			ackermann_msgs::AckermannDriveStamped drive_msg; 
			drive_msg.header.stamp = ros::Time::now();
			drive_msg.header.frame_id = "base_link";
			if(startcheck==1){
				drive_msg.drive.steering_angle = delta_d; //delta_d
				if(forcestop==0){ //If the optimization fails for some reason, we get nan: stop the vehicle
					drive_msg.drive.speed = velocity_MPC; //velocity_MPC
					printf("VEL: %lf\n",velocity_MPC);
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
			ros::Time ta = ros::Time::now();
			// printf("LIDAR end: %lf\n",ta.toSec()-current_time);
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
