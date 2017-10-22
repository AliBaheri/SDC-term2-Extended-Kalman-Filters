#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

using namespace std;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  * Calculate the RMSE here.
  */
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  if(estimations.size()!=ground_truth.size() || estimations.size()==0){
  	cout << "Invalid estimation or ground_truth data" << endl;
  	return rmse;
  }

  // accumulated squared residuals.
  for(unsigned int i=0; i<estimations.size(); ++i){
  	VectorXd residual = estimations[i] - ground_truth[i];
  	// coefficient wise multiplication
  	residual = residual.array()*residual.array();
  	rmse = rmse + residual;
  }
  // calculate the mean
  rmse = rmse / estimations.size();
  // calculate the squared root
  rmse = rmse.array().sqrt();

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
  * Calculate a Jacobian here.
  */
  MatrixXd Hj(3,4);
  // recover state 
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);
  // cout << "[px,py,vx,vy]:" << px, py, vx, vy;
  // printf("Jacobian [px, py, vx, vy]: %g %g %g %g \n", px, py, vx, vy);


  // compute the Jacobian matrix
  float px2npy2 = pow(px,2) + pow(py,2);
  float sq_px2_py2 = sqrt(px2npy2);
  float threetwo_px2_py2 = px2npy2 * sq_px2_py2;

    // if(px==0 || py==0 || (px==0 && py==0)){
  if(fabs(px2npy2) < 0.0001){
    cout << "Error - Divided by zero!" << endl;
    return Hj;
  }

  Hj << (px/sq_px2_py2), (py/sq_px2_py2), 0, 0,
        (-py/px2npy2), (px/px2npy2), 0, 0,
        (py*(vx*py - vy*px)/threetwo_px2_py2), (px*(vy*px - vx*py)/threetwo_px2_py2), (px/sq_px2_py2), (py/sq_px2_py2);

  return Hj;

}
