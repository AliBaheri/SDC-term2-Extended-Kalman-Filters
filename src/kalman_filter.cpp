#include "kalman_filter.h"
#include <math.h>       // for atan2 
using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

const float PI_ = 3.14159265358979f;


// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  * predict the state
  */
  x_ = F_ * x_;
  MatrixXd Ft_ = F_.transpose();
  P_ = F_ * P_ * Ft_ + Q_;
}

void KalmanFilter::Kalman(const VectorXd &y){
  /*
  * Kalman filter
  */
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;
  // new estimate
  x_ += K*y;
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K*H_) * P_;

}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  * update the state by using Kalman Filter equations
  */
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  Kalman(y);

}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  * update the state by using Extended Kalman Filter equations
  */
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);
  
  float rho = sqrt(px*px +py*py); // range
  float phi;

  // https://discussions.udacity.com/t/ekf-radar-causes-rmse-to-go-through-the-roof/243944/5
  if (fabs(px) >= 0.0001){ // avoid division by zero
    phi = atan2(py, px); // bearing
  }
  
  float rho_dot;
  
  if (fabs(rho) >= 0.0001){
    rho_dot = (px * vx + py * vy) / rho;
  }

  VectorXd h = VectorXd(3);
  h << rho, phi, rho_dot;

  VectorXd y = z - h;

  while (y(1) > PI_) y(1) -= 2*PI_;
  
  while (y(1) < -PI_) y(1) += 2*PI_;
  
  Kalman(y);

}
