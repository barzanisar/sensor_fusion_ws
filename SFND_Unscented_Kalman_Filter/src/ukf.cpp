#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
{

  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // previous timestamp
  time_us_ = 0;

  // state dimension
  n_x_ = 5;
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = 0.3*MatrixXd::Identity(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2 (half of max expected)
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.8;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
   * End DO NOT MODIFY section for measurement noise values 
   */

  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  weights_ = VectorXd(2 * n_aug_ + 1);

  // set weights
  double weights = 0.5 / (n_aug_ + lambda_);
  weights_.fill(weights);
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  // Measurement covariance
  R_radar_ = MatrixXd::Identity(3, 3);
  R_radar_(0, 0) = std_radr_ * std_radr_;
  R_radar_(1, 1) = std_radphi_ * std_radphi_;
  R_radar_(2, 2) = std_radrd_ * std_radrd_;

  R_lidar_ = MatrixXd::Identity(2, 2);
  R_lidar_(0, 0) = std_laspx_ * std_laspx_;
  R_lidar_(1, 1) = std_laspy_ * std_laspy_;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(const MeasurementPackage &meas_package)
{
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if (!is_initialized_)
  {
    //cout << "Kalman Filter Initialization " << endl;
    if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      std::cout << "Received Laser Meas! " << std::endl;
      // Initialise the state with position measurement from Lidar and initial guesses for velocity, yaw and yaw rate
      x_ << meas_package.raw_measurements_[0],
          meas_package.raw_measurements_[1],
          0,
          0,
          0;

      time_us_ = meas_package.timestamp_;

      is_initialized_ = true;

      std::cout << "Initial x_ :" << std::endl;
      std::cout << x_ << std::endl;
      std::cout << "Initial P_ :" << std::endl;
      std::cout << P_ << std::endl;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      std::cout << "Received Radar Meas! " << std::endl;
      // Initialise the state with position measurement from Radar and initial guesses for velocity, yaw and yaw rate
      double rho = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      double rho_dot = meas_package.raw_measurements_[2];

      x_ << rho * cos(phi),
          rho * sin(phi),
          rho_dot,
          phi,
          0; // v = rho_dot and yaw = phi when the tracked car is moving in a straight line towards or away from ego car.

      time_us_ = meas_package.timestamp_;

      is_initialized_ = true;

      std::cout << "Initial x_ :" << std::endl;
      std::cout << x_ << std::endl;
      std::cout << "Initial P_ :" << std::endl;
      std::cout << P_ << std::endl;
    }

    return;
  }

  // compute the time elapsed between the current and previous measurements
  // dt - expressed in seconds
  float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  std::cout << "Delta t in sec :" << std::endl;
  std::cout << dt << std::endl;

  Prediction(dt);

  switch (meas_package.sensor_type_)
  {
  case MeasurementPackage::LASER:
    UpdateLidar(meas_package);
    break;
  case MeasurementPackage::RADAR:
    UpdateRadar(meas_package);
    break;

  default:
    std::cout << "Some other measurement received!" << std::endl;
    break;
  }

  // print result
  std::cout << "Updated state" << std::endl;
  std::cout << x_ << std::endl;
  std::cout << "Updated covariance matrix" << std::endl;
  std::cout << P_ << std::endl;
}

void UKF::GenerateSigmaPoints(MatrixXd &Xsig_aug)
{
  // create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);

  // create augmented mean state
  x_aug.head(n_x_) = x_;
  x_aug.tail(2) = MatrixXd::Zero(2, 1); // zero mean for process noise = [nu_acc, nu_yawdd]

  // create augmented covariance matrix with process noise covariance in bottom right corner
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

  // create square root matrix and multiply with scaling factor
  MatrixXd A = P_aug.llt().matrixL();

  // create augmented sigma points matrix
  Xsig_aug.col(0) = x_aug;
  for (int i = 1; i <= n_aug_; ++i)
  {
    Xsig_aug.col(i) = x_aug + sqrt(lambda_ + n_aug_) * A.col(i - 1);
    Xsig_aug.col(n_aug_ + i) = x_aug - sqrt(lambda_ + n_aug_) * A.col(i - 1);
  }

  std::cout << "Xsig_aug = " << std::endl
            << Xsig_aug << std::endl;
}

void UKF::PredictSigmaPoints(const MatrixXd &Xsig_aug, double delta_t)
{
  // predict sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    // extract values for better readability
    double p_x = Xsig_aug(0, i);
    double p_y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    // predicted state values
    double px_p, py_p;

    // avoid division by zero
    if (fabs(yawd) > 0.001)
    {
      px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    }
    else
    {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;

    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    // write predicted sigma point into right column
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }

  std::cout << "Xsig_pred_ = " << std::endl
            << Xsig_pred_ << std::endl;
}

void UKF::CalculateMeanAndCovarianceFromSigmaPoints()
{
  // predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  { // iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  // predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  { // iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // angle normalization
    while (x_diff(3) > M_PI)
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI)
      x_diff(3) += 2. * M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

void UKF::Prediction(double delta_t)
{
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

  // Step 1: Generate sigma points
  // create augmented sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  GenerateSigmaPoints(Xsig_aug);

  // Step 2: Predict sigma points
  PredictSigmaPoints(Xsig_aug, delta_t);

  // Step 3: Predict mean and covariance from sigma points
  CalculateMeanAndCovarianceFromSigmaPoints();

  // print result
  std::cout << "Predicted state" << std::endl;
  std::cout << x_ << std::endl;
  std::cout << "Predicted covariance matrix" << std::endl;
  std::cout << P_ << std::endl;
}

void UKF::UpdateLidar(const MeasurementPackage &meas_package)
{
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  // Normal kalman filtering (normal sensor fusion)
  MatrixXd H = MatrixXd(2, 5);
  H << 1, 0, 0, 0, 0,
      0, 1, 0, 0, 0;

  // z_pred = H*x_pred
  VectorXd z_pred = VectorXd(2);
  z_pred << x_(0), x_(1);

  VectorXd y = meas_package.raw_measurements_ - z_pred;
  MatrixXd Ht = H.transpose();
  MatrixXd S = H * P_ * Ht + R_lidar_;
  MatrixXd S_inv = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * S_inv;

  //new estimate
  x_ = x_ + (K * y);

  uint x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H) * P_;

  // // angle normalization
  // while (x_(3) > M_PI)
  //   x_(3) -= 2. * M_PI;
  // while (x_(3) < -M_PI)
  //   x_(3) += 2. * M_PI;

  // Normalized Innovation Squared (NIS)
  double NIS = y.transpose() * S_inv * y;

  std::cout << "Lidar S_inv: " << std::endl
            << S_inv << std::endl
            << "Kalman gain :" << K << std::endl;

  std::cout << "Laser NIS: " << NIS << std::endl; // Laser NIS should be mostly between 0.103 and 5.991 for estimator to be called "consistent" 
}

void UKF::PredictRadarSigmaPointsAndMean(MatrixXd &Zsig_pred, VectorXd &Zmean_pred)
{
  int n_sigmaPts = 2 * n_aug_ + 1;

  for (int i = 0; i < n_sigmaPts; ++i)
  {
    // extract values for better readability
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    // predict measurements for each sigma point
    Zsig_pred(0, i) = sqrt(p_x * p_x + p_y * p_y);                             //rho_pred
    Zsig_pred(1, i) = atan2(p_y, p_x);                                         //phi_pred
    if (fabs(Zsig_pred(0, i)) > 0.00001){
      Zsig_pred(2, i) = v * (p_x * cos(yaw) + p_y * sin(yaw)) / Zsig_pred(0, i); //rho_dot_pred
    }
    else
    {
      Zsig_pred(2, i) = 0;
    }
    

    Zmean_pred += weights_(i) * Zsig_pred.col(i);
  }
  std::cout << "Radar Zsig_pred: " << std::endl
            << Zsig_pred << std::endl
            << "Radar Zmean_pred " << Zmean_pred << std::endl;
}

void UKF::CalculateRadarInnovationCovariance(const MatrixXd &Zsig_pred, const VectorXd &Zmean_pred, MatrixXd &S)
{
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    VectorXd z_diff = Zsig_pred.col(i) - Zmean_pred;
    // angle normalization
    while (z_diff(1) > M_PI)
      z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI)
      z_diff(1) += 2. * M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  S = S + R_radar_;

  std::cout << "Radar Innov Covar S: " << std::endl
            << S << std::endl;
}

void UKF::CalculateRadarT(const MatrixXd &Zsig_pred, const VectorXd &Zmean_pred, MatrixXd &T)
{
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // angle normalization
    while (x_diff(3) > M_PI)
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI)
      x_diff(3) += 2. * M_PI;

    VectorXd z_diff = Zsig_pred.col(i) - Zmean_pred;
    // angle normalization
    while (z_diff(1) > M_PI)
      z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI)
      z_diff(1) += 2. * M_PI;

    T = T + weights_(i) * x_diff * z_diff.transpose();
  }

  std::cout << "Radar T: " << std::endl
            << T << std::endl;
}

void UKF::UpdateRadar(const MeasurementPackage &meas_package)
{
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  // Step 1: Predict measurement sigma points, its mean and covariance
  // No need of augmentation of meas noise since it is simply added in the measurement model
  // (i.e. linear relation of noise wrt the measurement model)

  // predicted measurement sigma points
  MatrixXd Zsig_pred = MatrixXd(3, 2 * n_aug_ + 1);

  // mean of predicted measurement sigma points
  VectorXd Zmean_pred = VectorXd::Zero(3);

  // predict measurement sigma points from Xsig_pred and radar measurement model
  PredictRadarSigmaPointsAndMean(Zsig_pred, Zmean_pred);

  // Covariance of predicted measurement
  MatrixXd S = MatrixXd::Zero(3, 3);

  CalculateRadarInnovationCovariance(Zsig_pred, Zmean_pred, S);

  // Step2: Update State
  // Cross-correlation of sigma points in state space and in measurement space
  MatrixXd T = MatrixXd::Zero(n_x_, 3);
  CalculateRadarT(Zsig_pred, Zmean_pred, T);

  MatrixXd S_inv = S.inverse();
  MatrixXd K = T * S_inv;
  VectorXd y = meas_package.raw_measurements_ - Zmean_pred; // innovation

  // Update state and covariance
  x_ = x_ + K * y;
  P_ = P_ - K * S * K.transpose();

  // // angle normalization
  // while (x_(3) > M_PI)
  //   x_(3) -= 2. * M_PI;
  // while (x_(3) < -M_PI)
  //   x_(3) += 2. * M_PI;

  // Normalized Innovation Squared (NIS)
  double NIS = y.transpose() * S_inv * y;

  std::cout << "Radar S_inv: " << std::endl
            << S_inv << std::endl
            << "Kalman gain :" << K << std::endl;
  std::cout << "RADAR NIS: " << NIS << std::endl; // Radar NIS should be mostly between 0.352 and 7.815 for estimator to be called "consistent"
}