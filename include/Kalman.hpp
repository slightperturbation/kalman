// Licensed under the Eclipse Public License - v 1.0, see License.txt
#include <vector>
#include <array>
#include <memory>
#include <exception>
#include <algorithm>
#include <cmath>

#include <Eigen/Dense>

namespace support
{

/// KalmanFilter tracks the discrete movement of a single point in state_dim_t dimensions, 
/// as it moves according to a control signal of control_dim_t dimensions and with noisy 
/// input of measurement_dim_t dimensions.
/// Usage is very simple. Create an instance by providing the full system model in the constructor.
/// Each timestep, available measurements and the known control signal are combined to update
/// the state of the system in the update() method.
/// The output of the system is available either through currentState() or predictNext().
template< typename scalar, size_t state_dim_t, size_t measurement_dim_t, size_t control_dim_t>
class KalmanFilter
{
public:
    typedef Eigen::Matrix< scalar, state_dim_t, 1 > state_vector_t;
    typedef Eigen::Matrix< scalar, measurement_dim_t, 1 > measurement_vector_t;
    typedef Eigen::Matrix< scalar, control_dim_t, 1 > control_vector_t;

private:
    // old state to new state
    Eigen::Matrix< scalar, state_dim_t, state_dim_t> m_stateTransitionMatrix;
    Eigen::Matrix< scalar, state_dim_t, state_dim_t>& A = m_stateTransitionMatrix;

    // control vector to state vector
    Eigen::Matrix< scalar, state_dim_t, control_dim_t> m_controlMatrix;
    Eigen::Matrix< scalar, state_dim_t, control_dim_t>& B = m_controlMatrix;

    // state vector to measurement vector
    Eigen::Matrix< scalar, measurement_dim_t, state_dim_t> m_observationMatrix;
    Eigen::Matrix< scalar, measurement_dim_t, state_dim_t>& H = m_observationMatrix;


    /// Estimated process noise covariance, how much noise is expected when updating
    /// the state to the next time step.
    Eigen::Matrix< scalar, state_dim_t, state_dim_t> m_processNoiseCovariance;
    Eigen::Matrix< scalar, state_dim_t, state_dim_t>& Q = m_processNoiseCovariance;

    /// Estimated measurement noise covariance, how much noise is expected in the measurements.
    Eigen::Matrix< scalar, measurement_dim_t, measurement_dim_t> m_measurementNoiseCovariance;
    Eigen::Matrix< scalar, measurement_dim_t, measurement_dim_t>& R = m_measurementNoiseCovariance;
    
    //////////////////////////////
    // State variables
    state_vector_t m_stateEstimate;
    state_vector_t& x = m_stateEstimate;
    
    Eigen::Matrix< scalar, state_dim_t, state_dim_t > m_predictedCovarianceEstimate;
    Eigen::Matrix< scalar, state_dim_t, state_dim_t >& P = m_predictedCovarianceEstimate;
    
    /// y stores the last measurement error from the previous update for use by predictors.
    Eigen::Matrix< scalar, measurement_dim_t, 1 > y;

public:
    /// Create a KF to track a state using an online set of measurements.
    ///
    /// ProcessNoiseCovariance (Q) indicates how much uncertainty to expect in the state transitions.
    /// If an element of Q is large then that state element will be tracked more closely.  If Q is small
    /// the state will be tracked loosely.  Think of Q as the gain in a PD controller.
    ///
    /// MeasurementNoiseCovariance (R) is how much noise is expected in the measurement input. If R is
    /// large, the measurements are considered to be inaccurate, smaller R indicates the measurements
    /// are likely correct. Think of R as the damping in a PD controller.
    ///
    /// Note that R and Q are covariance matrices, so if state/measurements are independent, then they
    /// are diagonal.
    KalmanFilter( const Eigen::Matrix< scalar, state_dim_t, state_dim_t >& argStateTransitionMatrix
                , const Eigen::Matrix< scalar, state_dim_t, control_dim_t >& argControlMatrix
                , const Eigen::Matrix< scalar, measurement_dim_t, state_dim_t >& argObservationMatrix
                , const state_vector_t& argInitialState
                , const Eigen::Matrix< scalar, state_dim_t, state_dim_t >& argInitialCovarianceEstimate
                , const Eigen::Matrix< scalar, state_dim_t, state_dim_t>& argProcessNoiseCovariance
                , const Eigen::Matrix< scalar, measurement_dim_t, measurement_dim_t>& argMeasurementNoiseCovariance )
    {
        A = argStateTransitionMatrix;
        B = argControlMatrix;
        H = argObservationMatrix;
        x = argInitialState;
        P = argInitialCovarianceEstimate;
        Q = argProcessNoiseCovariance;
        R = argMeasurementNoiseCovariance;
    }
    
    Eigen::Matrix< scalar, state_dim_t, 1 > getCurrentState()
    {
        return x;
    }
    
    void update( const control_vector_t& control
               , const measurement_vector_t& measurement )
    {
        ///////////
        // Prediction
        // 1) Project current state (x) into the future using state transition and control.
        state_vector_t x_p = A * x + B * control;
        // 2) Project the covariance into the future using state transition and estimated error.
        Eigen::Matrix< scalar, state_dim_t, state_dim_t > P_p = A * P * A.transpose() + Q;

        // Correction
        // 1) Compute the Kalman gain
        // 1a) Estimate the "innovation" covariance from projected covariance plus the measurement noise R.
        Eigen::Matrix< scalar, measurement_dim_t, measurement_dim_t > S = H * P_p * H.transpose() + R;
        // 1b) Final Kalmna gain, K = PH^T(HPH^T+R)^-1
        Eigen::Matrix< scalar, state_dim_t, measurement_dim_t > K = P_p * H.transpose() * S.inverse();
        // 2) Update our state estimate
        // 2a) Compare measurement to what our prediction should have measured
        const measurement_vector_t& z = measurement;
        Eigen::Matrix< scalar, measurement_dim_t, 1 > y = z - H * x_p;
        // 2b) Update using Kalman gain times error
        x = x_p + K * y;
        // 3) Update our estimate of the noise covariance
        P = ( Eigen::Matrix< scalar, state_dim_t, state_dim_t >::Identity() - K * H ) * P_p;
    }
    
    // Provide a prediction without a corresponding measurement
    Eigen::Matrix< scalar, state_dim_t, 1 > predictNext() const
    {
        // Project the covariance into the future using state transition and estimated error.
        Eigen::Matrix< scalar, state_dim_t, state_dim_t > P_p = A * P * A.transpose() + Q;
        // Estimate the "innovation" covariance from projected covariance plus the measurement noise R.
        Eigen::Matrix< scalar, measurement_dim_t, measurement_dim_t > S = H * P_p * H.transpose() + R;
        // Final Kalmna gain, K = PH^T(HPH^T+R)^-1
        Eigen::Matrix< scalar, state_dim_t, measurement_dim_t > K = P_p * H.transpose() * S.inverse();
        
        state_vector_t out = A * x + K * y;
    }

};


}