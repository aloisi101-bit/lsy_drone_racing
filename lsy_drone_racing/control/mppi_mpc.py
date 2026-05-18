import numpy as np
from lsy_drone_racing.control import Controller

class MPPIController(Controller):
    def __init__(self, obs, info, config):
        super().__init__(obs, info, config)
        
        # MPPI Hyperparameters
        self.K = 1000      # Number of sampled trajectories
        self.T = 25        # Prediction horizon (same as your Acados N)
        self.dt = 1 / config.env.freq
        
        # Noise covariance (Tune these! [thrust, roll, pitch, yaw])
        self.noise_sigma = np.array([1.5, 0.1, 0.1, 0.1]) 
        self.lambda_ = 1.0 # Temperature parameter for softmax weighting
        
        # Initialize the nominal control sequence
        self.U = np.zeros((self.T, 4)) 
        # Hover thrust baseline
        self.U[:, 0] = self.drone_params["mass"] * 9.81

    def _batch_dynamics(self, states, controls):
        """
        Simulate 1 step forward for K trajectories simultaneously.
        states: (K, 12) array of [pos, rpy, vel, drpy]
        controls: (K, 4) array of [thrust, roll, pitch, yaw]
        """
        # Extract states
        pos = states[:, 0:3]
        rpy = states[:, 3:6]
        vel = states[:, 6:9]
        
        # Example Simplified Euler Integration (Replace with actual drone dynamics)
        # Note: You can translate the math from drone_models.so_rpy into numpy matrix operations
        next_pos = pos + vel * self.dt
        
        # Add physics logic here for next_vel based on thrust and orientation...
        next_vel = vel # + (acceleration * dt)
        next_rpy = controls[:, 1:4] # Assuming direct attitude control for simplicity
        
        # Combine back into shape (K, 12)
        next_states = np.concatenate([next_pos, next_rpy, next_vel, ...], axis=1)
        return next_states
    
    def compute_control(self, obs, info=None):
        # 1. Shift previous control sequence forward (Receding Horizon)
        self.U[:-1] = self.U[1:]
        self.U[-1] = self.U[-2] # Duplicate last action
        
        # 2. Generate random noise for K samples over T steps
        # noise shape: (K, T, 4)
        noise = np.random.normal(loc=0.0, scale=self.noise_sigma, size=(self.K, self.T, 4))
        
        # 3. Create K control sequences
        # V shape: (K, T, 4)
        V = self.U + noise 
        
        # Clip controls to actuator limits!
        V[:, :, 0] = np.clip(V[:, :, 0], self.drone_params["thrust_min"], self.drone_params["thrust_max"])
        V[:, :, 1:4] = np.clip(V[:, :, 1:4], -0.5, 0.5) # Example RPY limits
        
        # 4. Rollout Dynamics
        # Start all K trajectories at the current drone state
        current_state = self._extract_state_array(obs)
        trajectories = np.zeros((self.K, self.T, 12))
        states = np.tile(current_state, (self.K, 1)) # Shape (K, 12)
        
        for t in range(self.T):
            states = self._batch_dynamics(states, V[:, t, :])
            trajectories[:, t, :] = states
            
        # 5. Compute Costs
        costs = self._compute_costs(trajectories, obs)
        
        # 6. Information-Theoretic Update (Softmax)
        beta = np.min(costs) # Subtracted for numerical stability
        weights = np.exp(-1.0 / self.lambda_ * (costs - beta))
        weights = weights / np.sum(weights) # Normalize to sum to 1
        
        # 7. Update nominal control sequence
        # Weight the noise by the trajectory probabilities
        for t in range(self.T):
            self.U[t] += np.sum(weights[:, None] * noise[:, t, :], axis=0)
            
        # 8. Return the very first action of the optimized sequence
        action = self.U[0]
        
        self._tick += 1
        return action
    
    