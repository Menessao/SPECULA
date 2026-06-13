import numpy as np
from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.connections import InputValue
from specula.base_value import BaseValue
from specula.data_objects.iir_filter_data import IirFilterData


class DessenneGainOptimizer(BaseProcessingObj):
    """
    Dessenne Predictive Controller Optimizer processing object.
    
    Implements the online optimization of a linear predictive controller of order (p, q)
    using a modified Recursive Least Squares (RLS) algorithm as proposed in 
    Dessenne, Madec & Rousset 1998 (Section 4.C, Equation 22).
    
    Extended to support fractional loop delays via linear history interpolation.
    """

    def __init__(self,
                 nmodes: int,
                 p: int = 1,                      # Autoregressive order (past commands)
                 q: int = 0,                      # Moving average order (past open-loop values)
                 delay: float = 2.5,              # Loop latency in frames (can be fractional, e.g., 2.5)
                 forgetting_factor: float = 0.99, # RLS forgetting factor (\lambda)
                 rls_init_p: float = 1e3,         # Initial covariance scaling
                 initial_gain: float = 0.5,       # Starting integrator gain
                 target_device_idx: int = None,
                 precision: int = None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.nmodes = nmodes
        self.p = p
        self.q = q
        self.delay = delay
        self.lam = forgetting_factor
        
        # Deconstruct fractional delay into integer floor and fractional remainder
        self.delay_int = int(np.floor(delay))
        self.delay_frac = delay - self.delay_int
        
        # Total number of estimated parameters per mode
        self.n_vars = p + q + 1

        # RLS State Matrices:
        # Parameter matrix \theta: shape (nmodes, n_vars)
        self.theta = self.xp.zeros((self.nmodes, self.n_vars), dtype=self.dtype)
        
        # Initialize to act as a standard integrator at t=0
        if self.p >= 1:
            self.theta[:, 0] = 1.0               # a_1 coefficient (past command)
        self.theta[:, self.p] = initial_gain     # b_0 coefficient (current open-loop)

        # Covariance matrix P: shape (nmodes, n_vars, n_vars)
        self.P = self.xp.stack([self.xp.eye(self.n_vars, dtype=self.dtype) * rls_init_p 
                                for _ in range(self.nmodes)])

        # History queues to construct the delayed/interpolated regressor vector \phi
        self.c_history = []  # History of DM commands
        self.u_history = []  # History of Pseudo-Open-Loop values

        # Construct the target IirFilterData object directly
        self.iir_filter_data = IirFilterData()
        self.iir_filter_data.num = self.xp.zeros((self.nmodes, self.q + 1), dtype=self.dtype)
        self.iir_filter_data.den = self.xp.zeros((self.nmodes, self.p + 1), dtype=self.dtype)
        self.iir_filter_data.gain = self.xp.ones(self.nmodes, dtype=self.dtype)

        # Set up default baseline filters
        self.iir_filter_data.den[:, 0] = 1.0
        if self.p > 0:
            self.iir_filter_data.den[:, 1:] = -self.theta[:, :self.p]
        self.iir_filter_data.num[:, :] = self.theta[:, self.p:]

        # Inputs & Outputs definition
        self.inputs['wfs_meas'] = InputValue(type=BaseValue)
        self.inputs['dm_commands'] = InputValue(type=BaseValue)
        self.inputs['optical_gains'] = InputValue(type=BaseValue, optional=True)
        
        # Direct assignment of the data object to the outputs dictionary
        self.outputs['out_iir_filter_data'] = self.iir_filter_data

    @classmethod
    def input_names(cls):
        return {
            'wfs_meas': InputDesc(BaseValue, 'Residual WFS measurements vector'),
            'dm_commands': InputDesc(BaseValue, 'DM commands vector'),
            'optical_gains': InputDesc(BaseValue, 'Optional Optical Gain correction vector', optional=True)
        }

    @classmethod
    def output_names(cls):
        return {'iir_filter_data': OutputDesc(IirFilterData, 'Optimized IIR filter coefficients data object')}

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.current_wfs = self.local_inputs['wfs_meas'].value
        self.current_dm = self.local_inputs['dm_commands'].value
        
        if self.local_inputs['optical_gains'] is not None:
            self.current_og = self.local_inputs['optical_gains'].value
        else:
            self.current_og = None

    def _interpolate_history(self, history_list, target_idx):
        """
        Helper to perform linear interpolation across history elements for fractional indices.
        """
        lower_val = history_list[target_idx]
        upper_val = history_list[target_idx - 1]
        return (1.0 - self.delay_frac) * lower_val + self.delay_frac * upper_val

    def trigger_code(self):
        # 1. Compute current Pseudo-Open-Loop (POL) state: u = c + w / G_opt
        if self.current_og is not None:
            safe_og = self.xp.where(self.xp.abs(self.current_og) < 1e-6, 1e-6, self.current_og)
            u_k = self.current_dm + (self.current_wfs / safe_og)
        else:
            u_k = self.current_dm + self.current_wfs

        # Append current frames to historical registers
        self.c_history.append(self.current_dm.copy())
        self.u_history.append(u_k.copy())

        # Constrain structural memory depth. 
        # Since we look up to (delay_int + 1), we buffer an extra frame for the fractional ceiling.
        max_depth = self.delay_int + max(self.p, self.q) + 2
        if len(self.c_history) > max_depth:
            self.c_history.pop(0)
            self.u_history.pop(0)

        # 2. Run the adaptive update if enough historical frames are buffered
        if len(self.c_history) >= max_depth:
            phi_components = []
            
            # Interpolate and add past command states
            for i in range(1, self.p + 1):
                # Calculate the exact base integer index relative to the end of the list
                target_idx = -self.delay_int - i
                interpolated_c = self._interpolate_history(self.c_history, target_idx)
                phi_components.append(interpolated_c)
                
            # Interpolate and add past/present pseudo-open loop states
            for j in range(0, self.q + 1):
                target_idx = -self.delay_int - j
                interpolated_u = self._interpolate_history(self.u_history, target_idx)
                phi_components.append(interpolated_u)

            # Vector compilation: shape (n_vars, nmodes) -> transpose to (nmodes, n_vars)
            phi = self.xp.stack(phi_components, axis=0).T

            # 3. Vectorized RLS Estimation Math (Equation 22 from Section 4.C)
            P_phi = self.xp.squeeze(self.xp.matmul(self.P, phi[..., None]), axis=-1)
            
            denom = self.lam + self.xp.sum(phi * P_phi, axis=1)
            denom = self.xp.where(denom < 1e-12, 1e-12, denom)

            K = P_phi / denom[:, None]
            error = self.current_wfs

            # Update parameter estimations array (\theta)
            self.theta += K * error[:, None]

            # Update Covariance tracking array: P = (1 / \lambda) * (P - K * phi^T * P)
            K_P_phi = self.xp.matmul(K[..., None], P_phi[..., None, :])
            self.P = (self.P - K_P_phi) / self.lam

            # 4. Map the parameters back into the output IirFilterData object
            if self.p > 0:
                self.iir_filter_data.den[:, 0] = 1.0
                self.iir_filter_data.den[:, 1:] = -self.theta[:, :self.p]
            else:
                self.iir_filter_data.den[:, 0] = 1.0

            self.iir_filter_data.num[:, :] = self.theta[:, self.p:]

    def post_trigger(self):
        super().post_trigger()
        self.iir_filter_data.generation_time = self.current_time