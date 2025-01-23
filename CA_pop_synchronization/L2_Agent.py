import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from util import wrap_angle, saturate_soft, saturate_hard, bump_fun


# L2 AGENT ---------------------------------------
class L2Agent:
    def __init__(self,
                 dt                  : float = 0.01     , # sampling time                                               , used to update the dynamic threshold
                 thres_min           : float = 0        , # minimum value [rad] the threshold can take
                 thres_max           : float = np.pi / 2, # maximum value [rad] the threshold can take
                 speed_thres         : float = 4        , # time constant                                               : 1/speed_thres                       , settling time to 1% = 4.6/speed_thres
                 damping_factor      : float = 4        , # dynamics of thres is damping_factor times slower when growing w.r.t. when decreasing
                 mode_saturation     : str   = "soft"   , # use 'soft' or 'hard' saturation for the blending
                 p_norm              : int   = 20       , # coefficient to approximate a saturation in the 'soft' method
                 virtual_agent_index : int   = 0        , # index of the l2
                 ) -> None:
        self.dt              = dt
        self.thres           = thres_max
        self.thres_min       = thres_min
        self.thres_max       = thres_max
        self.speed_thres     = speed_thres
        self.damping_factor  = damping_factor
        self.wrapping_domain = '-pi to pi'
        if   mode_saturation == "soft":  self.saturate = partial(saturate_soft, p=p_norm)
        elif mode_saturation == "hard":  self.saturate = saturate_hard
        else:  raise ValueError(f"Invalid value for mode_saturation: {mode_saturation}")
        self.virtual_agent_index = virtual_agent_index


    def compute_phase_L2(self, phase_L0: float, phase_ideal: float) -> float:
        error = wrap_angle(phase_ideal - phase_L0, self.wrapping_domain)

        # Update the dynamic threshold
        steady_state_val = bump_fun(error / self.thres_max) * (self.thres_max - self.thres_min) + self.thres_min
        thres_dot = - self.speed_thres * (self.thres - steady_state_val)
        if self.thres <= steady_state_val:
            thres_dot = thres_dot / self.damping_factor
        self.thres = self.thres + self.dt * thres_dot

        # Compute L2's phase
        saturated_alteration = self.saturate(error, self.thres)
        return wrap_angle(phase_L0 + saturated_alteration, self.wrapping_domain)
