import math
import numpy as np
from sklearn.decomposition import PCA

class Phase_estimator_pca_online:
    def __init__(self, window_pca, interval_between_pca):
        self.trajectory = []
        self.time_instants = []

        self.pca = PCA(n_components=1)
        self.pca_direction = 1  # 1 or -1; used to preserve existing direction of the pca when recomputed
        self.window_pca = window_pca  # [s]
        self.interval_between_pca = interval_between_pca  # [s]

        self.pos_princ_comp = 0
        self.pos_princ_comp_prev = 0
        self.vel_princ_comp = 0
        self.vel_princ_comp_prev = 0
        self.acc_princ_comp = 0
        self.pos_princ_comp_offset = 0

        self.phase = 0
        self.is_first_estimation = True
        self.time_last_pca = None
        self.is_first_pca_computed = False

        self.amplitude_vel_p = 1
        self.amplitude_vel_n = 1
        self.amplitude_pos_n = 1
        self.amplitude_pos_p = 1

        self.n_samples_initial_direction_pca = 20  # Consider the first samples to determine which is the direction of the PCA
        self.start_from_extended_position = False  # For consistency, all instances of the class should have this equal. True means that the person starts with the arm extended position, 

    def estimate_phase(self, position, current_time):
        self.time_instants.append(current_time)
        current_time = self.time_instants[-1]
        self.trajectory.append(position)

        if self.is_first_estimation:
            self.time_last_pca = current_time
            self.is_first_estimation = False

        if not self.is_first_pca_computed:
            if current_time - self.time_last_pca >= self.window_pca:
                self.time_last_pca = current_time
                self.compute_PCA()
                self.is_first_pca_computed = True
                
        else:
            if current_time - self.time_last_pca >= self.interval_between_pca:
                self.time_last_pca = current_time
                self.compute_PCA()

        if self.is_first_pca_computed:  self.update_phase()

        return self.phase

    def compute_PCA(self):
        idx = 1
        is_found = False
        while not is_found:
            if self.time_instants[-1] - self.time_instants[-idx] >= self.window_pca:
                is_found = True
            else:
                idx += 1

        if self.is_first_pca_computed:  prev_pca_vec = self.pca.components_[0]

        self.pca.fit(np.array(self.trajectory)[-idx:, :])

        if self.is_first_pca_computed: 
            pca_vec = self.pca.components_[0]
            if np.dot(pca_vec, prev_pca_vec) < 0:  self.pca_direction = -1 * self.pca_direction  # TODO should prev_pca_vec be self.prev_pca_vec?
        else: # The first time you compute the PCA
            n_samples = min(len(self.trajectory), self.n_samples_initial_direction_pca)     # Take the first n_samples of the trajectory
            velocity = np.empty((self.trajectory[0].shape[0], n_samples-1))
            for i in range(n_samples-1): # Compute the velocity in this n_samples time instants
                velocity[:, i] = (self.trajectory[i+1]-self.trajectory[i])/(self.time_instants[i+1]-self.time_instants[i])
            avg_vel = np.mean(velocity, axis = 1) # Compute the average velocity in the first n_samples time instants
            if self.start_from_extended_position == True:    # If all the people start from the arm extended position
                if np.dot(self.pca.components_[0], avg_vel) > 0:  self.pca.components_[0] = -1 * self.pca.components_[0] # Rotate the pca vector when PCA and avg_vel are concordant
            else:   # If all the people start from the arm flexed position
                if np.dot(self.pca.components_[0], avg_vel) < 0:  self.pca.components_[0] = -1 * self.pca.components_[0] 


        score = self.pca_direction * np.reshape(self.pca.transform(np.array(self.trajectory)[-idx:, :]), -1)
        max_score = max(score)
        min_score = min(score)
        self.pos_princ_comp_offset = (max_score + min_score) / 2

        array_1d = np.array(np.array(self.trajectory)[-4, :])
        array_2d = array_1d.reshape(1, -1)  # Reshape to (1, 3)
        score = self.pca_direction * self.pca.transform(array_2d)
        self.pos_princ_comp = score[0, 0] - self.pos_princ_comp_offset

        array_1d = np.array(np.array(self.trajectory)[-5, :])
        array_2d = array_1d.reshape(1, -1)  # Reshape to (1, 3)
        score = self.pca_direction * self.pca.transform(array_2d)
        self.pos_princ_comp_prev = score[0, 0] - self.pos_princ_comp_offset

        sampling_time = self.time_instants[-5] - self.time_instants[-4]
        self.vel_princ_comp_prev = (self.pos_princ_comp - self.pos_princ_comp_prev) / sampling_time

        array_1d = np.array(np.array(self.trajectory)[-2, :])
        array_2d = array_1d.reshape(1, -1)  # Reshape to (1, 3)
        score = self.pca_direction * self.pca.transform(array_2d)
        self.pos_princ_comp = score[0, 0] - self.pos_princ_comp_offset

        array_1d = np.array(np.array(self.trajectory)[-3, :])
        array_2d = array_1d.reshape(1, -1)  # Reshape to (1, 3)
        score = self.pca_direction * self.pca.transform(array_2d)
        self.pos_princ_comp_prev = score[0, 0] - self.pos_princ_comp_offset

        sampling_time = self.time_instants[-2] - self.time_instants[-3]
        self.vel_princ_comp = (self.pos_princ_comp - self.pos_princ_comp_prev) / sampling_time

    def update_phase(self):
        array_1d = np.array(np.array(self.trajectory)[-1, :])
        array_2d = array_1d.reshape(1, -1)  # Reshape to (1, 3)
        score = self.pca_direction * self.pca.transform(array_2d)

        self.pos_princ_comp_prev = self.pos_princ_comp
        self.vel_princ_comp_prev = self.vel_princ_comp

        sampling_time = self.time_instants[-1] - self.time_instants[-2]

        self.pos_princ_comp = score[0, 0] - self.pos_princ_comp_offset
        self.vel_princ_comp = (self.pos_princ_comp - self.pos_princ_comp_prev) / sampling_time
        self.acc_princ_comp = (self.vel_princ_comp - self.vel_princ_comp_prev) / sampling_time

        if self.pos_princ_comp_prev < 0 and self.pos_princ_comp >= 0 and self.vel_princ_comp > 0:
            self.amplitude_vel_p = abs(self.vel_princ_comp)
        if self.pos_princ_comp_prev >= 0 and self.pos_princ_comp < 0 and self.vel_princ_comp < 0:
            self.amplitude_vel_n = abs(self.vel_princ_comp)
        if self.vel_princ_comp_prev >= 0 and self.vel_princ_comp < 0 and self.acc_princ_comp < 0:
            self.amplitude_pos_p = abs(self.pos_princ_comp)
        if self.vel_princ_comp_prev < 0 and self.vel_princ_comp >= 0 and self.acc_princ_comp > 0:
            self.amplitude_pos_n = abs(self.pos_princ_comp)
        if self.pos_princ_comp >= 0:
            position_normalized = self.pos_princ_comp / self.amplitude_pos_p
        else:
            position_normalized = self.pos_princ_comp / self.amplitude_pos_n
        if self.vel_princ_comp >= 0:
            velocity_normalized = self.vel_princ_comp / self.amplitude_vel_p
        else:
            velocity_normalized = self.vel_princ_comp / self.amplitude_vel_n

        self.phase = math.atan2(-velocity_normalized, position_normalized)

        # self.phase = np.mod(self.phase, 2*np.pi)  # wrap to [0, 2pi)