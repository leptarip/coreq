# Copyright (c) 2025 278097159+leptarip@users.noreply.github.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np

class KalmanFilter:
    def __init__(self, time_to_sec_factor):
        self.x = np.array([[0.0], [0.0], [0.0]])  # [position, velocity, acceleration]
        self.P = np.eye(3) * 1.0                   # Initial covariance
        self.Q = np.eye(3) * 0.01                  # Process noise covariance
        self.Rv = np.array([[0.5]])                # Velocity noise covariance
        self.Hv = np.array([[0, 1, 0]])             # Velocity-only observation
        self.time_factor = time_to_sec_factor      # to transform input delta time in seconds

    def predict(self, delta_t):
        F = np.array([
            [1, delta_t, 0.5 * delta_t ** 2],
            [0, 1, delta_t],
            [0, 0, 1]
        ])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update_velocity(self, z):
        y = z - self.Hv @ self.x
        S = self.Hv @ self.P @ self.Hv.T + self.Rv
        K = self.P @ self.Hv.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ self.Hv) @ self.P

    def get_state(self):
        return self.x

    def get_predicted_acc(self) -> float:
        return self.x[2][0]

    def predict_acc(self, delta_t, velocity:float):
        self.predict(delta_t * self.time_factor)
        self.update_velocity(np.array([[velocity]]))
        return self.get_predicted_acc()
