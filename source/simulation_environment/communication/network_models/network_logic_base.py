#!/usr/bin/env python
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


class NetworkLogicBase:

    def __init__(self):
        pass

    def compute_De2e(self, sim_time=None) -> int:
        pass

    def should_drop_packet(self, sim_time=None) -> bool:
        """
        Determines whether to discard a packet based on a given probability.

        This function simulates a random event with a probability `p`. It's
        useful in network simulations for modeling packet loss.

        Returns:
            True if the packet should be discarded, False otherwise.

        Raises:
            ValueError: If self.drop_rate is not in the valid probability range [0, 1].
        """
        pass
