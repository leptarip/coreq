#!/usr/bin/env python
# Copyright (c) 2023-2025 278097159+leptarip@users.noreply.github.com
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

DEFAULT_TARGET_A = 3.5  # [m/s]
DEFAULT_VEHICLE_LENGTH = 4.5 # [m]

## see https://intel.github.io/ad-rss-lib/ad_rss/Appendix-ParameterDiscussion/
MAX_BRAKE = 6  # [m/s²]
MIN_BRAKE = 2  # [m/s²]
MAX_RESPONSE_TIME = 1  # [s]
MIN_RESPONSE_TIME = 0.5  # [s]
MAX_ACC = 4  # [m/s²]
MITER_KM_RATIO = 3.6
