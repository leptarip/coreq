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

class Timings:
    def __init__(self):
        self.value = 0
        self.count = 0

    def get_avg(self) -> float:
        return round(self.value / self.count, 5)

    def add_value(self, value):
        self.value = self.value + value
        self.count = self.count + 1

    def get_new_avg(self, value) -> float:
        self.add_value(value)
        return self.get_avg()