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
import os

# Log lines are accumulated in a list and joined on write. Appending to a module
# global with "+=" is quadratic here: CPython's in-place string concatenation
# fast path only covers STORE_FAST/STORE_DEREF/STORE_NAME, not STORE_GLOBAL, so
# every append copied the whole accumulated log.
log_parts = []


class PrintL:
    STD_OUT = 0
    FILE = 1
    BOTH = 2
    IN_MEM = 3
    STD_OUT_AND_MEM = 4

    def __init__(self, tag, enabled):
        self.tag = tag
        self.enabled = enabled

    @staticmethod
    def print_to_std(message):
        print(message)

    def print_to_file(self, message):
        pass

    def send_to_redis(self, message):
        pass

    def to_print(self, header="", message="", force=False, where=IN_MEM):
        if self.enabled or force:
            if header != "":
                formatted = "[{0}] <{1}> {2}".format(self.tag, header, message)
            else:
                formatted = "[{0}] {1}".format(self.tag, message)

            if where == PrintL.STD_OUT:
                self.print_to_std(formatted)
            elif where == PrintL.FILE:
                pass
            elif where == PrintL.IN_MEM:
                log_parts.append("{0}{1}".format(formatted, os.linesep))
            elif where == PrintL.STD_OUT_AND_MEM:
                log_parts.append("{0}{1}".format(formatted, os.linesep))
                self.print_to_std(formatted)
            else:
                pass

    @staticmethod
    def write_to_file(filename: str):

        with open(filename, 'w') as f:
            f.write("".join(log_parts))

        del log_parts[:]
