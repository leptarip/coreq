# Copyright (c) 2026 278097159+leptarip@users.noreply.github.com
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

# command to compile cython:
# python setup_cython.py build_ext --inplace

from distutils.core import setup, Extension
from Cython.Build import cythonize

ext_modules = [

    Extension(
        name="source.simulation_environment.cython.matrix_helper_cy",
        sources=["source/simulation_environment/cython/matrix_helper_cy.pyx"],
        include_dirs=['.']
    ),
    Extension(
        name="source.simulation_environment.cython.basic_agent_cy",
        sources=["source/simulation_environment/cython/basic_agent_cy.pyx"],
        include_dirs=['.']
    ),
    Extension(
        name="source.simulation_environment.cython.controller_cy",
        sources=["source/simulation_environment/cython/controller_cy.pyx"],
        include_dirs=['.']
    ),
    Extension(
        name="source.simulation_environment.cython.global_route_planner_cy",
        sources=["source/simulation_environment/cython/global_route_planner_cy.pyx"],
        include_dirs=['.']
    ),
    Extension(
        name="source.simulation_environment.cython.local_planner_cy",
        sources=["source/simulation_environment/cython/local_planner_cy.pyx"],
        include_dirs=['.']
    ),
    Extension(
        name="source.simulation_environment.cython.misc_cy",
        sources=["source/simulation_environment/cython/misc_cy.pyx"],
        include_dirs=['.']
    ),
    Extension(
        name="source.simulation_environment.cython.perception_2d_gt_cy",
        sources=["source/simulation_environment/cython/perception_2d_gt_cy.pyx"],
        include_dirs=['.']
    )
]

setup(
    ext_modules=cythonize(ext_modules, annotate=True)
)

print("********CYTHON COMPLETE******")
