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
from enum import IntEnum

import shapely
from source.simulation_environment.agents.traj_shape import RoutePolygon

import plotly.graph_objects as go

class CriticalRegion:
    MARGIN = 0.3

    class Position(IntEnum):
        BEFORE_CR = 0
        AFTER_CR = 10
        INSIDE_CR = 20
        UNKNOWN = -10


    #        critical region
    #          |--------|
    # path--->>|*cn     |*cf -->>path
    #          |________|
    #
    def __init__(self, route_poly: RoutePolygon):
        self.cn = None
        self.cf = None
        self.cr_path = None
        self.cn_orig_d = -1
        self.cf_orig_d = -1
        self.route_poly = route_poly


    def compute_critical_points(self, other_traj_orig: shapely.Polygon):
        inter_rx = shapely.intersection(self.route_poly.path_rx, other_traj_orig.boundary)
        inter_lx = shapely.intersection(self.route_poly.path_lx, other_traj_orig.boundary)
        inter_c = shapely.intersection(self.route_poly.path_c, other_traj_orig.boundary)

        if ((inter_rx is None or inter_rx.is_empty) or
                (inter_lx is None or inter_lx.is_empty) or
                (inter_c is None or inter_c.is_empty)):
            return

        # we assume that the intersections are LineStrings
        if inter_rx.geom_type != "MultiPoint" or inter_lx.geom_type != "MultiPoint" or inter_c.geom_type != "MultiPoint":
            print("Problem critical region computation")
            raise Exception("Problem critical region computation")

        d_c = shapely.line_locate_point(self.route_poly.path_c, inter_c.geoms[0])
        d_lx = shapely.line_locate_point(self.route_poly.path_lx, inter_lx.geoms[0])
        d_rx = shapely.line_locate_point(self.route_poly.path_rx, inter_rx.geoms[0])

        temp = [(d_c, self.route_poly.path_c, inter_c.geoms),
                (d_lx, self.route_poly.path_lx, inter_lx.geoms),
                (d_rx, self.route_poly.path_rx, inter_rx.geoms)]
        # res[0] contains the min distance, res[1] contains the associated path, res[2] the associated intersection
        res = min(temp, key=lambda x: x[0])

        self.cr_path = res[1]

        # multipoints are not given in order! (Checked with plot)
        # find who is c_near and cn_orig_d  AND c_far and cf_orig_d
        d2 = shapely.line_locate_point(self.cr_path, res[2][1])
        # temp is [(d1, point1), (d2, point2)]
        temp = [(res[0], res[2][0]), (d2, res[2][1])]
        temp.sort(key=lambda x: x[0])
        self.cn = temp[0][1]
        self.cn_orig_d = temp[0][0] - CriticalRegion.MARGIN
        self.cf = temp[1][1]
        self.cf_orig_d = temp[1][0] + CriticalRegion.MARGIN
        #self.plot_regions(other_traj_orig)


    def plot_regions(self, other_poly: shapely.Polygon):

        x, y = self.route_poly.polygon.boundary.xy
        trace_0 = go.Scatter(
            x=x.tolist(),
            y=y.tolist(),
            mode="lines",
            name="poly",
            line={"color": "green"},
        )

        x, y = self.route_poly.path_c.xy
        trace_1 = go.Scatter(
            x=x.tolist(),
            y=y.tolist(),
            mode="lines",
            name="path_c",
            line={"color": "red"},
        )
        x, y = self.route_poly.path_rx.xy
        trace_2 = go.Scatter(
            x=x.tolist(),
            y=y.tolist(),
            mode="lines",
            name="path_rx",
            line={"color": "orange"}
        )

        x, y = self.route_poly.path_lx.xy
        trace_3 = go.Scatter(
            x=x.tolist(),
            y=y.tolist(),
            mode="lines",
            name="path_lx",
            line={"color": "yellow"},
        )

        x, y = other_poly.boundary.xy
        trace_4 = go.Scatter(
            x=x.tolist(),
            y=y.tolist(),
            mode="lines",
            name="other poly",
            line={"color": "blue"},
        )

        trace_5 = go.Scatter(
            x=[self.cf.x],
            y=[self.cf.y],
            mode="markers",
            name="cf",
            marker={"color": "brown", "size":8}
        )

        trace_6 = go.Scatter(
            x=[self.cn.x],
            y=[self.cn.y],
            mode="markers",
            name="cn",
            marker={"color": "black", "size":8}
        )

        cf_margin = self.cr_path.interpolate(self.cf_orig_d + CriticalRegion.MARGIN)
        trace_7 = go.Scatter(
            x=[cf_margin.x],
            y=[cf_margin.y],
            mode="markers",
            name="cf + {0} margin".format(CriticalRegion.MARGIN),
            marker={"color": "gold", "size": 8}
        )

        cn_margin = self.cr_path.interpolate(self.cn_orig_d - CriticalRegion.MARGIN)
        trace_8 = go.Scatter(
            x=[cn_margin.x],
            y=[cn_margin.y],
            mode="markers",
            name="cn + {0} margin".format(CriticalRegion.MARGIN),
            marker={"color": "gold", "size": 8}
        )

        traces = [trace_0, trace_1, trace_2, trace_3, trace_4, trace_5, trace_6, trace_7, trace_8]

        layout = go.Layout(
            title="test",
            hovermode='x unified'
        )

        fig = go.Figure(data=traces, layout=layout)
        fig.show()