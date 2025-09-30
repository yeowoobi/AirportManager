import numpy as np
from shapely import Point, Polygon

from .node import Node
from .LambertConformal import LambertXY, LambertLatLon, COORD_STANDARD


class Link:

    def __init__(self, name, node1, node2, link_type=None):
        self.name = name
        self.Id = name
        self.node1 = node1
        self.node2 = node2
        self.nodes = [node1, node2]
        self.node_connected = node1.node_connected + node2.node_connected

        if self.node1 in self.node_connected:
            self.node_connected.remove(self.node1)

        if self.node2 in self.node_connected:
            self.node_connected.remove(self.node2)

        self.link_connected = node1.link_connected + node2.link_connected
        if self in self.link_connected:
            self.link_connected.remove(self)
            self.link_connected.remove(self)

        self.type = link_type
        self.polygon = None

    def nodes(self):
        return [self.node1, self.node2]

    def node1_name(self):
        # return self._node1['name']
        return self.node1['name']

    def node2_name(self):
        return self.node2['name']

    def node_list(self):
        return [self.node1['name'], self.node2['name']]

    def node_connected_list(self):
        return [node.name for node in self.node_connected]

    def node1_connected(self):
        return self.node1.connected

    def node2_connected(self):
        return self.node2.connected

    def link_connected_list(self):
        return [link.name for link in self.link_connected]

    def get_length(self):
        return np.linalg.norm(self.get_vector())

    def get_vector(self):

        x_vec = self.node2.x - self.node1.x
        y_vec = self.node2.y - self.node1.y

        return [x_vec, y_vec]

    def get_dir_vector(self):
        return self.get_vector() / self.get_length()

    def create_polygon_xy(self, width, return_=False):
        n1 = self.node1
        n2 = self.node2

        x = np.array([n1.x, n2.x])
        y = np.array([n1.y, n2.y])

        x_vec = x[1] - x[0]
        y_vec = y[1] - y[0]
        n_vec = np.array([x_vec, y_vec]).transpose()
        vec_mag = np.linalg.norm(n_vec, 2)

        n_vec = n_vec / np.array([vec_mag, vec_mag]).transpose()
        o_vec = (np.array([y_vec, -x_vec]) / np.array([vec_mag, vec_mag])).transpose()

        x_p0 = np.zeros(x.shape)
        y_p0 = np.zeros(x.shape)
        x_m0 = np.zeros(x.shape)
        y_m0 = np.zeros(x.shape)

        x_p = np.zeros(x.shape)
        y_p = np.zeros(x.shape)
        x_m = np.zeros(x.shape)
        y_m = np.zeros(x.shape)

        x_p0[:] = x[:] + width * np.array([o_vec[0], o_vec[0]]).transpose()
        x_p[0] = x_p0[0] - width * n_vec[0]
        x_p[1] = x_p0[1] + width * n_vec[0]

        y_p0[:] = y[:] + width * np.array([o_vec[1], o_vec[1]]).transpose()
        y_p[0] = y_p0[0] - width * n_vec[1]
        y_p[1] = y_p0[1] + width * n_vec[1]

        x_m0[:] = x[:] - width * np.array([o_vec[0], o_vec[0]]).transpose()
        x_m[0] = x_m0[0] - width * n_vec[0]
        x_m[1] = x_m0[1] + width * n_vec[0]

        y_m0[:] = y[:] - width * np.array([o_vec[1], o_vec[1]]).transpose()
        y_m[0] = y_m0[0] - width * n_vec[1]
        y_m[1] = y_m0[1] + width * n_vec[1]

        p1 = np.column_stack((x_p[0], y_p[0]))
        p2 = np.column_stack((x_p[1], y_p[1]))
        p3 = np.column_stack((x_m[1], y_m[1]))
        p4 = np.column_stack((x_m[0], y_m[0]))
        polygon = Polygon([p1[0], p2[0], p3[0], p4[0]])

        self.polygon = polygon

        if return_:
            return polygon

    def create_polygon_xy2(self, width, return_=False):
        n1 = self.node1
        n2 = self.node2

        x = np.array([n1.x, n2.x])
        y = np.array([n1.y, n2.y])

        x_vec = x[1] - x[0]
        y_vec = y[1] - y[0]
        n_vec = np.array([x_vec, y_vec]).transpose()
        vec_mag = np.linalg.norm(n_vec, 2)

        n_vec = n_vec / np.array([vec_mag, vec_mag]).transpose()
        o_vec = (np.array([y_vec, -x_vec]) / np.array([vec_mag, vec_mag])).transpose()

        x_p0 = np.zeros(x.shape)
        y_p0 = np.zeros(x.shape)
        x_m0 = np.zeros(x.shape)
        y_m0 = np.zeros(x.shape)

        x_p = np.zeros(x.shape)
        y_p = np.zeros(x.shape)
        x_m = np.zeros(x.shape)
        y_m = np.zeros(x.shape)

        x_p0[:] = x[:] + width * np.array([o_vec[0], o_vec[0]]).transpose()
        x_p[0] = x_p0[0]
        x_p[1] = x_p0[1]

        y_p0[:] = y[:] + width * np.array([o_vec[1], o_vec[1]]).transpose()
        y_p[0] = y_p0[0]
        y_p[1] = y_p0[1]

        x_m0[:] = x[:] - width * np.array([o_vec[0], o_vec[0]]).transpose()
        x_m[0] = x_m0[0]
        x_m[1] = x_m0[1]

        y_m0[:] = y[:] - width * np.array([o_vec[1], o_vec[1]]).transpose()
        y_m[0] = y_m0[0]
        y_m[1] = y_m0[1]

        p1 = np.column_stack((x_p[0], y_p[0]))
        p2 = np.column_stack((x_p[1], y_p[1]))
        p3 = np.column_stack((x_m[1], y_m[1]))
        p4 = np.column_stack((x_m[0], y_m[0]))
        polygon = Polygon([p1[0], p2[0], p3[0], p4[0]])

        self.polygon = polygon

        if return_:
            return polygon

    def convert_polygon_latlon(self):

        coord_standard = self.node1.coord_standard

        p1 = self.polygon.exterior.coords[0]
        p2 = self.polygon.exterior.coords[1]
        p3 = self.polygon.exterior.coords[2]
        p4 = self.polygon.exterior.coords[3]

        p1 = LambertLatLon(p1, coord_standard)
        p2 = LambertLatLon(p2, coord_standard)
        p3 = LambertLatLon(p3, coord_standard)
        p4 = LambertLatLon(p4, coord_standard)

        return [p1, p2, p3, p4]

    def update(self):
        self.name = self.name
        self.node1 = self.node1
        self.node2 = self.node2
        self.node_connected = self.node1.node_connected + self.node2.node_connected
        self.link_connected = self.node1.link_connected + self.node2.link_connected
        if self in self.link_connected:
            self.link_connected.remove(self)
            self.link_connected.remove(self)
        else:
            print("no")

    def __getattr__(self, item):
        return 'no attribute'


if __name__ == "__main__":

    A = Node('n1', 34, 127)
    B = Node('n2', 37, 128)

    A.update(node_connected=['B', 'C'])

    C = Link('TT', A, B)





