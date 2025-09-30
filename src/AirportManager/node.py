from shapely import Point

from .LambertConformal import LambertXY, LambertLatLon


class Node:

    # def __init__(self, Name, Latitude, Longitude, Altitude, NodeType):

    def __init__(self, name, lat, lon, **kwargs):

        # Essential Attributes
        kwargs.setdefault('name', name)
        kwargs.setdefault('lat', lat)
        kwargs.setdefault('lon', lon)
        kwargs.setdefault('coord_standard', None)

        # Additional Attributes
        kwargs.setdefault('alt', 0)
        kwargs.setdefault('type', 'unknown')
        kwargs.setdefault('node_connected', [])
        kwargs.setdefault('link_connected', [])

        self._kwargs = kwargs
        self.rad = None
        self.circle = None

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.x, self.y = LambertXY([lat, lon], self.coord_standard)

    def __getitem__(self, item):
        return self._kwargs[item]

    def __getattr__(self, item):
        return 'no attribute'

    def node_connected_list(self):
        return [node.name for node in self.node_connected]

    def link_connected_list(self):
        return [link.name for link in self.link_connected]

    def update(self, **kwargs):
        self._kwargs.update(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.x, self.y = LambertXY([self.lat, self.lon], self.coord_standard)

    def create_circle(self, rad, return_=False):
        circle = Point(self.x, self.y).buffer(rad)
        self.rad = rad
        self.circle = circle
        if return_:
            return circle

    def Point(self):
        return Point(self.x, self.y)

    # def convert_circle_latlon(self):


if __name__ == '__main__':

    print("Hello World")

