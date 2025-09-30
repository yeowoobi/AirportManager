import numpy as np

R0 = 6371 * 10 ** 3


def sind(degree):
    return np.sin(np.deg2rad(degree))


def cosd(degree):
    return np.cos(np.deg2rad(degree))


def tand(degree):
    return np.tan(np.deg2rad(degree))


def asind(value):
    return np.rad2deg(np.arcsin(value))


def acosd(value):
    return np.rad2deg(np.arccos(value))


def atand(value):
    return np.rad2deg(np.arctan(value))


def atan2d(value1, value2):
    return np.rad2deg(np.arctan2(value1, value2))


def node_latlon(df):
    Node1_lat = list(df.Node1Latitude)
    Node2_lat = list(df.Node2Latitude)
    lat = np.array([Node1_lat, Node2_lat]).transpose()

    Node1_lon = list(df.Node1Longitude)
    Node2_lon = list(df.Node2Longitude)
    lon = np.array([Node1_lon, Node2_lon]).transpose()

    return lat, lon


# South Korea
COORD_STANDARD = {'KOR': {'lat_min': 33.0, 'lon_min': 124.5, 'lat_max': 38.9,  'lon_max': 132.0},
                  'RKSI': {'lat_min': 37.44148968, 'lon_min': 126.4135517, 'lat_max': 37.49238333, 'lon_max': 126.4743333}
                  }


def LambertXY(coord, key='RKSI'):
    if key is None:
        coord_standard = COORD_STANDARD['KOR']
    elif type(key) is dict:
        coord_standard = key
    elif key in COORD_STANDARD:
        coord_standard = COORD_STANDARD[key]
    else:
        return 'Invalid Key'

    lat_min = coord_standard['lat_min']
    lon_min = coord_standard['lon_min']
    lat_max = coord_standard['lat_max']
    lon_max = coord_standard['lon_max']

    lat = coord[0]
    lon = coord[1]

    x0 = 0
    y0 = 0

    psi = 90 - lat

    lat0 = lat_min
    lon0 = (lon_min + lon_max) / 2
    psi0 = 90 - lat0

    psi1 = lat_min + (1 / 6) * (lat_max - lat_min)
    psi2 = lat_min + (5 / 6) * (lat_max - lat_min)

    k = np.log(sind(psi1) / sind(psi2)) / np.log(tand(psi1 / 2) / tand(psi2 / 2))
    C = R0 * sind(psi1) / k / tand(psi1 / 2) ** k
    R_psi = C * tand(psi / 2) ** k
    R_psi0 = C * tand(psi0 / 2) ** k

    x = x0 + R_psi * np.sin(k * np.deg2rad(lon - lon0))
    y = y0 + R_psi0 - R_psi * np.cos(k * np.deg2rad(lon - lon0))

    return x, y


def LambertLatLon(coord, key=None, inverse=False):
    if key is None:
        coord_standard = COORD_STANDARD['KOR']
    elif type(key) is dict:
        coord_standard = key
    elif key in COORD_STANDARD:
        coord_standard = COORD_STANDARD[key]
    else:
        return 'Invalid Key'

    lat_min = coord_standard['lat_min']
    lon_min = coord_standard['lon_min']
    lat_max = coord_standard['lat_max']
    lon_max = coord_standard['lon_max']

    x = coord[0]
    y = coord[1]

    x0 = 0
    y0 = 0

    x = np.array(x)
    y = np.array(y)

    lat0 = lat_min
    lon0 = (lon_min + lon_max) / 2
    psi0 = 90 - lat0

    psi1 = lat_min + (1 / 6) * (lat_max - lat_min)
    psi2 = lat_min + (5 / 6) * (lat_max - lat_min)

    k = np.log(sind(psi1) / sind(psi2)) / np.log(tand(psi1 / 2) / tand(psi2 / 2))
    C = R0 * sind(psi1) / k / tand(psi1 / 2) ** k
    R_psi0 = C * tand(psi0 / 2) ** k
    lon = lon0 + (1 / k) * atan2d(x - x0, y0 + R_psi0 - y)

    if x != x0:
        num = x - x0
        den = C * np.sin(k * np.deg2rad((lon - lon0)))
        psi = 2 * atand((num / den) ** (1 / k))
    else:
        num = y0 - y + R_psi0
        den = C
        psi = 2 * atand((num / den) ** (1 / k))

    lat = 90 - psi

    if inverse:
        return lon, lat

    else:
        return lat, lon


def LambertXY_series(coord, key='RKSI'):
    if key is None:
        coord_standard = COORD_STANDARD['KOR']
    elif type(key) is dict:
        coord_standard = key
    elif key in COORD_STANDARD:
        coord_standard = COORD_STANDARD[key]
    else:
        return 'Invalid Key'

    lat_min = coord_standard['lat_min']
    lon_min = coord_standard['lon_min']
    lat_max = coord_standard['lat_max']
    lon_max = coord_standard['lon_max']

    lat = coord[0]
    lon = coord[1]

    x0 = 0
    y0 = 0

    psi = 90 - lat

    lat0 = lat_min
    lon0 = (lon_min + lon_max) / 2
    psi0 = 90 - lat0

    psi1 = lat_min + (1 / 6) * (lat_max - lat_min)
    psi2 = lat_min + (5 / 6) * (lat_max - lat_min)

    k = np.log(sind(psi1) / sind(psi2)) / np.log(tand(psi1 / 2) / tand(psi2 / 2))
    C = R0 * sind(psi1) / k / tand(psi1 / 2) ** k
    R_psi = C * tand(psi / 2) ** k
    R_psi0 = C * tand(psi0 / 2) ** k

    x = x0 + R_psi * np.sin(k * np.deg2rad(lon - lon0))
    y = y0 + R_psi0 - R_psi * np.cos(k * np.deg2rad(lon - lon0))

    return x, y


def LambertLatLon_series(coord, key=None):
    if key is None:
        coord_standard = COORD_STANDARD['KOR']
    elif type(key) is dict:
        coord_standard = key
    elif key in COORD_STANDARD:
        coord_standard = COORD_STANDARD[key]
    else:
        return 'Invalid Key'

    lat_min = coord_standard['lat_min']
    lon_min = coord_standard['lon_min']
    lat_max = coord_standard['lat_max']
    lon_max = coord_standard['lon_max']

    x = coord[0]
    y = coord[1]

    x0 = 0
    y0 = 0

    x = np.array(x)
    y = np.array(y)

    lat0 = lat_min
    lon0 = (lon_min + lon_max) / 2
    psi0 = 90 - lat0

    psi1 = lat_min + (1 / 6) * (lat_max - lat_min)
    psi2 = lat_min + (5 / 6) * (lat_max - lat_min)

    k = np.log(sind(psi1) / sind(psi2)) / np.log(tand(psi1 / 2) / tand(psi2 / 2))
    C = R0 * sind(psi1) / k / tand(psi1 / 2) ** k
    R_psi0 = C * tand(psi0 / 2) ** k
    lon = lon0 + (1 / k) * atan2d(x - x0, y0 + R_psi0 - y)

    if x.ndim < 2:
        n = len(x)
        psi = np.zeros(n)
        for i in range(n):
            if (x != x0).all():
                num = x[i] - x0
                den = C * np.sin(k * np.deg2rad((lon[i] - lon0)))
                psi[i] = 2 * atand((num / den) ** (1 / k))
            else:
                num = y0 - y[i] + R_psi0
                den = C
                psi[i] = 2 * atand((num / den) ** (1 / k))
    else:
        [n, l] = x.shape
        psi = np.zeros((n, l))
        for i in range(n):
            for j in range(l):
                if (x != x0).all():
                    num = x[i, j] - x0
                    den = C * np.sin(k * np.deg2rad((lon[i, j] - lon0)))
                    psi[i, j] = 2 * atand((num / den) ** (1 / k))
                else:
                    num = y0 - y[i, j] + R_psi0
                    den = C
                    psi[i, j] = 2 * atand((num / den) ^ (1 / k))

    lat = 90 - psi

    return lat, lon


def get_distance(coord1, coord2, key='RKSI'):

    xy1 = LambertXY(coord1, key)
    xy2 = LambertXY(coord2, key)

    x_vec = xy1[0] - xy2[0]
    y_vec = xy1[1] - xy2[1]

    return np.linalg.norm([x_vec, y_vec])


def get_distanceXY(coord1, coord2):
    x_vec = coord1[0] - coord2[0]
    y_vec = coord1[1] - coord2[1]

    return np.linalg.norm([x_vec, y_vec])


if __name__ == "__main__":

    LambertLatLon((-243.69418123381215, -461.0391463134438), key='RKSI')

