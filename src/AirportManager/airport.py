import datetime
import json

import itertools
from collections import Counter

import pandas as pd
import numpy as np

from shapely import Point, Polygon, points

import folium
from folium.plugins import AntPath

import pydeck as pdk
from seaborn import color_palette


from . import util
from .aircraft import Aircraft
from .node import Node
from .link import Link
from .LambertConformal import LambertXY, LambertLatLon

# from aircraft import Aircraft
# from LambertConformal import COORD_STANDARD

airport_path = {
    'RKSI': "../data/RKSI22_NodeLink_old.csv"
    # 'RKSI': "./RKSI22_NodeLink_new.csv"
}


def return_path(airport_code):
    return airport_path[airport_code]


# 공항 객체 생성
class Airport:
    def __init__(self, csv_path='RKSI', Id='RKSI'):

        # df = pd.read_csv(csv_path)
        # 공항 데이터 읽어오기

        if csv_path.endswith('.csv'):
            csv_path = csv_path

        else:
            csv_path = airport_path[csv_path]

        self.df = pd.read_csv(csv_path)
        self.Id = Id
        self.path = csv_path

        # Mean이 column에 존재할 경우, 참조할 dictionary 생성
        if 'Mean' in self.df.columns:
            self.ref = self.df[['Mean', 'StdDev']].set_index(self.df['LinkId']).transpose().to_dict()
        else:
            self.ref = None

        self.df = pd.read_csv(csv_path).iloc[:, 0:13]

        self.Polygons = None
        self.Circles = None

        self.RunwayPolygons = None

        #
        # if 'Mean' in self.df.columns:
        #     self.ref = self.df[['Mean', 'StdDev']].set_index(self.df['LinkId']).transpose().to_dict()

        self.lat_center = np.array(list(self.df['Node1Latitude']) + list(self.df['Node2Latitude'])).mean()
        self.lon_center = np.array(list(self.df['Node1Longitude']) + list(self.df['Node2Longitude'])).mean()

        dict_minmax = {'lat_min': pd.concat([self.df.Node1Latitude, self.df.Node2Latitude]).min(),
                       'lat_max': pd.concat([self.df.Node1Latitude, self.df.Node2Latitude]).max(),
                       'lon_min': pd.concat([self.df.Node1Longitude, self.df.Node2Longitude]).min(),
                       'lon_max': pd.concat([self.df.Node1Longitude, self.df.Node2Longitude]).max()}

        LinkId = self.df['LinkId']
        Node1Id = self.df['Node1Name']
        Node2Id = self.df['Node2Name']
        LinkType = self.df['LinkType']

        node1 = self.df.iloc[:, 2:7]
        node2 = self.df.iloc[:, 7:len(self.df.columns) - 1]
        node1.columns = node2.columns = ['Name', 'Longitude', 'Latitude', 'Altitude', 'Type']
        df_node = pd.concat([node1, node2]).drop_duplicates('Name').reset_index(drop=True)
        self.coord_standard = dict_minmax
        self.nodes_df = df_node

        b1 = LambertXY([dict_minmax['lat_min'], dict_minmax['lon_min']], self.coord_standard)
        b2 = LambertXY([dict_minmax['lat_min'], dict_minmax['lon_max']], self.coord_standard)
        b3 = LambertXY([dict_minmax['lat_max'], dict_minmax['lon_max']], self.coord_standard)
        b4 = LambertXY([dict_minmax['lat_max'], dict_minmax['lon_min']], self.coord_standard)
        self.Border = Polygon([b1, b2, b3, b4])

        self.coord_standard_XY = {'x_min': b1[0],
                                  'x_max': b3[0],
                                  'y_min': b1[1],
                                  'y_max': b3[1]}

        node_names = list(set(list(self.df.Node1Name) + list(self.df.Node2Name)))

        connected_nodes = {}
        for node_name in node_names:
            linked_list1 = list(set(self.df[(self.df.Node1Name == node_name)].Node2Name))
            linked_list2 = list(set(self.df[(self.df.Node2Name == node_name)].Node1Name))

            connected_nodes[node_name] = linked_list1 + linked_list2

        connected_links = {}
        for node_name in node_names:
            linked_list1 = list(set(self.df[(self.df.Node1Name == node_name)].LinkId))
            linked_list2 = list(set(self.df[(self.df.Node2Name == node_name)].LinkId))

            connected_links[node_name] = linked_list1 + linked_list2

        temp_nodes = {}
        for i in range(len(df_node)):
            temp_nodes[df_node['Name'][i]] = Node(df_node['Name'][i],
                                                  df_node['Latitude'][i],
                                                  df_node['Longitude'][i],
                                                  type=df_node['Type'][i],
                                                  coord_standard=self.coord_standard)  # connected=connected_nodes[df_node['Name'][i]])

        temp_links = {}
        for i in range(len(LinkId)):
            temp_links[LinkId[i]] = Link(LinkId[i],
                                         temp_nodes[Node1Id[i]],
                                         temp_nodes[Node2Id[i]],
                                         LinkType[i])

        for k, v in temp_nodes.items():
            update_node_list = []
            update_link_list = []

            for c_node in connected_nodes[k]:
                update_node_list.append(temp_nodes[c_node])

            for c_node in connected_links[k]:
                update_link_list.append(temp_links[c_node])

            temp_nodes[k].node_connected = update_node_list
            temp_nodes[k].link_connected = update_link_list

        for i in range(len(LinkId)):
            temp_links[LinkId[i]].node1 = temp_nodes[Node1Id[i]]
            temp_links[LinkId[i]].node2 = temp_nodes[Node2Id[i]]
            temp_links[LinkId[i]].update()

        self.nodes = temp_nodes
        self.links = temp_links

        runway_nodes = list(self.nodes_df[self.nodes_df.Type == "Runway"].Name)

        runway_threshold = sorted([k for k in runway_nodes if k.split('-')[1][0].isdigit()])
        airport_code = self.df.Airport[0]

        self.runways = {}
        self.runways_nodes = {}
        self.runways_links = {}

        for runway1 in runway_threshold:

            runway1_num = runway1.split('-')[1][:-1]  # R, L 삭제
            runway1_side = runway1[-1]

            if int(runway1_num) + 18 <= 36:
                runway2_num = int(runway1_num) + 18
            else:
                runway2_num = int(runway1_num) - 18

            if runway1_side == 'R':
                runway2_side = 'L'

            else:
                runway2_side = 'R'

            runway2 = '-'.join([airport_code, str(runway2_num) + runway2_side])

            temp_link = Link(runway1, self.nodes[runway1], self.nodes[runway2], link_type="Runway")

            self.runways[runway1] = temp_link

            std_node = runway1
            link_li = []
            node_li = [std_node]

            while 1:
                temp_link = [t_link.name for t_link in self.nodes[std_node].link_connected if
                             (t_link.node1.type == "Runway") &
                             (t_link.node2.type == "Runway") &
                             (t_link.name not in link_li)]
                if temp_link:
                    temp_link = temp_link[0]
                else:
                    break

                link_li.append(temp_link)

                std_node = [t_node.name for t_node in self.links[temp_link].nodes if
                            (t_node.name not in node_li) & (t_node.type == "Runway")][0]
                node_li.append(std_node)

            self.runways_nodes[runway1] = node_li
            self.runways_links[runway1] = link_li

        tri_nodes = [
            ['RKSI-TX92', 'RKSI-TX118', 'RKSI-TX93'],
            ['RKSI-TX96', 'RKSI-TX139', 'RKSI-TX98'],
            ['RKSI-TX100', 'RKSI-TX101', 'RKSI-TX102'],
            ['RKSI-TX109', 'RKSI-TX110', 'RKSI-TX111'],
            ['RKSI-TX29', 'RKSI-TX28', 'RKSI-TX30'],
            ['RKSI-TX36', 'RKSI-TX37', 'RKSI-TX38'],
            ['RKSI-TX44', 'RKSI-TX45', 'RKSI-TX46']]

        tri_links = [
            ['RKSI-TX92=TX118', 'RKSI-TX93=TX118'],
            ['RKSI-TX96=TX139', 'RKSI-TX98=TX139'],
            ['RKSI-TX100=TX101', 'RKSI-TX101=TX102'],
            ['RKSI-TX109=TX110', 'RKSI-TX110=TX111'],
            ['RKSI-TX28=TX29', 'RKSI-TX28=TX30'],
            ['RKSI-TX36=TX37', 'RKSI-TX37=TX38'],
            ['RKSI-TX44=TX45', 'RKSI-TX45=TX46']]

        self.obtuse_nodes = [temp_node[1] for temp_node in tri_nodes]
        self.tri_nodes = tri_nodes + [[nodes[2], nodes[1], nodes[0]] for nodes in tri_nodes]
        self.tri_links = tri_links + [[links[1], links[0]] for links in tri_links]

    def check_to_contain(self, coord):

        cond1 = self.coord_standard['lat_min'] <= coord[0] <= self.coord_standard['lat_max']
        cond2 = self.coord_standard['lon_min'] <= coord[1] <= self.coord_standard['lon_max']

        if cond1 and cond2:
            return True
        else:
            return False

    def check_to_contain_XY(self, coord):

        cond1 = self.coord_standard_XY['x_min'] <= coord[0] <= self.coord_standard_XY['x_max']
        cond2 = self.coord_standard_XY['y_min'] <= coord[1] <= self.coord_standard_XY['y_max']

        if cond1 and cond2:
            return True
        else:
            return False

    def node2link(self, node1, node2):

        n1_str = node1.replace(f'{self.Id}-', "")
        n2_str = node2.replace(f'{self.Id}-', "")

        cand1 = f'{self.Id}-{n1_str}={n2_str}'
        cand2 = f'{self.Id}-{n2_str}={n1_str}'

        if cand1 in self.links:
            linkId = cand1
        elif cand2 in self.links:
            linkId = cand2
        else:
            linkId = f"Invalid Link"

        return linkId

    def determine_flight(self, aircraft):

        if aircraft.df.Destination.values[0] == self.Id:
            FlightType = "Arrival"
        elif aircraft.df.Origin.values[0] == self.Id:
            FlightType = "Departure"
        else:
            FlightType = "Unknown"

        aircraft.FlightType = FlightType
        return

    def create_polygon(self, width1=40, width2=20, width3=15, width4=10, return_=False):

        width_each_type = {"Runway": width1,
                           "RapidExitTaxiway": width2,
                           "Taxiway": width2,
                           "Ramp": width3,
                           "Gate": width4}
        Polygons = {}
        for k, v in self.links.items():
            Polygons[k] = v.create_polygon_xy(width_each_type[v.type], True)

        self.Polygons = Polygons

        # Circle
        Circles = {}

        for k, v in self.nodes.items():
            candidate = [width_each_type[t_link.type] for t_link in v.link_connected]
            rad = max(candidate) * np.sqrt(2)
            Circles[k] = v.create_circle(rad, return_=True)

        self.Circles = Circles

        if return_:
            return Polygons

    # Runway Polygon 생성
    # width = 50 m
    def create_runway_polygon(self, width=50, return_=False):

        RunwayPolygons = {}

        for k, v in self.runways.items():
            RunwayPolygons[k] = v.create_polygon_xy(width, return_=True)

        # 속성으로 저장
        self.RunwayPolygons = RunwayPolygons

        if return_:
            return RunwayPolygons

    # Polygon 내에 항공기 위치하는 지 확인
    def find_runway(self, aircraft, find_all=False):
        if self.RunwayPolygons is None:
            self.create_runway_polygon()

        # 데이터가 없을 경우 에러 반환
        if aircraft.df.empty:
            error = 'Data Error(no data)'

            return error

        n = len(aircraft.df)

        # Departure --> 시간 순으로 정렬하여, 폴리곤 내 항공기 확인
        # 최초 10개의 데이터만 이용
        if aircraft.Origin == self.Id:
            loop_order = range(n)

        elif aircraft.Destination == self.Id:
            loop_order = reversed(range(n))

        else:
            loop_order = range(n)
            find_all = True

        # 탐지한 활주로와 해당 row의 index를 저장
        detected_dict = {'runway': [],
                         'idx': []}
        for i in loop_order:

            point = Point(aircraft.x[i], aircraft.y[i])

            for k, v in self.RunwayPolygons.items():
                if v.contains(point):
                    runway_vector = self.runways[k].get_dir_vector()

                    head = aircraft.df['Course'][i]
                    aircraft_vector = [np.sin(np.deg2rad(head)), np.cos(np.deg2rad(head))]

                    if np.dot(runway_vector, aircraft_vector) >= np.cos(np.deg2rad(45)):
                        detected_dict['runway'].append(k)
                        detected_dict['idx'].append(i)

            if not find_all:
                if len(detected_dict['runway']) >= 10:
                    break

        return detected_dict

    def determine_runway(self, aircraft):
        detected_dict = self.find_runway(aircraft, find_all=False)

        # detected_dict에서 가장 많이 검출된 runway를 사용한다고 판별
        if detected_dict['runway']:
            counter = Counter(detected_dict['runway'])
            max_count = max(counter.values())

            runway = [rw for rw in counter.keys() if counter.get(rw) == max_count]

            if len(runway) == 1:
                runway = runway[0]

            else:
                runway = 'Unknown'
        else:
            runway = 'Unknown'

        # 해당 내용을 아래의 column에 저장
        aircraft.df.loc[:, 'RunwayDesignation'] = runway
        aircraft.RunwayDesignation = runway

        return

    # runway내에 항공기가 위치한 케이스만 데이터프레임으로 출력
    def on_runway(self, aircraft):

        self.determine_runway(aircraft)
        detected_dict = self.find_runway(aircraft, find_all=True)
        on_runway_df = aircraft.df.iloc[sorted(detected_dict['idx'])].copy()
        on_runway_df['RunwayCurrent'] = detected_dict['runway']
        return on_runway_df

    # detect
    def detect(self, aircraft, refer=False, origin_columns=False, return_=True):

        aircraft.df["RunwayDesignation"] = aircraft.df["RunwayDesignation"].astype("string")
        aircraft.df["AircraftStand"] = aircraft.df["AircraftStand"].astype("string")

        # LinkId = self.df['LinkId']
        if isinstance(aircraft.df, str):
            error = 'Data Error(no data)'
            return error

        if aircraft.df.empty:
            error = 'Data Error(no data)'
            return error

        self.determine_runway(aircraft)

        aircraft_course = aircraft.df.Course

        # if aircraft.df.Destination.values[0] == self.Id:
        #     FlightType = "Arrival"
        #     aircraft.FlightType = FlightType
        # elif aircraft.df.Origin.values[0] == self.Id:
        #     FlightType = "Departure"
        #     aircraft.FlightType = FlightType
        # else:
        #     FlightType = "Unknown"
        #     aircraft.FlightType = FlightType

        self.determine_flight(aircraft)
        FlightType = aircraft.FlightType

        n = len(aircraft.x)

        node1 = []
        node2 = []
        progress = []
        LinkType = []
        detected_link = []

        flag_node = []
        flag_link = []
        ref_type = None
        nd_count = 0

        initial_idx = 0

        runway_flag = True
        for i in range(initial_idx, n):

            point = Point(aircraft.x[i], aircraft.y[i])
            recheck = False

            if self.Border.contains(point):
                pass

            else:
                node1.append('-')
                node2.append('-')
                progress.append('-')
                LinkType.append('-')
                detected_link.append('no detected')

                flag_node = []
                flag_link = []
                ref_type = None
                nd_count += 1
                continue

            ##

            junction_candidate = []

            if not flag_node:
                for k, v in self.Circles.items():
                    if v.contains(point):
                        junction_candidate.append(self.nodes[k])

            else:
                # if isinstance(flag_node, list):
                new_circles_dict = {t_node: self.Circles[t_node] for t_node in flag_node}

                # else:
                #     new_circles_dict = {flag_node: self.Circles[flag_node]}
                #     new_circles_dict.update({t_node: self.Circles[t_node] for t_node in self.nodes[flag_node].node_connected_list()})

                for k, v in new_circles_dict.items():
                    if v.contains(point):
                        junction_candidate.append(self.nodes[k])

            if junction_candidate:
                if len(junction_candidate) == 1:
                    junction_node = junction_candidate[0]

                else:
                    distance_list = []
                    for junc in junction_candidate:
                        distance_list.append(point.distance(junc.Point()))

                    junction_node = junction_candidate[distance_list.index(min(distance_list))]

                if junction_node.name in self.obtuse_nodes:
                    obtuse_flag = 0
                    pre_tri_nodes = [nodes[0] for nodes in self.tri_nodes if junction_node.name in nodes]
                    for pre_tri_node in pre_tri_nodes:
                        if [temp_link for temp_link in progress if pre_tri_node in temp_link]:     # 둔각 링크가 삼각형 꼴 이후에 나타났을 경우
                            obtuse_flag = True

                    if obtuse_flag:
                        pass

                    else:
                        node1.append(junction_node.name)
                        node2.append(junction_node.name)
                        progress.append(junction_node.name)
                        detected_link.append('/'.join(self.nodes[junction_node.name].link_connected_list()))

                        flag_node = [junction_node.name] + junction_node.node_connected_list()
                        flag_link = junction_node.link_connected_list()
                        nd_count = 0

                        LinkType.append('##' + junction_node.type)
                        continue

                else:

                    node1.append(junction_node.name)
                    node2.append(junction_node.name)
                    progress.append(junction_node.name)
                    detected_link.append('/'.join(self.nodes[junction_node.name].link_connected_list()))

                    flag_node = [junction_node.name] + junction_node.node_connected_list()
                    flag_link = junction_node.link_connected_list()
                    nd_count = 0

                    # LinkType.append("Junction")
                    # LinkType.append(f'Junction/{junction_node.type}')
                    LinkType.append('##' + junction_node.type)
                    continue

            else:
                # flag_node = []
                if not flag_node:
                    pass
                else:
                    flag_node = list(
                        set(itertools.chain(*[self.nodes[temp].node_connected_list() for temp in flag_node])))

            # Polygon
            is_inside = []

            if not flag_link:
                for k, v in self.Polygons.items():
                    if v.contains(point):
                        is_inside.append(self.links[k])

            else:
                # if isinstance(flag_link, list):
                new_polygons_dict = {t_link: self.Polygons[t_link] for t_link in flag_link}

                # else:
                #     new_polygons_dict = {flag_link: self.Polygons[flag_link]}
                #     new_polygons_dict.update({t_link: self.Polygons[t_link] for t_link in self.links[flag_link].link_connected_list()})

                for k, v in new_polygons_dict.items():
                    if v.contains(point):
                        is_inside.append(self.links[k])

            # case 1 : empty
            if not is_inside:
                node1.append('-')
                node2.append('-')
                progress.append('-')
                LinkType.append('-')
                detected_link.append('no detected')
                if not flag_link:
                    pass
                else:
                    flag_link = list(
                        set(itertools.chain(*[self.links[temp].link_connected_list() for temp in flag_link])))
                nd_count += 1


            else:
                # case 2 : only one link, no junction
                if len(is_inside) == 1:
                    temp_head_vec = [np.sin(aircraft_course[i]), np.cos(aircraft_course[i])]

                    temp_link = is_inside[0]
                    temp_link_vec = [temp_link.node2.x - temp_link.node1.x,
                                     temp_link.node2.y - temp_link.node1.y]

                    temp_position_vec = [aircraft.x[i] - temp_link.node1.x,
                                         aircraft.y[i] - temp_link.node1.y]

                    temp_position = np.round(np.dot(temp_position_vec, temp_link_vec) /
                                             (np.linalg.norm(temp_link_vec)) ** 2 * 100)

                    temp_head_dir = np.dot(temp_head_vec, temp_link_vec)

                    if temp_head_dir >= 0:
                        node1.append(temp_link.node1.name)
                        node2.append(temp_link.node2.name)
                        progress.append(f'{temp_position} %')

                    if temp_head_dir < 0:
                        node1.append(temp_link.node2.name)
                        node2.append(temp_link.node1.name)
                        progress.append(f'{100 - temp_position} %')
                        # node1[i] = temp_link.node2.name
                        # node2[i] = temp_link.node1.name
                        # progress[i] = f'{100 - temp_position} %'

                    LinkType.append(temp_link.type)
                    LinkType[i] = temp_link.type
                    ref_type = temp_link.type

                    flag_node = temp_link.node_list() + temp_link.node_connected_list()
                    flag_link = [temp_link.name] + temp_link.link_connected_list()
                    nd_count = 0

                # Junction
                else:
                    # sublist = [[t_link.node1, t_link.node2] for t_link in is_inside]
                    # junction_candidate = [common_node for node_couple in sublist for common_node in node_couple]

                    junction_candidate = []
                    for temp_link in is_inside:
                        junction_candidate.extend([temp_link.node1, temp_link.node2])
                    junction_candidate = list(set(junction_candidate))

                    junction_candidate_dict = {}

                    for temp_node in junction_candidate:
                        # temp_point = Point(temp_node.x, temp_node.y)
                        junction_candidate_dict[temp_node.name] = point.distance(temp_node.Point())

                    distance_min = min(junction_candidate_dict.values())
                    junction_node = [self.nodes[k] for k, v in junction_candidate_dict.items() if v == distance_min][0]

                    if recheck:
                        pass

                    else:

                        node1.append(junction_node.name)
                        node2.append(junction_node.name)
                        progress.append(junction_node.name)

                        if junction_node.type == "RapidExitTaxiway":
                            LinkType.append('Junction/RapidExitTaxiway')
                        else:
                            # LinkType.append('Junction')
                            # LinkType.append(f'Junction/{junction_node.type}')
                            # LinkType.append("##" + junction_node.type)
                            LinkType.append("#" + junction_node.type)

                    flag_node = [junction_node.name] + junction_node.node_connected_list()
                    flag_link = list(set([temp_link for temp_dl in is_inside for temp_link in temp_dl.link_connected_list()]))
                    ref_type = junction_node.type
                    nd_count = 0

                temp_names = [t_node.name for t_node in is_inside]

                if recheck:
                    for t_dl in detected_link[-1].split('/'):
                        if t_dl not in detected_link[-1]:
                            temp_names = [t_dl] in temp_names

                    # detected_link.append('/'.join(temp_names))
                    detected_link[len(detected_link)-1] = temp_names

                else:
                    detected_link.append('/'.join(temp_names))

            if len(node1) > i+1:
                flag = "ASDFASDFASDFASDFA"

            # if ((FlightType == "Departure") & (nd_count > 5) & (ref_type == "Runway")) | ((FlightType == "Departure") & (nd_count > 5) & (ref_type == "Taxiway")):
            if nd_count > 5:
                flag_node = []
                flag_link = []

                if ((FlightType == "Departure") & (ref_type == "Runway")) | (
                        (FlightType == "Departure") & (nd_count > 10) & (ref_type == "Taxiway")):
                    final_idx = i
                    node1 += ['-'] * (n - final_idx - 1)
                    node2 += ['-'] * (n - final_idx - 1)
                    progress += [""] * (n - final_idx - 1)
                    LinkType += ['-'] * (n - final_idx - 1)
                    detected_link += ['no detected'] * (n - final_idx - 1)
                    break

                elif (FlightType == "Arrival") & (ref_type == "Gate") & (nd_count > 30):
                    final_idx = i
                    node1 += ['-'] * (n - final_idx - 1)
                    node2 += ['-'] * (n - final_idx - 1)
                    progress += [""] * (n - final_idx - 1)
                    LinkType += ['-'] * (n - final_idx - 1)
                    detected_link += ['no detected'] * (n - final_idx - 1)
                    break

        if FlightType == "Arrival":
            if ref_type == "Gate":
                gate_link = detected_link[[i for i in range(len(LinkType)) if LinkType[i] == "Gate"][-1]]
                AircraftStand = [temp.name for temp in self.links[gate_link].nodes if temp.type == "Gate"][0]
                aircraft.df.loc[:, "AircraftStand"] = AircraftStand

            else:
                AircraftStand = "Unknown"
                aircraft.df.loc[:, "AircraftStand"] = AircraftStand

            aircraft.AircraftStand = AircraftStand

        elif FlightType == "Departure":
            # if ("Gate" in LinkType) & (LinkType[0] == "Gate"):
            if ("Gate" in LinkType):
                gate_candidate = Counter([detected_link[temp_idx] for temp_idx in range(len(LinkType)) if
                                          LinkType[temp_idx] == 'Gate']).most_common(2)
                if len(gate_candidate) == 1:
                    gate_link = gate_candidate[0][0]
                    AircraftStand = [temp.name for temp in self.links[gate_link].nodes if temp.type == "Gate"][0]
                    aircraft.df.loc[:, "AircraftStand"] = \
                    [temp.name for temp in self.links[gate_link].nodes if temp.type == "Gate"][0]

                elif gate_candidate[0][1] > gate_candidate[1][1]:
                    gate_link = gate_candidate[0][0]
                    AircraftStand = [temp.name for temp in self.links[gate_link].nodes if temp.type == "Gate"][0]
                    aircraft.df.loc[:, "AircraftStand"] = \
                    [temp.name for temp in self.links[gate_link].nodes if temp.type == "Gate"][0]

                else:
                    AircraftStand = "Unknown"
                    aircraft.df.loc[:, "AircraftStand"] = AircraftStand

                # gate_link = detected_link[LinkType.index("Gate")]
                # AircraftStand = [temp.name for temp in self.links[gate_link].nodes if temp.type == "Gate"][0]
                # aircraft.df.loc[:, "AircraftStand"] = [temp.name for temp in self.links[gate_link].nodes if temp.type == "Gate"][0]
            else:
                AircraftStand = "Unknown"
                aircraft.df.loc[:, "AircraftStand"] = AircraftStand

            aircraft.AircraftStand = AircraftStand

        dict_to_add = {'DetectedLink': detected_link,

                       'Node1': node1,
                       'Node2': node2,
                       'Progress/Node': progress,
                       'LinkType': LinkType}

        new_df = aircraft.df.assign(**dict_to_add)
        aircraft.df = new_df

        if origin_columns:
            new_df.columns = ['*Id', 'Callsign', 'Type', 'WakeTurbulenceCategory(WTC)', 'Origin', 'Destination',
                              'TimeSnapShot(sec-utc)', 'Latitude(deg)', 'Longitude(deg)', 'Altitude(ft)',
                              'Groundspeed(kt)', 'VerticalRate(fpm)', 'Course(deg)',
                              'RunwayDesignation', 'AircraftStand', 'DetectedLink', 'Node1', 'Node2', 'Progress/Node',
                              'LinkType']

        if return_:
            return new_df
        else:
            return

    # print
    def to_html(self, name, *aircrafts, palette_option='Set1', ant_path=True,
                opacity=0.7, map_opacity=0.5, aircraft_popup=True, map_style='CartoDB dark_matter', target='ALL', plot_circle=False,
                route=None, route_weight=5, route_color='yellow'):

        if route is None:
            route = []
        lat_center = np.array(list(self.df['Node1Latitude']) + list(self.df['Node2Latitude'])).mean()
        lon_center = np.array(list(self.df['Node1Longitude']) + list(self.df['Node2Longitude'])).mean()

        center_coord = [lat_center, lon_center]
        # tiles = "http://mt0.google.com/vt/lyrs=m&hl=ko&x={x}&y={y}&z={z}"
        tiles = "http://mt0.google.com/vt/lyrs=y&hl=ko&x={x}&y={y}&z={z}"

        attr = "Google"
        my_map = folium.Map(location=center_coord, zoom_start=13, tiles=tiles, attr=attr)

        color_each_type = {"Runway": "red",
                           "RapidExitTaxiway": "magenta",
                           "Taxiway": "blue",
                           "Ramp": "cyan",
                           "Gate": "green"}

        if target == "ALL":
            target_polygons = self.Polygons
            target_circles = self.nodes

        elif target == "RUNWAY":
            target_polygons = self.RunwayPolygons
            target_circles = {}

        else:
            target_polygons = {k: v for k, v in self.Polygons.items() if self.links[k].type == target}
            target_circles = {k: v for k, v in self.nodes.items() if self.nodes[k].type == target}

            if not target_polygons:
                return "Invalid Link Type"

        for i, [k, v] in enumerate(target_polygons.items()):

            p1 = LambertLatLon(v.exterior.coords[0], self.coord_standard)
            p2 = LambertLatLon(v.exterior.coords[1], self.coord_standard)
            p3 = LambertLatLon(v.exterior.coords[2], self.coord_standard)
            p4 = LambertLatLon(v.exterior.coords[3], self.coord_standard)

            polygon = [p1, p2, p3, p4]

            if target == "RUNWAY":
                temp_type = "Runway"
            else:
                temp_type = self.links[k].type
            color = color_each_type[temp_type]

            popup = folium.Popup(f"<h5>{k}</h5>", min_width=10, max_width=1000)

            folium.Polygon(locations=polygon,
                           width=1,
                           color=color,
                           fill_color=color,
                           popup=popup,
                           fill_opacity=0.3 * map_opacity,
                           opacity=map_opacity).add_to(my_map)

        for i, [k, v] in enumerate(target_circles.items()):
            loc = [v.lat, v.lon]

            temp_type = v.type
            color = color_each_type[temp_type]

            rad = v.rad
            popup = folium.Popup(f"<h5>{k}</h5>" +
                                 f"<h6>Type: {v.type}</h6>" +
                                 f"<h6>Lat: {np.round(v.lat, 4)}</h6>" +
                                 f"<h6>Lon: {np.round(v.lon, 4)}</h6>", min_width=10, max_width=1000)
            folium.Circle(location=loc, opacity=map_opacity, radius=rad, popup=popup, color=color, fill=True,
                          fill_opacity=0.5 * map_opacity).add_to(
                my_map)

        if aircrafts:

            if palette_option.startswith("#"):
                colors = [palette_option] * len(aircrafts)

            elif palette_option == "default":
                palette = ['red', 'blue', 'green', 'cyan', 'magenta']
                num_palette = len(palette)
                cycle = len(aircrafts) // num_palette
                left = len(aircrafts) % num_palette
                colors = palette * cycle + palette[:left]

            else:

                palette = color_palette(palette_option, n_colors=len(aircrafts))
                colors = palette
                new_colors = []

                for color in colors:
                    r_scaled = int(color[0] * 255)
                    g_scaled = int(color[1] * 255)
                    b_scaled = int(color[2] * 255)

                    new_colors.append(f"#{r_scaled:02x}{g_scaled:02x}{b_scaled:02x}")

                colors = new_colors

            for temp_aircraft, color in zip(aircrafts, colors):

                loc = np.column_stack([temp_aircraft.lat, temp_aircraft.lon])

                if aircraft_popup:
                    popup = folium.Popup(f"<h5>{temp_aircraft.Callsign}</h5>" +
                                         f"<h6>{temp_aircraft.path}</h6>", min_width=10, max_width=1000)
                else:
                    popup = None

                if ant_path:
                    AntPath(locations=loc, delay=2000, weight=5, color=color, dash_array=[10, 20], opacity=opacity,
                            popup=popup).add_to(my_map)
                else:
                    folium.PolyLine(locations=loc, weight=8, color=color, opacity=opacity, popup=popup).add_to(my_map)
                    if plot_circle:
                        for circle_idx, point in enumerate(loc):
                            circle_popup = folium.Popup(f"<h5>index: {circle_idx}</h5>" +
                                                        f"<h6>coord: {loc[0]}</h6>", min_width=10, max_width=1000)
                            folium.CircleMarker(
                                location=point,
                                radius=5,
                                color=color,
                                fill=True,
                                fill_color=color,
                                fill_opacity=opacity*1.1,
                                popup=circle_popup
                            ).add_to(my_map)

        if route:
            route_loc = np.empty((0, 2))
            for t_node in route:
                route_loc = np.vstack([route_loc, [self.nodes[t_node].lat, self.nodes[t_node].lon]])

            AntPath(locations=route_loc, delay=2000, weight=route_weight, color=route_color, dash_array=[10, 20], opacity=opacity).add_to(my_map)

        if map_style == 'default':
            pass
        else:
            folium.TileLayer(tiles=map_style).add_to(my_map)

        my_map.save(name)

    def convert_timestamp(row, index):
        timestamp = row['Timestamp']
        date = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
        time = datetime.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')

        if timestamp in index:
            raw_data = 'raw'
            lat_raw = row['Lat']
            lon_raw = row['Lat']
            alt_raw = row['Alt']
            gs_raw = row['GS']
        else:
            raw_data = 'interp'
            lat_raw = ""
            lon_raw = ""
            alt_raw = ""
            gs_raw = ""

        return date, time, raw_data, lat_raw, lon_raw, alt_raw, gs_raw

    def get_taxi_route(self, aircraft:Aircraft):

        try:

            df = aircraft.df
            airport.determine_flight(aircraft)
            FT = ac.FlightType

            gt = aircraft.AircraftStand
            rw = aircraft.RunwayDesignation
            gate_ramp = airport.nodes[gt].node_connected_list()[0]

        except:
            return "Invalid Data"

        detected_nodes = list(df[df['LinkType'].str.contains("##")]["Progress/Node"])
        order = [detected_nodes[0]] + [detected_nodes[i] for i in range(1, len(detected_nodes)) if detected_nodes[i] != detected_nodes[i-1]]

        unlinked_idx = []
        additional_nodes = []

        for i in range(len(order) - 1):
            if order[i + 1] not in self.nodes[order[i]].node_connected_list():
                common_node = list(set(self.nodes[order[i]].node_connected_list()) & set(self.nodes[order[i + 1]].node_connected_list()))
                if common_node:
                    unlinked_idx.append(i + 1)
                    additional_nodes.append(common_node[0])

                else:
                    return

        if unlinked_idx:
            for i, (idx, node) in enumerate(zip(unlinked_idx, additional_nodes)):
                order.insert(idx + i, node)

        gate_ramp_flag = True

        if FT == "Departure": # check push-back
            gate_ramp_count = order.count(gate_ramp)

            if gate_ramp_count == 0:

                if gate_ramp in self.nodes[order[0]].node_connected_list():
                    order.insert(0,gate_ramp)
                    gate_ramp_flag = False

                else:
                    return


            elif gate_ramp_count == 1:      # for starting at Gate-Ramp Node
                start_idx = order.index(gate_ramp)

            else:
                start_idx = len(order) - 1 - order[::-1].index(gate_ramp) # for starting at Gate-Ramp Node (ignore push-back)

            order = order[start_idx:]

        section_time_list = []
        section_index_list = []

        section_time_dict = {}
        section_index_dict = {}

        points_array = points(np.column_stack((aircraft.x, aircraft.y)))

        for i in range(len(order)-1):

            if gate_ramp_flag:

                # continuous?
                # not continuous --> ignore push-back

                if (order[i] == gate_ramp) and (FT == "Departure"):
                    index_list1 = df[df['Progress/Node'] == order[i]].index
                    not_conti = [i for i in range(1, len(index_list1)) if index_list1[i] != index_list1[i-1]+1]  # check continuity

                    if not_conti:
                        index_list1 = index_list1[not_conti[-1]:]

                    else:
                        pass

                else:
                    index_list1 = df[df['Progress/Node'] == order[i]].index

            else:
                index_list1 = df[df['Progress/Node'] == order[i]].index
            index_list2 = df[df['Progress/Node'] == order[i+1]].index



            if not index_list1:

                alt_list1 = df[df['Progress/Node'] == order[i-1]]

                if alt_list1:
                    initial_index = alt_list1[-1]
                else:
                    initial_index = df.index[0]

                if index_list2:
                    final_index = index_list2[0]
                else:
                    final_index = df.index[-1]

                index_list1 = list(range(initial_index, final_index+1))

            else:
                pass

            temp_points_array1 = points_array[index_list1]

            temp_node1 = order[i]
            temp_point1 = Point(airport.nodes[temp_node1].x, airport.nodes[temp_node1].y)
            section_start_index = index_list1[0] + pd.Series(temp_points_array1).apply(lambda p: p.distance(temp_point1)).idxmin()


            if not index_list2:

                if index_list1:
                    initial_index = index_list1[-1]
                else:
                    initial_index = df.index[0]

                if i + 2 <= len(order) - 1:
                    final_index = df[df['Progress/Node'] == order[i+2]].index[0]
                else:
                    final_index = df.index[-1]

                index_list2 = list(range(initial_index, final_index+1))

            else:
                pass

            temp_points_array2 = points_array[index_list2]

            temp_node2 = order[i]
            temp_point2 = Point(airport.nodes[temp_node1].x, airport.nodes[temp_node1].y)
            section_end_index = index_list2[0] + pd.Series(temp_points_array2).apply(lambda p: p.distance(temp_point2)).idxmin()

            section_time = section_end_index - section_start_index

            if section_time < 0:
                return
            section_index_list.append([section_start_index, section_end_index])
            section_time_list.append(section_time)

            temp_link = self.node2link(order[i], order[i+1])

            section_time_dict[temp_link] = section_time

        section_time_dict['total'] = sum(section_time_li)

        aircraft.section_time_dict = section_time_dict
        aircraft.section_index_dict = section_index_list

        return


















    def get_route(self, aircraft, p=True, tri_link=True, unknown=False):

        try:

            if not aircraft.FlightType:
                self.determine_flight(aircraft)

            if aircraft.FlightType == "Arrival":
                runway_df = aircraft.df[aircraft.df["Progress/Node"].isin(self.runways_nodes[aircraft.RunwayDesignation])]

                if runway_df[runway_df.index.diff() > 30].empty:
                    start_idx = runway_df.index[-1]
                else:
                    std_idx = runway_df[runway_df.index.diff() > 30].index[0]
                    start_idx = runway_df.index[list(runway_df.index).index(std_idx) - 1]

                end_idx = aircraft.df[aircraft.df.LinkType == "##Ramp"].index[0]

            elif aircraft.FlightType == "Departure":
                runway_df = aircraft.df[aircraft.df["Progress/Node"].isin(self.runways_nodes[aircraft.RunwayDesignation])]

                start_idx = aircraft.df[aircraft.df.LinkType == "##Ramp"].index[-1]

                if runway_df[runway_df.index.diff() > 30].empty:
                    end_idx = runway_df.index[0]
                else:
                    std_idx = runway_df[runway_df.index.diff() > 30].index[0]
                    end_idx = runway_df.index[list(runway_df.index).index(std_idx)]

            else:
                if unknown:
                    taxi_start_idx = aircraft.df[aircraft.df.LinkType == "##Taxiway"].index[0]
                    temp_df = aircraft.df.iloc[:taxi_start_idx]
                    start_idx = temp_df[temp_df.LinkType == "##Ramp"].index[-1]

                    taxi_end_idx = aircraft.df[aircraft.df.LinkType == "##Taxiway"].index[-1]
                    temp_df = aircraft.df.iloc[taxi_end_idx:]

                    if temp_df[temp_df.LinkType == "##Ramp"].empty:
                        end_idx = taxi_end_idx
                    else:
                        end_idx = temp_df[temp_df.LinkType == "##Ramp"].index[0]
                else:
                    return "Unknown FlightType"

            new_df = aircraft.df.iloc[start_idx:end_idx]
            taxi_time = len(new_df)
            new_df = new_df[new_df.LinkType.isin(["Taxiway", "##Taxiway", "#Taxiway"])]

            N1 = new_df.Node1
            N2 = new_df.Node2
            DL = new_df.DetectedLink
            LT = new_df.LinkType
            connection_flag = False

            route_link = []
            connection = ""         # 다음으로 연결되어야 할 노드

            for n1, n2, dl, lt in zip(N1, N2, DL, LT):

                if n1 == n2:  # junction
                    if connection_flag:
                        if connection == n1:
                            pass
                        else:
                            interp_link = self.node2link(connection, n1)
                            if (interp_link != "Invalid Link") & (interp_link != route_link[-1]):
                                route_link.append(interp_link)

                    elif lt == "#Taxiway":
                        temp_links = dl.split("/")
                        for temp_link in temp_links:
                            route_candidate = []
                            if connection in self.links[temp_link].nodes:
                                route_candidate.append(temp_link)
                            if (len(route_candidate)) == 1:
                                if route_candidate[0] != route_link[-1]:
                                    route_link.append(route_candidate[0])

                    connection = n1
                    connection_flag = True

                else:
                    if not route_link:  # 비어있다면 추가
                        route_link.append(dl)
                    elif dl == route_link[-1]:  # 직전 경로가 같으면 제외
                        pass

                    elif dl in self.links[route_link[-1]].link_connected_list():  # 마지막 링크와 연결 된다면 추가
                        route_link.append(dl)

                    elif connection in self.links[route_link[-1]].node_connected_list():
                        for candi in self.links[route_link[-1]].node_connected_list():
                            link_interp = self.node2link(candi, connection)
                            if (link_interp != "Invalid Link") & (link_interp != route_link[-1]):
                                route_link.append(link_interp)
                                route_link.append(dl)
                    else:
                        route_link.append("error")

                    connection_flag = False

                if (lt == "Ramp") and (aircraft.FlightType == "Arrival"):
                    break

            if p:  # a-b-c 로 되게 하는 것
                removal_list = []
                for i in range(1, len(route_link) - 1):

                    for t_node in self.links[route_link[i]].nodes:

                        if ((t_node in self.links[route_link[i - 1]].nodes) and
                                (t_node in self.links[route_link[i + 1]].nodes)):
                            removal_list.append(route_link[i])
                            break

                for t_link in removal_list:
                    route_link.remove(t_link)

            # for n1, n2, dl, lt in zip(N1, N2, DL, LT):
            #
            #     if n1 == n2:                    # junction
            #         if connection_flag:
            #             if connection == n1:
            #                 pass
            #             else:
            #                 interp_link = self.node2link(connection, n1)
            #                 if interp_link != "Invalid Link":
            #                     route_link.append(interp_link)
            #
            #         connection = n1
            #         connection_flag = True
            #
            #     else:
            #         if not route_link:      # 비어있다면 추가
            #             route_link.append(dl)
            #         elif dl == route_link[-1]:  # 직전 경로가 같으면 제외
            #             pass
            #
            #         elif dl in self.links[route_link[-1]].link_connected_list():    # 마지막 것과 연결 된다면 추가
            #             route_link.append(dl)
            #
            #         elif connection in self.links[route_link[-1]].node_connected_list():
            #             print("elif_pass")
            #             for candi in self.links[route_link[-1]].node_connected_list():
            #                 link_interp = self.node2link(candi, connection)
            #                 if (link_interp != "Invalid Link") & (link_interp != route_link[-1]):
            #                     route_link.append(link_interp)
            #                     route_link.append(dl)
            #         else:
            #             route_link.append("error")
            #             return "error"
            #
            #         connection_flag = False
            #
            #     if (lt == "Ramp") and (aircraft.FlightType == "Arrival"):
            #         break
            #
            # if p:               # a-b-c 로 되게 하는 것
            #     removal_list = []
            #     for i in range(1, len(route_link) - 1):
            #
            #         for t_node in self.links[route_link[i]].nodes:
            #
            #             if ((t_node in self.links[route_link[i - 1]].nodes) and
            #                     (t_node in self.links[route_link[i + 1]].nodes)):
            #                 removal_list.append(route_link[i])
            #                 break
            #
            #     for t_link in removal_list:
            #         route_link.remove(t_link)

            if tri_link:
                remove_list = []

                for route_idx in range(1, len(route_link)):

                    links = [route_link[route_idx-1], route_link[route_idx]]

                    if links in self.tri_links:

                        remove_list.append(links)

                for temp_links in remove_list:

                    nodes1 = self.links[temp_links[0]].nodes
                    nodes2 = self.links[temp_links[1]].nodes
                    uncommon_nodes = list(set(nodes1) ^ set(nodes2))

                    rep_idx = route_link.index(temp_links[0])

                    for temp_link in temp_links:
                        route_link.remove(temp_link)

                    route_link.insert(rep_idx, self.node2link(uncommon_nodes[0].name, uncommon_nodes[1].name))

            # get route_node
            pass_flag = False
            route_node = []

            for i in range(1, len(route_link)):

                n1, n2 = self.links[route_link[i]].nodes

                if (i == 1) | pass_flag:

                    sp1, sp2 = self.links[route_link[i - 1]].nodes

                    if sp1 in [n1, n2]:
                        sp = sp2
                        np = sp1
                    else:
                        sp = sp1
                        np = sp2

                    route_node.append(sp.name)
                    route_node.append(np.name)

                if n1.name == route_node[-1]:
                    route_node.append(n2.name)
                elif n2.name == route_node[-1]:
                    route_node.append(n1.name)
                else:
                    pass_flag = True

        except Exception as e:
            return str(e)

        aircraft.route = route_link
        aircraft.route_link = route_link
        aircraft.route_node = route_node
        aircraft.route_key = '='.join(route_node).replace(f'{self.Id}-', "")
        aircraft.taxi_time = taxi_time
        return

    def convert_route_node(self, route_link):

        pass_flag = False
        route_node = []

        for i in range(1, len(route_link)):

            n1, n2 = self.links[route_link[i]].nodes

            if (i == 1) | pass_flag:

                sp1, sp2 = self.links[route_link[i - 1]].nodes

                if sp1 in [n1, n2]:
                    sp = sp2
                    np = sp1
                else:
                    sp = sp1
                    np = sp2

                route_node.append(sp.name)
                route_node.append(np.name)

            if n1.name == route_node[-1]:
                route_node.append(n2.name)
            elif n2.name == route_node[-1]:
                route_node.append(n1.name)
            else:
                pass_flag = True

        return route_node

    def convert_route_key(self, route_key):

        node_split = route_key.split("=")
        node_li = [[node_split[i - 1], node_split[i]] for i in range(1, len(node_split))]
        route_link = [f'{self.node2link(temp_link[0], temp_link[1])}' for temp_link in node_li]

        return route_link

    def route3(self, file_path, route_link, taxi_time_dict, taxi_spd_dict):

        polygon_data = []

        if type(list(taxi_spd_dict.values())[0][0]) == list:
            spd_flag = True
        else:
            spd_flag = False

        for temp_link in route_link:

            # polygon = [LambertLatLon(coord, self.Id, inverse=True) for coord in self.Polygons[temp_link].exterior.coords]
            polygon = [LambertLatLon(coord, self.Id, inverse=True) for coord in self.links[temp_link].create_polygon_xy2(8, return_=True).exterior.coords]
            section_time = np.mean(taxi_time_dict[temp_link])

            if spd_flag:
                section_spd = np.mean([spd for spd_li in taxi_spd_dict[temp_link] for spd in spd_li])
            else:
                section_spd = np.mean(taxi_spd_dict[temp_link])

            temp_info = {
                'coordinates': [polygon],
                'elevation': np.round(section_spd, 2),
                'sectionTime': section_time,
                'color': util.get_color(section_time, 0, 60, 0.85, color_map='Reds')
            }
            polygon_data.append(temp_info)

        # PolygonLayer 설정
        polygon_layer = pdk.Layer(
            "PolygonLayer",
            polygon_data,
            get_polygon="coordinates",  # 다각형 좌표
            get_elevation="elevation",  # 고도 (기둥의 높이)
            elevation_scale=5,  # 고도 스케일
            get_fill_color='color',  # 각 다각형의 색상 (RGBA)
            get_line_color=[0, 0, 0, 255],
            get_line_width=20,
            pickable=True,
            extruded=True,  # 3D 기둥을 만들기 위해 extruded 옵션을 True로 설정
            wireframe=False,  # 기둥의 외곽선을 보여줄지 여부
            stroked=True
        )

        # 지도의 초기 상태 설정

        view_state = pdk.ViewState(
            latitude=self.lat_center,
            longitude=self.lon_center,
            zoom=14,  # 줌 레벨
            pitch=45,  # 지도 기울기
        )

        deck = pdk.Deck(
            layers=[polygon_layer],
            initial_view_state=view_state,
            map_style='road',
            tooltip={"text": "Time: {sectionTime} sec\nSpeed:\t{elevation} knots"},  # 툴팁 설정
        )

        deck.to_html(file_path, notebook_display=False)

    def route3_aircraft(self, path, aircraft):

        route_link = aircraft.route_link
        taxi_time_dict = aircraft.taxi_time_dict
        taxi_spd_dict = aircraft.taxi_speed_dict

        self.route3(path, route_link, taxi_time_dict, taxi_spd_dict)

    def searchMap(self, path):
        from LambertConformal import LambertLatLon

        polygons = []
        circles = []

        color_each_type = {"Runway": "red",
                           "RapidExitTaxiway": "magenta",
                           "Taxiway": "blue",
                           "Ramp": "cyan",
                           "Gate": "green"}

        for k, v in self.Polygons.items():
            p1 = LambertLatLon(v.exterior.coords[0], self.coord_standard)
            p2 = LambertLatLon(v.exterior.coords[1], self.coord_standard)
            p3 = LambertLatLon(v.exterior.coords[2], self.coord_standard)
            p4 = LambertLatLon(v.exterior.coords[3], self.coord_standard)

            polygon = {
                "name": k,
                "coordinates": [p1, p2, p3, p4],
                "color": color_each_type[self.links[k].type],
                "length": np.round(self.links[k].get_length(), 2)
            }
            polygons.append(polygon)

        for k, v in self.nodes.items():

            circle = {
                "name": k,
                "coordinates": [v.lat, v.lon],
                "radius": v.rad,
                "color": color_each_type[self.nodes[k].type],
            }
            circles.append(circle)

        polygon_json = json.dumps(polygons)
        circle_json = json.dumps(circles)

        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Leaflet Polygon Search</title>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/ljagis/leaflet-measure@2.1.7/dist/leaflet-measure.min.css"/>
            <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
            <script src="https://cdn.jsdelivr.net/gh/ljagis/leaflet-measure@2.1.7/dist/leaflet-measure.min.js"></script>
            

            <style>
                #map {{
                    height: 880px;
                }}
                #search {{
                    margin: 10px;
                    width: 300px; /* 검색 창 컨테이너의 너비 */
                }}
                #searchBox {{
                    width: 70%; /* 입력 상자의 너비를 검색 창 컨테이너의 70%로 설정 */
                    font-size: 12px; /* 입력 상자 텍스트 크기 */
                    margin-right: 5px; /* 버튼과의 간격 */
                }}
               #search button {{
                    font-size: 12px; /* 버튼 텍스트 크기 */
                }}
            </style>
        </head>
        <body>
            <div id="search">
                <input type="text" id="searchBox" placeholder="Enter Link Id or Node Id">
                <button onclick="searchPolygon()">Search</button>
            </div>
            <div id="map"></div>

        <script>
                // 지도 초기화
                var map = L.map('map').setView([{self.lat_center}, {self.lon_center}], 12);

/*
                // OSM 타일 레이어 추가
                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                    maxZoom: 18,
                    attribution: '© OpenStreetMap contributors'
                }}).addTo(map);
*/              
                

            var cartoDBDarkMatter = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}.png', {{
                maxZoom: 19,
                attribution: '&copy; OpenStreetMap contributors & CartoDB'
            }});
            
            var cartoDBPositron = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}.png', {{
                maxZoom: 19,
                attribution: '&copy; OpenStreetMap contributors & CartoDB'
            }});

            var osm = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                maxZoom: 19,
                attribution: '© OpenStreetMap contributors'
            }});
            
            var googleMap = L.tileLayer('https://mt1.google.com/vt/lyrs=m&x={{x}}&y={{y}}&z={{z}}',{{
                maxZoom: 22,
                attribution: 'google'
            }})
            
            var googleSat = L.tileLayer('http://www.google.cn/maps/vt?lyrs=s@189&gl=cn&x={{x}}&y={{y}}&z={{z}}', {{
                maxZoom: 22,
                attribution: 'google'
            }})
            
            cartoDBDarkMatter.addTo(map);


            var baseMaps = {{
                
                "Dark": cartoDBDarkMatter, 
                "Light": cartoDBPositron, 
                "OpenStreetMap": osm,
                "Google Map" : googleMap,
                "Google Satellite":googleSat
            }};

        L.control.layers(baseMaps).addTo(map);

                // Python에서 생성한 Polygon 데이터
                var polygonData = {polygon_json};
                var circleData = {circle_json};


                // Polygon 데이터를 저장할 객체
                var polygonMap = {{}};
                var circleMap = {{}};


                // polygonData를 사용하여 다각형을 지도에 추가
                polygonData.forEach(function(polygon) {{
                    var coordinates = polygon.coordinates

                    var leafletPolygon = L.polygon(coordinates, {{
                        color: polygon.color,
                        opacity: 0.5,
                        fillColor: polygon.color,
                        fillOpacity: 0.3 * 0.5,
                    }}).addTo(map).bindPopup(`<b>${{polygon.name}}</b><br>${{polygon.length}} m`);

                    // 다각형의 이름을 키로 하여 지도 객체를 저장
                    polygonMap[polygon.name] = leafletPolygon;
                }});
                
                circleData.forEach(function(circle) {{
                    var coordinates = circle.coordinates
                    var leafletCircle = L.circle(coordinates, {{
                        color: circle.color,
                        opacity: 0.5,
                        fillColor: circle.color,
                        fillOpacity: 0.3 * 0.5,
                        radius: circle.radius
                    }}).addTo(map).bindPopup(`<b>${{circle.name}}</b>`);

                    // 다각형의 이름을 키로 하여 지도 객체를 저장
                    circleMap[circle.name] = leafletCircle;
                }});

                // 검색 기능 구현
                function searchPolygon() {{
                    var searchText = document.getElementById('searchBox').value;
                    var polygon = polygonMap[searchText];
                    var circle = circleMap[searchText];
                    
                    console.log(polygon);


                    if (polygon) {{
                        map.fitBounds(polygon.getBounds()); // 다각형의 위치로 지도를 조정
                        polygon.openPopup(); // 다각형의 팝업 열기
                        
                    }} else if (circle) {{
                        map.fitBounds(circle.getBounds()); // 다각형의 위치로 지도를 조정
                        circle.openPopup(); // 다각형의 팝업 열기
                    

                    }} else {{
                        alert("Polygon not found");
                    }}
                }}
                
            // Measure Control 추가
            var measureControl = new L.Control.Measure({{
                primaryLengthUnit: 'meters',  // 기본 길이 단위
                secondaryLengthUnit: 'kilometers',  // 보조 길이 단위
                primaryAreaUnit: 'sqmeters',  // 기본 면적 단위
                secondaryAreaUnit: 'hectares',  // 보조 면적 단위
                activeColor: '#ffe863',  // 측정 중 라인의 색상
                completedColor: '#6fff11'  // 측정 완료 시 라인의 색상
            }});
            measureControl.addTo(map);
            
            </script>
        </body>
        </html>
        """

        # HTML 파일로 저장
        with open(path, 'w') as f:
            f.write(html_template)

        print(f"HTML created: {path}")


if __name__ == "__main__":
    print("Hello World")