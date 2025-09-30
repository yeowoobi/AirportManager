import datetime

import numpy as np
import pandas as pd

from collections import OrderedDict

import folium

from LambertConformal import LambertXY_series, LambertLatLon_series, COORD_STANDARD


class Aircraft:

    def __init__(self, path, std_alt=None, coord_standard='RKSI', preprocessed=False, for_conv=False):

        self.path = path

        if preprocessed:
            self.df = pd.read_csv(path, low_memory=False).reset_index(drop=True)
        else:
            self.df = pd.read_csv(path, low_memory=False).iloc[:, 0:15].reset_index(drop=True)

        self.Origin = self.df.Origin[0]
        self.Destination = self.df.Destination[0]


        if for_conv:
            self.Callsign = self.df.CallSign[0]
        else:
            self.Callsign = self.df.Callsign[0]
        self.path = path

        # Unit Reference
        # TimeSnapShot  sec-utc
        # Latitude      deg
        # Longitude     deg
        # Altitude      ft
        # GroundSpeed   knot
        # VerticalRate  ft per minute
        # Course        deg
        if preprocessed:
            columns = ['*Id', 'CallSign', 'Type', 'WakeTurbulenceCategory',
                       'Origin', 'Destination', 'TimeSnapShot',
                       'Latitude', 'Longitude', 'Altitude',
                       'GroundSpeed', 'VerticalRate', 'Course',
                       'RunwayDesignation', 'AircraftStand',
                       'DetectedLink', 'Node1', 'Node2', 'Progress/Node', 'LinkType']

        else:
            columns = ['*Id', 'CallSign', 'Type', 'WakeTurbulenceCategory',
                       'Origin', 'Destination', 'TimeSnapShot',
                       'Latitude', 'Longitude', 'Altitude',
                       'GroundSpeed', 'VerticalRate', 'Course',
                       'RunwayDesignation', 'AircraftStand']

        self.df.columns = columns[:len(self.df.columns)]
        self.original_timestep = list(self.df['TimeSnapShot'])

        if std_alt:
            self.df = self.df[self.df['Altitude'] <= std_alt].reset_index(drop=True)

        try:
            self.Id = self.df["*Id"].values[0].split("_")[1]
        except:
            self.Id = "Unknown"

        self.lat = self.df.Latitude
        self.lon = self.df.Longitude
        self.coord_standard = coord_standard

        self.FlightType = None
        self.route = None
        self.route_link = None
        self.route_node = None
        self.route_key = None
        self.taxi_time = None
        self.taxi_time_dict = None
        self.taxi_speed_dict = None

        if not self.df['Origin'].empty:
            self.Origin = list(self.df['Origin'])[0]

        if not self.df['Destination'].empty:
            self.Destination = list(self.df['Destination'])[0]

        self.x, self.y = LambertXY_series([self.lat, self.lon], key=coord_standard)

        if preprocessed:
            self.FlightType = None
            self.RunwayDesignation = self.df.RunwayDesignation.values[0]
            self.AircraftStand = self.df.AircraftStand.values[0]

        if len(self.df.columns) > 13:

            if self.df.RunwayDesignation.values[0]:
                self.RunwayDesignation = self.df.RunwayDesignation.values[0]
            else:
                self.RunwayDesignation = None

            if self.df.AircraftStand.values[0]:
                self.AircraftStand = self.df.AircraftStand.values[0]
            else:
                self.AircraftStand = None

    def interp(self, order=1):

        if self.df.empty:
            self.df = 'Data Error(no data)'
            return

        col_interp = ['TimeSnapShot',
                      'Latitude', 'Longitude', 'Altitude',
                      'GroundSpeed', 'VerticalRate', 'Course']

        interp_df = self.df[col_interp]
        # timesnap을 index화 진행  (ex 5, 10, 16,...)
        interp_df.set_index('TimeSnapShot', inplace=True)

        interp_df.loc[:, 'Latitude'] = list(self.y)
        interp_df.loc[:, 'Longitude'] = list(self.x)

        # 비워있는 시간 index를 채우기  (ex 5,6,7,8,9,10,...)
        new_index = pd.RangeIndex(start=interp_df.index.min(), stop=interp_df.index.max() + 1, step=1)

        # 시간 index에 따라 2nd Order interpolate 진행
        interp_df.loc[:, 'Course'] = interp_df['Course'].ffill()
        interp_df = interp_df.reindex(new_index)
        interp_df = interp_df.interpolate(method='polynomial', order=order)

        # interp_df['TimeSnapShot(sec-utc)'] = interp_df.index
        interp_df.insert(0, 'TimeSnapShot', interp_df.index)
        interp_df.reset_index(drop=True, inplace=True)

        interp_df.GroundSpeed = np.round(interp_df.GroundSpeed, 2)

        # not interpolated columns
        interp_df.insert(0, 'Destination', self.df['Destination'][0])
        interp_df.insert(0, 'Origin', self.df['Origin'][0])
        interp_df.insert(0, 'WakeTurbulenceCategory', self.df['WakeTurbulenceCategory'][0])
        interp_df.insert(0, 'Type', self.df['Type'][0])
        interp_df.insert(0, 'CallSign', self.df['CallSign'][0])
        interp_df.insert(0, '*Id', self.df['*Id'][0])

        # interp_df 내에 Course, RunwayDesignation, AircraftStand 값 넣기
        interp_df.insert(len(interp_df.columns), 'RunwayDesignation', self.df['RunwayDesignation'][0])
        interp_df.insert(len(interp_df.columns), 'AircraftStand', self.df['AircraftStand'][0])

        # x, y 로 interpolation된 값을 lat, lon으로 변환
        temp_lat, temp_lon = LambertLatLon_series([interp_df.Longitude, interp_df.Latitude], key=self.coord_standard)

        temp_x = interp_df.Longitude.copy()
        temp_y = interp_df.Latitude.copy()

        interp_df.loc[:, 'Longitude'] = temp_lon
        interp_df.loc[:, 'Latitude'] = temp_lat

        # interp_df.Longitude = temp_lon
        # interp_df.Latitude = temp_lat

        self.x = temp_x
        self.y = temp_y

        # # course interpolation 진행
        # new_vec = np.arctan2(x_diff, y_diff)
        # new_deg = np.round(np.degrees(new_vec), 2)
        #
        # mask = new_deg < 0
        # new_deg[mask] += 360
        #
        # for k, v in zip(self.df.TimeSnapShot, self.df.Course):
        #     interp_df.loc[interp_df.TimeSnapShot == k, 'Course'] = v
        #
        # interp_df.loc[interp_df.Course == -1, 'Course'] = new_deg[interp_df.Course == -1]

        self.df = interp_df

        # self.lat, self.lon  interpolate값 넣어주기
        self.lat = temp_lat
        self.lon = temp_lon

        return

    def to_html(self, file_name, coord_standard_key='RKSI'):

        center_lat = (COORD_STANDARD[coord_standard_key]['lat_min'] + COORD_STANDARD['RKSI']['lat_max']) / 2
        center_lon = (COORD_STANDARD[coord_standard_key]['lon_min'] + COORD_STANDARD['RKSI']['lon_max']) / 2

        center_coord = [center_lat, center_lon]
        my_map = folium.Map(location=center_coord, zoom_start=12)
        loc = np.column_stack([self.lat, self.lon])
        folium.PolyLine(locations=loc, weight=4, color='blue').add_to(my_map)

        my_map.save(file_name)

    def get_taxi_time_dict(self):
        # get taxi time

        route_link = self.route_link
        route_node = self.route_node

        # dl = list(self.df[self.df.LinkType == "Taxiway"].DetectedLink)
        # dl = [dl[0]] + [dl[i] for i in range(1, len(dl)) if dl[i] != dl[i - 1]]

        jn = list(self.df[self.df.LinkType.isin(["##Runway", "##Taxiway", "##Ramp"])].Node1)
        jn = [jn[0]] + [jn[i] for i in range(1, len(jn)) if jn[i] != jn[i - 1]]

        time_dict = {r: 0 for r in route_link}
        speed_dict = {r: [] for r in route_link}

        uncommon_nodes = list(set(route_node) - set(jn))
        uncommon_idx = [
            [route_node.index(temp_node) - 1, route_node.index(temp_node),] for temp_node in uncommon_nodes]
        block_set = set(idx for sublist in uncommon_idx for idx in sublist)

        for node_idx in range(len(route_node)):

            # time in junction
            junction_df = self.df[(self.df['Progress/Junction'] == route_node[node_idx]) & (self.df['LinkType'].isin(["##Runway", "##Taxiway", "##Ramp"]))]
            time_junction = len(junction_df)
            speed_junction = list(junction_df.GroundSpeed)
            if time_junction != 0:

                if node_idx == 0:
                    time_dict[route_link[node_idx]] += 1
                elif node_idx == len(route_node) - 1:
                    time_dict[route_link[node_idx - 1]] += 1

                else:

                    time_dict[route_link[node_idx - 1]] += (time_junction / 2)
                    time_dict[route_link[node_idx]] += (time_junction / 2)

                    speed_dict[route_link[node_idx - 1]] += speed_junction
                    speed_dict[route_link[node_idx]] += speed_junction

            # time in link
            if node_idx < len(route_node) - 1:
                if node_idx not in block_set:
                    start = self.df[(self.df['Progress/Junction'] == route_node[node_idx]) & (self.df['LinkType'].isin(["##Runway", "##Taxiway", "##Ramp"]))].index[-1] + 1
                    end = self.df[(self.df['Progress/Junction'] == route_node[node_idx + 1]) & (self.df['LinkType'].isin(["##Runway", "##Taxiway", "##Ramp"]))].index[0] - 1
                    time_link = end - start + 1
                    time_dict[route_link[node_idx]] += time_link
                    speed_dict[route_link[node_idx]] += list(self.df.iloc[start:end].GroundSpeed)

                else:
                    pass

        for sublist in uncommon_idx:
            link1 = route_link[sublist[0]]
            link2 = route_link[sublist[1]]

            link1_idx = self.df[self.df['DetectedLink'] == link1].index
            link2_idx = self.df[self.df['DetectedLink'] == link2].index

            start1 = link1_idx[0]
            end1 = link1_idx[-1]
            time1 = end1 - start1 + 1

            start2 = link2_idx[0]
            end2 = link2_idx[-1]
            time2 = end2 - start2 + 1

            common_time = (start2 - 1) - (end1 + 1) + 1

            time_dict[link1] += time1 + common_time / 2
            speed_dict[link1] += list(self.df.iloc[start1:end1].GroundSpeed)

            time_dict[link2] += time2 + common_time / 2
            speed_dict[link2] += list(self.df.iloc[start2:end2].GroundSpeed)

        self.taxi_time_dict = time_dict
        self.taxi_speed_dict = speed_dict

        return

    def get_ttd(self):
        self.get_taxi_time_dict()
        return

    # def get_ttd_test(self):
    #     # get taxi time
    #
    #     route_link = self.route_link
    #     route_node = self.route_node
    #
    #     # dl = list(self.df[self.df.LinkType == "Taxiway"].DetectedLink)
    #     # dl = [dl[0]] + [dl[i] for i in range(1, len(dl)) if dl[i] != dl[i - 1]]
    #
    #     jn = list(self.df[self.df.LinkType.isin(["##Runway", "##Taxiway", "##Ramp"])].Node1)
    #     jn = [jn[0]] + [jn[i] for i in range(1, len(jn)) if jn[i] != jn[i - 1]]
    #
    #     time_dict = {r: [] for r in route_link}
    #
    #     uncommon_nodes = list(set(route_node) - set(jn))
    #     uncommon_idx = [
    #         [route_node.index(temp_node) - 1, route_node.index(temp_node)] for temp_node in uncommon_nodes]
    #     block_set = set(idx for sublist in uncommon_idx for idx in sublist)
    #
    #     for node_idx in range(len(route_node)):
    #
    #         # time in junction
    #         # time_junction = len(self.df[self.df['Progress/Junction'] == route_node[node_idx]])
    #         time_junction = len(self.df[(self.df['Progress/Junction'] == route_node[node_idx]) & (self.df['LinkType'].isin(["##Runway", "##Taxiway", "##Ramp"]))])
    #         if time_junction != 0:
    #
    #             if node_idx == 0:
    #                 time_dict[route_link[node_idx]].append(time_junction)
    #             elif node_idx == len(route_node) - 1:
    #                 time_dict[route_link[node_idx - 1]].append(time_junction)
    #             else:
    #
    #                 time_dict[route_link[node_idx - 1]].append(time_junction / 2)
    #                 time_dict[route_link[node_idx]].append(time_junction / 2)
    #
    #         else:
    #             time_dict[route_link[node_idx - 1]].append(0)
    #             time_dict[route_link[node_idx]].append(0)
    #
    #         # time in link
    #         if node_idx < len(route_node) - 1:
    #             if node_idx not in block_set:
    #                 start = self.df[(self.df['Progress/Junction'] == route_node[node_idx]) & (self.df['LinkType'].isin(["##Runway", "##Taxiway", "##Ramp"]))].index[-1] + 1
    #                 end = self.df[(self.df['Progress/Junction'] == route_node[node_idx + 1]) & (self.df['LinkType'].isin(["##Runway", "##Taxiway", "##Ramp"]))].index[0] - 1
    #                 time_link = end - start + 1
    #                 time_dict[route_link[node_idx]].append(time_link)
    #             else:
    #                 time_dict[route_link[node_idx]].append(0)
    #                 pass
    #
    #     for sublist in uncommon_idx:
    #         link1 = route_link[sublist[0]]
    #         link2 = route_link[sublist[1]]
    #
    #         link1_idx = self.df[self.df['DetectedLink'] == link1].index
    #         link2_idx = self.df[self.df['DetectedLink'] == link2].index
    #
    #         start1 = link1_idx[0]
    #         end1 = link1_idx[-1]
    #         time1 = end1 - start1 + 1
    #
    #         start2 = link2_idx[0]
    #         end2 = link2_idx[-1]
    #         time2 = end2 - start2 + 1
    #
    #         common_time = (start2 - 1) - (end1 + 1) + 1
    #
    #         time_dict[link1].append(time1 + common_time / 2)
    #         time_dict[link2].append(time2 + common_time / 2)
    #
    #     self.taxi_time_dict_test = time_dict
    #
    #     return

    # def cal_taxi_time(self):
    #     try:
    #         taxi_df = self.df[self.df.LinkType.isin(["##Taxiway", "Taxiway"])]["TimeSnapShot"]
    #
    #         if self.FlightType == "Arrival":
    #             LDT = taxi_df.index[0]
    #             IBT = taxi_df.index[-1]
    #             taxi_time = IBT - LDT
    #
    #         elif self.FlightType == "Departure":
    #             TOT = taxi_df.index[0]
    #             OBT = taxi_df.index[-1]
    #             taxi_time = OBT - TOT
    #
    #         else:
    #             taxi_time = None
    #
    #         return taxi_time
    #
    #     except:
    #         # print("Can't Calculate Taxi Time")
    #         return
    #
    # def cal_taxi_time_by_link(self):
    #
    #     taxi_time_dict = {}
    #
    #     try:
    #
    #         # no_junction_type = ['Runway', 'Taxiway', 'RapidExitTaxiway', 'Ramp', 'Gate']
    #         # no_junction_list = list(OrderedDict.fromkeys(self.df[self.df.LinkType.isin(no_junction_type)].DetectedLink))
    #         no_junction_list = self.route_link
    #         junction_list = list(OrderedDict.fromkeys(self.df[self.df.LinkType.str.contains('Junction')].DetectedLink))
    #
    #         for no_junction in no_junction_list:
    #
    #             section_time = len(self.df[self.df.DetectedLink == no_junction])
    #             if no_junction in taxi_time_dict:
    #                 taxi_time_dict[no_junction] += section_time
    #
    #             else:
    #                 taxi_time_dict[no_junction] = section_time
    #
    #         for junction in junction_list:
    #
    #             section_time = len(self.df[self.df.DetectedLink == junction])
    #             temp_link_list = [temp_junction_link for temp_junction_link in junction.split('/') if
    #                               temp_junction_link in no_junction_list]
    #
    #             for temp_link in temp_link_list:
    #
    #                 if temp_link in taxi_time_dict:
    #                     taxi_time_dict[temp_link] += section_time / len(temp_link_list)
    #
    #                 else:
    #                     taxi_time_dict[temp_link] = section_time / len(temp_link_list)
    #
    #         return taxi_time_dict
    #
    #     except:
    #         print("Can't Calculate Taxi Time")
    #         return

    def rename_columns(self):

        if 'CallSign' in self.df.columns:
            self.df.rename(columns={'CallSign': 'Callsign',
                                    'WakeTurbulenceCategory': 'WakeTurbulenceCategory(WTC)',
                                    'TimeSnapShot': 'TimeSnapShot(sec-utc)',
                                    'Latitude': 'Latitude(deg)',
                                    'Longitude': 'Longitude(deg)',
                                    'Altitude': 'Altitude(ft)',
                                    'GroundSpeed': 'Groundspeed(kt)',
                                    'VerticalRate': 'VerticalRate(fpm)',
                                    'Course': 'Course(deg)'}, inplace=True)
        else:
            self.df.rename(columns={'Callsign': 'CallSign',
                                    'WakeTurbulenceCategory(WTC)': 'WakeTurbulenceCategory',
                                    'TimeSnapShot(sec-utc)': 'TimeSnapShot',
                                    'Latitude(deg)': 'Latitude',
                                    'Longitude(deg)': 'Longitude',
                                    'Altitude(ft)': 'Altitude',
                                    'Groundspeed(kt)': 'GroundSpeed',
                                    'VerticalRate(fpm)': 'VerticalRate',
                                    'Course(deg)': 'Course'}, inplace=True)

    def cut_by_alt(self, alt):
        if alt:
            self.df = self.df[self.df['Altitude'] <= alt]

        ini_idx = self.df.index[0]
        end_idx = self.df.index[-1]

        self.lat = self.lat.iloc[ini_idx:end_idx].reset_index(drop=True)
        self.lon = self.lon.iloc[ini_idx:end_idx].reset_index(drop=True)

        self.x = self.x.iloc[ini_idx:end_idx].reset_index(drop=True)
        self.y = self.y.iloc[ini_idx:end_idx].reset_index(drop=True)

        self.df = self.df.reset_index(drop=True)

    def set_columns(self, col_type='type1'):

        if col_type == 'type1':
            while 'Callsign' not in self.df.columns:
                self.rename_columns()
        elif col_type == 'type2':
            while 'CallSign' not in self.df.columns:
                self.rename_columns()
        else:
            return "Invalid Columns"

    def save(self, filename):

        self.set_columns('type1')
        self.df.to_csv(filename, index=False)
        self.rename_columns()

        return

    def convert_timestamp(self, row):
        timestamp = row['Timestamp']
        date = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
        time = datetime.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
        progress = row['Progress']
        procedure = row['Procedure']
        runway = row['Runway'].replace("RKSI-", "")

        if isinstance(progress, str):
            if '%' in progress:
                progress = int(float(progress.replace("%", "").replace(" ", "")))

            else:
                progress = ""

        if procedure:
            if procedure.startswith("##"):
                procedure = procedure.lstrip("##")

        if timestamp in self.original_timestep:
            raw_data = 'raw'
            lat_raw = row['Lat']
            lon_raw = row['Lon']
            alt_raw = row['Alt']
            vr_raw = row['VR']
            gs_raw = row['GS']
            trk_raw = row['Trk']
        else:
            raw_data = 'interp'
            lat_raw = ""
            lon_raw = ""
            alt_raw = ""
            vr_raw = ""
            gs_raw = ""
            trk_raw = ""

        return date, time, raw_data, lat_raw, lon_raw, alt_raw, vr_raw, gs_raw, trk_raw, progress, procedure, runway

    def getMasterDataset(self):

        self.df.columns = ['*Id', 'Callsign', 'Type', 'WakeTurbulenceCategory(WTC)', 'Origin', 'Destination',
                           'TimeSnapShot(sec-utc)', 'Latitude(deg)', 'Longitude(deg)', 'Altitude(ft)',
                           'Groundspeed(kt)', 'VerticalRate(fpm)', 'Course(deg)',
                           'RunwayDesignation', 'AircraftStand', 'DetectedLink', 'Node1', 'Node2', 'Progress/Node',
                           'LinkType']

        new_columns = [
            '*Id', 'Callsign', 'Type', 'Date', 'Time', 'Timestamp', 'Raw_data',
            'Lat_raw', 'Lon_raw', 'Alt_raw', 'VR_raw', 'GS_raw', 'Trk_raw',
            'Lat', 'Lon', 'Alt', 'VR', 'GS', 'Trk', 'FL', 'AGL', 'Dist', 'Gamma', 'Roll', 'Pitch', 'Yaw', 'TAS', 'CAS',
            'IAS', 'Mass', 'CL', 'CD',
            'Lift', 'Drag', 'Thrust', 'Throttle', 'Flap', 'Gear',
            'Wx', 'Wy', 'Wz', 'WS',
            'Trk_wind', 'P', 'T', 'P0', 'T0',
            'Airspace', 'Origin', 'Destination', 'Entry', 'Exit', 'Progress', 'Procedure',
            'Vectoring', 'Grid_main', 'Grid_sub', 'Cluster_main', 'Cluster_sub', 'Runway']

        mapping = {
            '*Id': '*Id',
            'Callsign': 'Callsign',
            'Type': 'Type',
            'TimeSnapShot(sec-utc)': 'Timestamp',
            'Latitude(deg)': 'Lat',
            'Longitude(deg)': 'Lon',
            'Altitude(ft)': 'Alt',
            'VerticalRate(fpm)': 'VR',
            'Course(deg)': 'Trk',
            'Groundspeed(kt)': 'GS',
            'Origin': 'Origin',
            'Destination': 'Destination',
            'Node1': 'Entry',
            'Node2': 'Exit',
            'Progress/Node': 'Progress',
            'LinkType': 'Procedure',
            'RunwayDesignation': 'Runway'}

        new_df = pd.DataFrame(columns=new_columns)

        cols_to_copy = [col for col in mapping.keys() if col in self.df.columns]
        new_df[[mapping[col] for col in cols_to_copy]] = self.df[cols_to_copy]
        # new_df[[col for col in new_df if col not in mapping.values()]] = 0

        new_df[['Date', 'Time', 'Raw_data',
                'Lat_raw', 'Lon_raw', 'Alt_raw', 'VR_raw', 'GS_raw', 'Trk_raw',
                'Progress', 'Procedure', 'Runway']] = new_df.apply(self.convert_timestamp, axis=1).apply(pd.Series)

        new_df.loc[self.df['Node1'] == "-", ["Entry", "Exit", "Procedure"]] = ""
        new_df['Cluster_sub'] = 1
        if new_df['Runway'].values[0] == "Unknown":
            new_df['Runway'] = ""

        return new_df


if __name__ == '__main__':
    print("Hello World")