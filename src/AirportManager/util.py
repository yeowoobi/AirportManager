import os
import random
import pickle
import json

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from datetime import datetime, timezone, timedelta


from .aircraft import Aircraft


from tqdm import tqdm

data_path = "/Volumes/T7 Shield/Data/ASDE"
data_path_detected = "/Volumes/T7 Shield/Data/ASDE_detected"
KST = timezone(timedelta(hours=9))


def load_file_list(path, data_type=None, parent_path=False):
    data_list = os.listdir(path)
    if data_type == 'dir':
        data_list = sorted([temp_data for temp_data in data_list
                            if os.path.isdir(os.path.join(path, temp_data))])
    else:
        data_list = [temp_data for temp_data in data_list if temp_data.endswith(f'.{data_type}')]

    if parent_path:
        data_list = [os.path.join(path, temp_data) for temp_data in data_list]

    return data_list


def save_pickle(file_path, data, default_path=False):
    if default_path:
        file_path = os.path.join('/Volumes/T7 Shield/Data/pickles', file_path)

    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def load_pickle(file_path, default_path=False):
    if default_path:
        file_path = os.path.join('/Volumes/T7 Shield/Data/pickles', file_path)

    with open(file_path, "rb") as file:
        return pickle.load(file)


def slice_quantile(data, quantiles):
    data = sorted(data)
    quantiles_cnt = int(len(data) * quantiles)
    cnt = len(data)
    return data[quantiles_cnt:cnt - quantiles_cnt]


def load_samples(n=100, file_path="/Volumes/T7 Shield/Data/ASDE", data_type='path'):

    folder_list = load_file_list(file_path, data_type='dir', parent_path=True)

    data_list = []

    for folder in folder_list:
        temp_folder = load_file_list(folder, data_type='csv', parent_path=True)

        for temp_file in temp_folder:
            data_list.append(temp_file)

    sample_list = random.sample(data_list, n)

    if data_type == 'path':
        return sample_list
    elif data_type == 'Aircraft':

        sample_list = [Aircraft(temp_file) for temp_file in sample_list]
        return sample_list
    else:
        print("Invalid sample type")
        return


def load_data_by_date(date, dataPath=data_path, data_type='path', std_alt=None):

    date = str(date)

    file_path = os.path.join(dataPath, "ASDE_" + date)
    data_list = load_file_list(file_path, data_type='csv', parent_path=True)

    if data_type == 'path':
        return data_list
    elif data_type == 'Aircraft':
        if dataPath == data_path_detected:
            data_list = [Aircraft(temp_file, preprocessed=True, std_alt=std_alt) for temp_file in data_list]
        else:
            data_list = [Aircraft(temp_file) for temp_file in data_list]
        return data_list
    else:
        print("Invalid sample type")
        return


def get_color(value, vmin=0, vmax=100, alpha=1, color_map='cool'):
    # 색상 맵 설정 (viridis, plasma, inferno 등 사용 가능)
    cmap = plt.get_cmap(color_map)  # colormap 설정
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)  # 값의 범위를 0-1로 정규화
    rgba_color = cmap(norm(value))  # 값을 색상에 매핑
    # RGBA 값을 0-255 범위로 변환
    color_li = [int(c * 255) for c in rgba_color]
    color_li[-1] = color_li[-1] * alpha
    return color_li


def get_save_path():
    return '/Volumes/T7 Shield/Data'


def check_date(sec, tz=9, format="%Y%m%d"):
    if isinstance(sec, str):
        sec = int(sec)

    dt_utc = datetime.fromtimestamp(sec, tz=timezone.utc)
    kst = timezone(timedelta(hours=tz))
    dt = dt_utc.astimezone(kst)
    date = dt.strftime(format)

    return date


def convert_date(date, tz=9, format="%Y%m%d"):
    if isinstance(date, int|float):
        date = str(date)

    tz = timezone(timedelta(hours=tz))
    dt = datetime.strptime(date, format)
    dt = dt.replace(tzinfo=tz)

    utc_seconds = int(dt.timestamp())

    return utc_seconds


def convert_day(date):
    date = datetime.strptime(str(date), "%Y%m%d")  # 문자열을 날짜로 변환
    return date.strftime("%A")  # 요일 반환 (예: Sunday)


def cal_date(date_str, delta):
    # 문자열을 datetime 객체로 변환
    date_obj = datetime.strptime(str(date_str), "%Y%m%d")

    # 7일 더하기
    new_date_obj = date_obj + timedelta(days=delta)

    # 결과 출력 (%Y%m%d 포맷으로 변환)
    new_date_str = new_date_obj.strftime("%Y%m%d")

    return int(new_date_str)


def convert_time(time_utc_sec, time_format="%y-%m-%d %H:%M:%S"):
    dt = datetime.fromtimestamp(int(time_utc_sec), KST)
    time_str = dt.strftime(time_format)

    return time_str


def log_text(file_path, text, reset=False):

    if reset:
        write_type = "w"
    else:
        write_type = "a"

    with open(file_path, write_type, encoding="utf-8") as file:
        file.write(text + "\n")


def dict_to_json(file_path, Dict):

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(Dict, f, ensure_ascii=False, indent=4)


def json_to_dict(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)

    return loaded_data


def merge_DataFrame(path_list:list, chunk:int=50, time_range_utc=None):

    merged_df = pd.DataFrame()
    path_chunks = [path_list[i:i + chunk] for i in range(0, len(path_list), chunk)]

    if time_range_utc is not None:
        start_utc = time_range_utc[0]
        end_utc = time_range_utc[1]
    else:
        start_utc = 0
        end_utc = 0

    for path_list in path_chunks:

        temp_merged_df = pd.DataFrame()

        for path in path_list:
            if time_range_utc is not None:

                lines = get_start_end_line(path)

                if len(lines) > 1:

                    temp_start = lines[1].split(",")[6]
                    temp_end = lines[2].split(",")[6]

                    if temp_end < start_utc:
                        continue

                    if temp_start > end_utc:
                        continue

                else:
                    continue

            temp_df = pd.read_csv(path, dtype={18: str})
            temp_merged_df = pd.concat([temp_merged_df, temp_df], ignore_index=True)

        merged_df = pd.concat([merged_df, temp_merged_df])

    # for path in tqdm(path_list):
    #     temp_df = pd.read_csv(path)
    #     merged_df = pd.concat([merged_df, temp_df], ignore_index=True)
    #
    if merged_df.empty:
        return None

    else:
        merged_df.sort_values(by=['TimeSnapShot(sec-utc)'], ascending=True, inplace=True)
        merged_df.reset_index(drop=True, inplace=True)

        return merged_df


def get_start_end_line(file_path):
    lines = []

    # 첫 번째, 두 번째 줄 읽기
    with open(file_path, 'rb') as f:
        for _ in range(2):
            line = f.readline()
            if not line:
                break
            lines.append(line.decode('utf-8').strip())

    # 마지막 줄 읽기
    with open(file_path, 'rb') as f:
        f.seek(0, 2)  # 파일 끝으로 이동
        pos = f.tell()
        line = b''
        while pos > 0:
            pos -= 1
            f.seek(pos)
            char = f.read(1)
            if char == b'\n' and line:
                break
            line = char + line
        last_line = line.decode('utf-8').strip()

    # 중복 제거 (예: 파일이 1~2줄밖에 없을 때 마지막 줄이 이미 포함된 경우)
    if last_line not in lines:
        lines.append(last_line)

    return lines


def get_start_end_time(file_path):

    lines = get_start_end_line(file_path)

    if len(lines) > 1:
        start_time = lines[1][6]
        end_time = lines[2][6]

        return [start_time, end_time]

    else:
        return None

def dict_to_json(dict_file):
    return json.dumps(dict_file, default=int)

def to_json(json_file_path, json_file):
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(json_file, f, indent=2, ensure_ascii=False, default=int)

    # create Aircraft Image
    # for temp_air, temp_flight in zip(['air_cyan', 'air_magenta', 'air_yellow'], ['arr', 'dep', 'ukn']):
    #     old_image = Image.open(f"/Users/parkminkyun/Python/GroundAnalysis/TaxiTimeAnalysis/folium/{temp_air}.png")
    #     image = old_image.resize((150, 150))
    #
    #     for i in range(360):
    #         new = image.rotate(-1 * i)
    #         new.save(f'./folium/{temp_flight}_img/image{i}.png')

# 날짜 포맷 코드
# -------------------------------
# %Y     4자리 연도 (e.g., 2025)
# %y     2자리 연도 (e.g., 25)
# %m     월 (01–12)
# %B     월 이름 전체 (e.g., April)
# %b     월 이름 축약 (e.g., Apr)
# %d     일 (01–31)
# %j     1년 중 몇 번째 날 (001–366)
# %U     1년 중 몇 번째 주 (일요일 시작)
# %W     1년 중 몇 번째 주 (월요일 시작)

# 시간 포맷 코드
# -------------------------------
# %H     시 (24시간제, 00–23)
# %I     시 (12시간제, 01–12)
# %p     AM / PM
# %M     분 (00–59)
# %S     초 (00–59)
# %f     마이크로초 (000000–999999)

# 요일 및 기타
# -------------------------------
# %A     요일 이름 전체 (e.g., Monday)
# %a     요일 이름 축약 (e.g., Mon)
# %w     요일 숫자 (0=일요일, 6=토요일)
# %c     로컬 날짜 및 시간 표현 (e.g., Mon Apr  9 15:30:00 2025)
# %x     로컬 날짜 (e.g., 04/09/25)
# %X     로컬 시간 (e.g., 15:30:00)
# %Z     타임존 이름 (e.g., KST)
# %z     타임존 오프셋 (e.g., +0900)
# %%     % 문자 자체 출력










