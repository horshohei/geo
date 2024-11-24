import requests
from geopy.distance import geodesic
import folium
import plotly.express as px
import pandas as pd
import os
import base64
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# OpenAI APIキーを設定
api_key: str = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 特定の地域の類似したPOIの一覧を取得する関数
def get_amenity_pois(bbox: str, amenity: str) -> dict:
    overpass_url: str = "http://overpass-api.de/api/interpreter"
    overpass_query: str = f"""
    [out:json];
    node["amenity"="{amenity}"]({bbox});
    out body;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    data: dict = response.json()
    return data

# 指定された地点を中心とした四角形のバウンディングボックスを生成する関数 デフォルトは0.005度（約1kmの四角形）
def create_bbox(lat, lon, distance=0.005):
    """
    指定された地点を中心とした四角形のバウンディングボックスを生成する関数。
    :param lat: 中心地点の緯度
    :param lon: 中心地点の経度
    :param distance: 中心地点からの距離（度単位）。デフォルトは0.005度（約1kmの四角形）
    :return: bbox文字列
    """
    lat_min = lat - distance
    lat_max = lat + distance
    lon_min = lon - distance
    lon_max = lon + distance
    return f"{lat_min},{lon_min},{lat_max},{lon_max}"

# POIのリストから指定された距離以内にあるPOIのペアを抽出する関数
def filter_pois_by_distance(pois: list[dict], threshold: float) -> list[tuple[dict, dict, float]]:
    close_pois: list[tuple[dict, dict, float]] = []
    for i in range(len(pois)):
        for j in range(i + 1, len(pois)):
            distance: float = geodesic([pois[i]['lat'], pois[i]['lon']], [pois[j]['lat'], pois[j]['lon']]).meters
            if distance < threshold:
                close_pois.append((pois[i], pois[j], distance))
    return close_pois

# POIを地図上に表示する関数
def create_map(poi_data: list[dict]) -> folium.Map:
    # 地図の中心を計算
    center_lat = sum(poi['lat'] for poi in poi_data) / len(poi_data)
    center_lon = sum(poi['lon'] for poi in poi_data) / len(poi_data)

    # 地図を作成
    m = folium.Map(location=[center_lat, center_lon], zoom_start=16)

    # POIを地図上に追加
    for poi in poi_data:
        popup_text = poi.get('name', 'No name')
        folium.Marker(
            location=[poi['lat'], poi['lon']],
            popup=popup_text
        ).add_to(m)

    return m

# OpenAI APIにリクエストを送信する関数
def ask_openai_question(image_path1: str, image_path2: str, question: str) -> dict:
    with open(image_path1, "rb") as image_file1, open(image_path2, "rb") as image_file2:
        base64_image1: str = base64.b64encode(image_file1.read()).decode('utf-8')
        base64_image2: str = base64.b64encode(image_file2.read()).decode('utf-8')

    image_name1: str = re.search(r"pairs/(.*)\.png", image_path1).group(1)
    image_name2: str = re.search(r"pairs/(.*)\.png", image_path2).group(1)

    response: dict = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{image_name1}と{image_name2}という2つのカフェの場所について、これらについての地図画像を見て、地図上の違いを教えてください。"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image1}"
                        },
                    },{
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image2}"
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    return response


# 地図上の違いを強調表示し、POI間に線を引く関数
def highlight_differences_on_map(map_object: folium.Map, close_pois: list[tuple[dict, dict, float]]) -> folium.Map:
    for poi1, poi2, _ in close_pois:
        # POI間に線を引く
        folium.PolyLine(
            locations=[(poi1['lat'], poi1['lon']), (poi2['lat'], poi2['lon'])],
            color='red',
            weight=2.5,
            opacity=1
        ).add_to(map_object)

        # POIをマーカーで表示
#        folium.Marker(location=(poi1['lat'], poi1['lon']), icon=folium.Icon(color='red')).add_to(map_object)
#        folium.Marker(location=(poi2['lat'], poi2['lon']), icon=folium.Icon(color='red')).add_to(map_object)

    return map_object

# POIの中心地点を地図上に表示する関数
def create_poi_center_map(poi: dict) -> px.scatter_mapbox:
    df: pd.DataFrame = pd.DataFrame([poi], columns=['lat', 'lon'])
    fig: px.scatter_mapbox = px.scatter_mapbox(df, lat='lat', lon='lon', zoom=18, height=1200)
    fig.update_layout(mapbox_style="open-street-map")
    return fig

#%%
# 使用例
center_lat = 35.6586  # 中心地点の緯度
center_lon = 139.7454  # 中心地点の経度
bbox = create_bbox(center_lat, center_lon)

amenity: str = "cafe"
threshold: float = 50  # 200m以内にあるPOIを抽出

data: dict = get_amenity_pois(bbox, amenity)
#%%
poi_data: list[dict] = []
for p in data['elements']:
    poi: dict = {'id': p['id'], 'lat': p['lat'], 'lon': p['lon']}
    if 'tags' in p:
        if 'name' in p['tags']:
            poi['name'] = p['tags']['name']
        if 'amenity' in p['tags']:
            poi['amenity'] = p['tags']['amenity']
    poi_data.append(poi)



close_pois: list[tuple[dict, dict, float]] = filter_pois_by_distance(poi_data, threshold)

map_object = create_map(poi_data)
highlightmap = highlight_differences_on_map(map_object, close_pois)

highlightmap.save('TokyoTower.html')
#%%
os.makedirs('pairs', exist_ok=True)
# Close POI pairsについて、それぞれのPOIを地図上に表示してPNG画像として保存
for idx, (poi1, poi2, _) in enumerate(close_pois, start=1):
    name1 = poi1.get('name', 'unknown').replace(' ', '_')
    name2 = poi2.get('name', 'unknown').replace(' ', '_')

    fig1 = create_poi_center_map(poi1)
    fig2 = create_poi_center_map(poi2)

    fig1.write_image(f"pairs/{idx}-1_{name1}.png", width=1600, height=1200)
    fig2.write_image(f"pairs/{idx}-2_{name2}.png", width=1600, height=1200)
#%%
image_path1 = "pairs/1-1_カフェ・ベローチェ.png"
image_path2 = "pairs/1-2_宮越屋珈琲.png"

response: dict = ask_openai_question(image_path1, image_path2, "What are the differences between these two cafes?")

#%%
print(response.choices[0].message.content)
#bbox = "35.6586,139.7454,35.6895,139.7690"# 東京タワー周辺のバウンディングボックス
