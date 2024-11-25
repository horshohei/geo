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
from PIL import Image
from io import BytesIO

load_dotenv()

# OpenAI APIキーを設定
openai_api_key: str = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
client = OpenAI(api_key=openai_api_key)


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
                    {"type": "text", "text": question.format(image_name1, image_name2)},
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

# POIの中心地点を地図上に表示する関数、zoomは17か18を推奨、17は周辺300m程度、18は周辺50m程度が描写される
def create_poi_center_map(poi: dict, zoom=18) -> px.scatter_mapbox:
    df: pd.DataFrame = pd.DataFrame([poi], columns=['lat', 'lon'])
    fig: px.scatter_mapbox = px.scatter_mapbox(df, lat='lat', lon='lon', zoom=zoom, height=1200)
    fig.update_layout(mapbox_style="open-street-map")
    return fig

def make_poi_data(data: dict) -> list[dict]:
    poi_data: list[dict] = []
    for p in data['elements']:
        poi: dict = {'id': p['id'], 'lat': p['lat'], 'lon': p['lon']}
        if 'tags' in p:
            if 'name' in p['tags']:
                poi['name'] = p['tags']['name']
            if 'amenity' in p['tags']:
                poi['amenity'] = p['tags']['amenity']
        poi_data.append(poi)
    return poi_data

# Google Street View APIを使用してストリートビュー画像を取得する関数
def get_street_view_image(lat: float, lon: float, heading: int = 0, pitch: int = 10, fov: int = 120) -> bytes:
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": "640x480",  # 画像サイズ
        "location": f"{lat},{lon}",
        "heading": heading,
        "pitch": pitch,
        "fov": fov,
        "source": "outdoor",
        "key": google_api_key
    }
    response = requests.get(base_url, params=params)
    return response.content
#%%
# 画像を指定したDPIとサイズで保存する関数
def save_image_with_dpi(image_data: bytes, file_path: str, dpi: int = 600, size: tuple = (2048, 1024)):
    image = Image.open(BytesIO(image_data))
    #image = image.resize(size)
    image.save(file_path, format='PNG', dpi=(dpi, dpi))


#%%


# 使用例
center_lat = 35.6586  # 中心地点の緯度
center_lon = 139.7454  # 中心地点の経度
bbox = create_bbox(center_lat, center_lon, distance=0.005) #中心地点を中心としたバウンディングボックスを生成 0.005度（約1kmの四角形）　0.01度で約2km

amenity: str = "cafe" # カフェを検索 現在はamenity属性のみ対応
threshold: float = 50  # 200m以内にあるPOIを抽出

# 指定されたバウンディングボックス内の指定されたamenityのPOIを取得
data: dict = get_amenity_pois(bbox, amenity)
# 処理しやすい形式に変換
poi_data: list[dict] = make_poi_data(data)
# しきい値以下の距離にあるPOIのペアを抽出
close_pois: list[tuple[dict, dict, float]] = filter_pois_by_distance(poi_data, threshold)

# 地図上にPOIを表示
map_object = create_map(poi_data)
highlightmap = highlight_differences_on_map(map_object, close_pois)
# 地図を保存
highlightmap.save('TokyoTower.html')
#%%
# Close POI pairsについて、それぞれのPOIを地図上に表示してPNG画像として保存、現在はpairsディレクトリに保存
os.makedirs('pairs', exist_ok=True)
for idx, (poi1, poi2, _) in enumerate(close_pois, start=1):
    name1 = poi1.get('name', 'unknown').replace(' ', '_')
    name2 = poi2.get('name', 'unknown').replace(' ', '_')

    fig1 = create_poi_center_map(poi1, zoom=18)
    fig2 = create_poi_center_map(poi2, zoom=18)

    fig1.write_image(f"pairs/{idx}-1_{name1}.png", width=1600, height=1200)
    fig2.write_image(f"pairs/{idx}-2_{name2}.png", width=1600, height=1200)
#%%
# 2つの画像をOpenAI APIに送信して質問を行う
image_path1 = "pairs/1-1_カフェ・ベローチェ.png"
image_path2 = "pairs/1-2_宮越屋珈琲.png"
prompt = "{image_name1}と{image_name2}という2つのカフェの場所について、これらについての地図画像を見て、地図上の違いを教えてください。"

response: dict = ask_openai_question(image_path1, image_path2, prompt)

# OpenAI APIの応答を表示
print(response.choices[0].message.content)

#%%
# 2つの画像をGoogle Street View APIを使用して取得
os.makedirs('streetview_images', exist_ok=True)
for idx, (poi1, poi2, _) in enumerate(close_pois, start=1):
    lat1, lon1 = poi1['lat'], poi1['lon']
    lat2, lon2 = poi2['lat'], poi2['lon']

    image1 = get_street_view_image(lat1, lon1)
    image2 = get_street_view_image(lat2, lon2)

    save_image_with_dpi(image1, f"streetview_images/{idx}_1_{poi1['name']}.png", dpi=600)
    save_image_with_dpi(image2, f"streetview_images/{idx}_2_{poi2['name']}.png", dpi=600)

