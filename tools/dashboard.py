import dash
from dash.dependencies import Input, Output, State
from dash import html, dcc
from pathlib import Path
import plotly.graph_objects as go
import pickle
import base64
import struct
import numpy as np
import cv2
import dash_player as dp
import os
import sys
from flask import send_from_directory

def get_video(dir):
    max_frames = -1
    video = None
    for f in Path(dir).iterdir():
        if str(f)[-4:] == ".mp4":
            cap = cv2.VideoCapture(f)
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, 'little').decode('utf-8')
            if fourcc == "h264":
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                max_frames = max(max_frames, frames)
                if max_frames == frames:
                    video = f
            cap.release()
    return video


def gen(cap):
    while True:
        _, image = cap.read()
        _, frame = cv2.imencode('.jpg', image)
        if frame is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame.tobytes() +
                   b"\r\n")
        else:
            print("frame is none")


def decode_plotly_bdata(bdata_dict):
    binary_data = base64.b64decode(bdata_dict['bdata'])
    plotly_dtype_map = {'f8': 'd', 'i8': 'q', 'i4': 'i', 'f4': 'f'}
    struct_format = f"<{len(binary_data) // struct.calcsize(plotly_dtype_map[bdata_dict['dtype']])}{plotly_dtype_map[bdata_dict['dtype']]}"
    unpacked_data = struct.unpack(struct_format, binary_data)
    return np.array(unpacked_data)

dir = f"/kpsort/output/{sys.argv[1]}"
path_video = get_video(dir)

cap = cv2.VideoCapture(path_video)  # video_name is the video being called
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
fps = cap.get(cv2.CAP_PROP_FPS)
ret, frame = cap.read()
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video = go.Figure()
video.add_trace(
    go.Image(z=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), hoverinfo="none"))

with open("/kpsort/plt_series.pkl", mode="rb") as f:
    fig_lines = pickle.load(f)

data_x = [i for i in range(frames)]
data_y = decode_plotly_bdata(fig_lines['data'][0]['y'])
# 初期グラフの作成
fig = go.Figure()
fig_lines.add_trace(
    go.Scatter(x=[fig_lines["data"][2]["x"][0]],
               y=[data_y[0]],
               mode='markers',
               marker=dict(size=20, color='yellow'),
               name='Current Position'))
fig_lines.update_layout(xaxis={"range": [100 - .5, 1000 + .5]},
                        hovermode="x unified")

# dashアプリケーションの定義
asst_path = os.path.join(os.getcwd(), "assets")
app = dash.Dash(__name__, assets_folder=asst_path)


@app.server.route("/video")
def serve_video():
    return send_from_directory(dir, path_video.name)


# レイアウトの定義
app.layout = html.Div([
    html.Div([
        dp.DashPlayer(id="player",
                      url="http://127.0.0.1:8050/video",
                      controls=False,
                      width=None,
                      height=None,
                      className="video"),
        dcc.Slider(id="data_index_slider",
                   min=0,
                   max=frames,
                   value=100,
                   step=1,
                   updatemode="drag",
                   marks={
                       i: f"{i}"
                       for i in range(frames) if i % int(frames / 20) == 0
                   },
                   className="video-slider",
                   tooltip={
                       "placement": "bottom",
                       "always_visible": False
                   }),
    ],
             className="player"),
    html.Div(
        [html.Button("◀", id="btn_prev"),
         html.Button("▶", id="btn_next")]),
    dcc.Graph(id="line_chart", figure=fig_lines),
],
                      className="dashboard")

@app.callback([
    Output(component_id="player",
           component_property="seekTo",
           allow_duplicate=True),
    Output(component_id="line_chart",
           component_property="extendData",
           allow_duplicate=True),
], [Input(component_id="data_index_slider", component_property="value")],
              prevent_initial_call=True)

def update_graph(index):
    seek_to = (index + .5) / fps

    return seek_to, [
        dict(x=[[data_x[index]]], y=[[data_y[index - 100]]]), [-1], 1
    ]


@app.callback([
    Output(component_id="player",
           component_property="seekTo",
           allow_duplicate=True),
    Output(component_id="line_chart",
           component_property="extendData",
           allow_duplicate=True),
    Output(component_id="data_index_slider",
           component_property="value",
           allow_duplicate=True)
], [Input(component_id="btn_next", component_property="n_clicks")],
              State(component_id="data_index_slider",
                    component_property="value"),
              prevent_initial_call=True)

def frame_next(_, index):
    seek_to = (index + .5) / fps

    return seek_to, [
        dict(x=[[data_x[index]]], y=[[data_y[index - 100]]]), [-1], 1
    ], index + 1


@app.callback([
    Output(component_id="player", component_property="seekTo"),
    Output(component_id="line_chart", component_property="extendData"),
    Output(component_id="data_index_slider", component_property="value")
], [Input(component_id="btn_prev", component_property="n_clicks")],
              State(component_id="data_index_slider",
                    component_property="value"))

def frame_prev(_, index):
    seek_to = (index + .5) / fps

    return seek_to, [
        dict(x=[[data_x[index]]], y=[[data_y[index - 100]]]), [-1], 1
    ], index - 1


# アプリケーションの実行
if __name__ == '__main__':
    #filename = Path(sys.argv[1]).stem
    #print(filename)
    app.run(debug=True, port="8050")
