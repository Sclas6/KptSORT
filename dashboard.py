import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import dash
from dash.dependencies import Input, Output, State
from dash import html, dcc
import dash_bootstrap_components as dbc
from pathlib import Path
import plotly.graph_objects as go
import pickle
import base64
import struct
import numpy as np
import cv2
import dash_player as dp
from flask import send_from_directory
from tools.generate_graph import gen_graphs

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

dir_1 = f"output/{sys.argv[1]}"
dir_2 = f"output/{sys.argv[2]}"
path_video_1 = get_video(dir_1)
path_video_2 = get_video(dir_2)
with open(f"{dir_1}/bees.pkl", "rb") as f:
    bees_1 = pickle.load(f)
with open(f"{dir_2}/bees.pkl", "rb") as f:
    bees_2 = pickle.load(f)

cap = cv2.VideoCapture(path_video_1)  # video_name is the video being called
fps_1 = cap.get(cv2.CAP_PROP_FPS)
frames_1 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap = cv2.VideoCapture(path_video_2)
fps_2 = cap.get(cv2.CAP_PROP_FPS)
frames_2 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

figs = gen_graphs("test/", bees_1, bees_2, fps_1)

"""
with open("plt_series.pkl", mode="rb") as f:
    fig_lines = pickle.load(f)
""" 
"""
with open("tools/figs.pkl", mode="rb") as f:
    figs = pickle.load(f)
"""

#data_x = [i for i in range(frames_1)]
#data_y = decode_plotly_bdata(fig_lines['data'][0]['y'])
# 初期グラフの作成
fig = go.Figure()
"""
fig_lines.add_trace(
    go.Scatter(x=[fig_lines["data"][2]["x"][0]],
               y=[data_y[0]],
               mode='markers',
               marker=dict(size=20, color='yellow'),
               name='Current Position'))
fig_lines.update_layout(xaxis={"range": [100 - .5, 1000 + .5]},
                        hovermode="x unified")
"""

asst_path = os.path.join(os.getcwd(), "assets")
app = dash.Dash(__name__, assets_folder=asst_path, external_stylesheets=[dbc.themes.BOOTSTRAP])


@app.server.route("/video1")
def serve_video_1():
    return send_from_directory(dir_1, path_video_1.name)

@app.server.route("/video2")
def serve_video_2():
    return send_from_directory(dir_2, path_video_2.name)

SCROLLABLE_COL_STYLE = {
    'maxHeight': '100vh',
    'overflowY': 'auto',
    'padding': '15px',
    'border': '1px solid #ddd'
}

app.layout = dbc.Container([
    dbc.Row(
            dbc.Col(
                width=12
            )
        ),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    dp.DashPlayer(id="player1",
                                url="http://172.16.15.116:8050/video1",
                                controls=False,
                                width="100%",
                                height=None,
                                className="video1"),
                    dcc.Slider(id="data_index_slider",
                            min=0,
                            max=frames_1,
                            value=100,
                            step=1,
                            updatemode="drag",
                            marks={
                                i: f"{i}"
                                for i in range(frames_1) if i % int(frames_1 / 20) == 0
                            },
                            className="video-slider",
                            tooltip={
                                "placement": "bottom",
                                "always_visible": False
                            }),
                ],
                        className="player1"),
                html.Div(
                    [html.Button("◀", id="btn_prev"),
                    html.Button("▶", id="btn_next"),
                    dcc.Input(
                        id="th",
                        type="number",
                        placeholder="input threshould",
                    )]
                ),
            ]),
            html.Div([
                html.Div([
                    dp.DashPlayer(id="player2",
                                url="http://172.16.15.116:8050/video2",
                                controls=False,
                                width="100%",
                                height=None,
                                className="video2"),

                ],
                        className="player2"),
            ])
        ], width=5, style=SCROLLABLE_COL_STYLE),
        dbc.Col([
            html.Div([
                dcc.Graph(id="fig_TotalDistanceTraveled", figure=figs["TotalDistanceTraveled"], style={'width': '95vh', 'height': '45vh'}),
                dcc.Graph(id="fig_TotalRearingTime", figure=figs["TotalRearingTime"], style={'width': '95vh', 'height': '45vh'}),
                dcc.Graph(id="fig_TotalTrophallaxisTime", figure=figs["TotalTrophallaxisTime"], style={'width': '95vh', 'height': '45vh'}),
                dcc.Graph(id="fig_Caring_Network", figure=figs["Caring_Network"], style={'width': '95vh', 'height': '75vh'}),
                dcc.Graph(id="fig_Caring_Heatmap", figure=figs["Caring_Heatmap"], style={'width': '95vh', 'height': '75vh'}),
                dcc.Graph(id="fig_Trophallaxis_Network", figure=figs["Trophallaxis_Network"], style={'width': '95vh', 'height': '75vh'}),
                dcc.Graph(id="fig_Trophallaxis_Heatmap", figure=figs["Trophallaxis_Heatmap"], style={'width': '95vh', 'height': '75vh'}),
                
                #dcc.Graph(id="line_chart", figure=fig_lines),      
            ])
        ], width=7, style=SCROLLABLE_COL_STYLE)
    ])  
],fluid=True,
                      className="dashboard")
"""
@app.callback([
    Output(component_id="player1",
           component_property="seekTo",
           allow_duplicate=True),
    Output(component_id="player2",
           component_property="seekTo",
           allow_duplicate=True),
    Output(component_id="line_chart",
           component_property="extendData",
           allow_duplicate=True),
], [Input(component_id="data_index_slider", component_property="value")],
              prevent_initial_call=True)

def update_graph_(index):
    seek_to = (index + .5) / fps_1

    return seek_to, seek_to, [
        dict(x=[[data_x[index]]], y=[[data_y[index - 100]]]), [-1], 1
    ]
"""
@app.callback([
    Output(component_id="fig_TotalRearingTime", component_property="figure", allow_duplicate=True),
    Output(component_id="fig_TotalTrophallaxisTime", component_property="figure", allow_duplicate=True),
    Output(component_id="fig_Caring_Network", component_property="figure", allow_duplicate=True),
    Output(component_id="fig_Caring_Heatmap", component_property="figure", allow_duplicate=True),
    Output(component_id="fig_Trophallaxis_Network", component_property="figure", allow_duplicate=True),
    Output(component_id="fig_Trophallaxis_Heatmap", component_property="figure", allow_duplicate=True),
], [Input(component_id="th", component_property="value", allow_optional=True)],
              prevent_initial_call=True)
def update_graph(th):
    figs = gen_graphs("test/", bees_1, bees_2, th)
    return list(figs.values())[1:]


@app.callback([
    Output(component_id="player1", component_property="seekTo", allow_duplicate=True),
    Output(component_id="player2", component_property="seekTo", allow_duplicate=True),
    #Output(component_id="line_chart", component_property="extendData", allow_duplicate=True),
    Output(component_id="data_index_slider", component_property="value", allow_duplicate=True)
], [Input(component_id="btn_next", component_property="n_clicks")],
              State(component_id="data_index_slider",
                    component_property="value"),
              prevent_initial_call=True)

def frame_next(_, index):
    seek_to = (index + .5) / fps_1

    #return seek_to, seek_to, [dict(x=[[data_x[index]]], y=[[data_y[index - 100]]]), [-1], 1], index + 1
    return seek_to, seek_to, index + 1


@app.callback([
    Output(component_id="player1", component_property="seekTo"),
    Output(component_id="player2", component_property="seekTo"),
    #Output(component_id="line_chart", component_property="extendData"),
    Output(component_id="data_index_slider", component_property="value")
], [Input(component_id="btn_prev", component_property="n_clicks")],
              State(component_id="data_index_slider",
                    component_property="value"))

def frame_prev(_, index):
    seek_to = (index + .5) / fps_1

    #return seek_to, seek_to, [dict(x=[[data_x[index]]], y=[[data_y[index - 100]]]), [-1], 1], index - 1
    return seek_to, seek_to, index - 1

if __name__ == '__main__':
    app.run(debug=True, port="8050", host="0.0.0.0")
