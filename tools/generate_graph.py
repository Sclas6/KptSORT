from AssignBeeHive import Bee, CaringEvent, TrophallaxisEvent
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import japanize_matplotlib
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import collections
import networkx as nx
from networkx.drawing.layout import bipartite_layout
import numpy as np

def export_to_csv(df: pd.DataFrame, path_out: str, filename: str):
    """DataFrameをCSVファイルとして出力するヘルパー関数"""
    try:
        df.to_csv(f"{path_out}{filename}", index=True)
    except Exception as e:
        print(e)
        
def create_caring_adj_matrix(edges, bee_ids, hive_ids):
    """
    ハチと幼虫のエッジリストから隣接行列（相互作用の頻度）を作成する。
    行: ハチID, 列: 幼虫ID
    """
    # ユニークなIDを抽出し、ソート
    sorted_bee_ids = sorted(list(bee_ids))
    sorted_hive_ids = sorted(list(hive_ids))

    bee_to_index = {id: i for i, id in enumerate(sorted_bee_ids)}
    hive_to_index = {id: j for j, id in enumerate(sorted_hive_ids)}
    
    n_bees = len(sorted_bee_ids)
    n_hives = len(sorted_hive_ids)
    
    adj_matrix = np.zeros((n_bees, n_hives), dtype=int)
    
    for bee_id, hive_id, count in edges:
        i = bee_to_index.get(bee_id)
        j = hive_to_index.get(hive_id)

        if i is not None and j is not None:
            adj_matrix[i, j] += count
            
    df_adj = pd.DataFrame(adj_matrix, index=sorted_bee_ids, columns=sorted_hive_ids)
    
    df_adj.index = df_adj.index.astype(str)
    df_adj.columns = df_adj.columns.astype(str)
    
    return df_adj

def create_adj_matrix(edges, bee_ids):
    """
    エッジリストから隣接行列（相互作用の頻度）を作成する。
    
    :param edges: (bee_id_A, bee_id_B, count) のタプルのリスト。
    :param bee_ids: コロニー内の全てのハチのIDのリスト。
    :return: 相互作用の頻度を示すDataFrame形式の隣接行列。
    """
    sorted_ids = sorted(list(bee_ids))
    id_to_index = {id: i for i, id in enumerate(sorted_ids)}
    n = len(sorted_ids)
    
    adj_matrix = np.zeros((n, n), dtype=int)
    
    for id1, id2, count in edges:
        i = id_to_index.get(id1)
        j = id_to_index.get(id2)

        if i is not None and j is not None:
            adj_matrix[i, j] += count
            adj_matrix[j, i] += count
            
    df_adj = pd.DataFrame(adj_matrix, index=sorted_ids, columns=sorted_ids)
    df_adj.columns = df_adj.columns.astype(str)
    df_adj.index = df_adj.index.astype(str)
    return df_adj

def gen_network(edges, max_weight_global, title=""):
    G = nx.Graph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=w * 10)
    if not G.edges():
        return go.Figure()
    
    num_nodes = len(G.nodes())
    k_value = 1.0 / (num_nodes ** 0.5) * 1.5
    
    pos = nx.spring_layout(
        G, 
        k=k_value * 1000,
                         
        iterations=100,  
        seed=42,         
        center=[0.5, 0.5],
        scale=0.8        
    )

    edge_weights_dict = nx.get_edge_attributes(G, 'weight')
    all_weights = list(edge_weights_dict.values())
    #max_weight = max(all_weights) if all_weights else 1
    
    edge_traces = []
    
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        
        #scaled_width = (weight / max_weight) * 5
        scaled_width = (weight / max_weight_global) * 5
        
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        trace_edge = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=scaled_width, color='black'),
            hoverinfo='text',
            mode='lines',
            opacity=0.7,
            text=[f"相互作用: {u} - {v}<br>回数: {weight/10:.1f}"],
            showlegend=False
        )
        edge_traces.append(trace_edge)

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    node_sizes = [15] * len(G.nodes())
    node_text = [f"ID: {node}" for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=list(G.nodes()),
        textposition='middle center',
        marker=dict(
            showscale=False,
            size=40,
            color='white',
            line_width=2,
            line_color='black'
        ),
        textfont=dict(size=12, color='black'),
        hovertext=node_text,
        showlegend=False
    )

    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(
                        title={
                            'text': title,
                            'x': 0.5, 
                            'xanchor': 'center',
                            'font': {'size': 18} 
                        },
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    )
    )
    
    # 描画範囲の調整
    """
    x_min, x_max = min(node_x), max(node_x)
    y_min, y_max = min(node_y), max(node_y)
    fig.update_layout(
        xaxis_range=[x_min - 0.1, x_max + 0.1],
        yaxis_range=[y_min - 0.1, y_max + 0.1]
    )
    """
    return fig

def gen_bipartite_network(nodes_left, legend_left, nodes_right, legend_right, edges, max_weight_global, title=""):
    B = nx.Graph()
    for e in edges: B.add_edge(e[0], e[1], weight=e[2])
    node_bees_unique = list(set(nodes_left))
    B.add_nodes_from(node_bees_unique, bipartite=1)
    node_hives_unique = list(set(nodes_right))
    B.add_nodes_from(node_hives_unique, bipartite=0)
    top_nodes = {n for n, d in B.nodes(data=True) if d.get('bipartite') == 1}

    pos = bipartite_layout(B, top_nodes)

    edge_x = []
    edge_y = []
    hover_text_edges = []
    #max_weight = max(nx.get_edge_attributes(B, 'weight').values()) if nx.get_edge_attributes(B, 'weight') else 1

    for edge in B.edges(data=True):
        x0, y0 = pos[edge[0]] # ノードUの座標
        x1, y1 = pos[edge[1]] # ノードVの座標
        
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        weight = edge[2].get('weight', 1)
        hover_text_edges.append(f"相互作用の重み: {weight/10:.1f}")
    edge_traces = []
    current_edge_index = 0

    for edge in B.edges(data=True):
        weight = edge[2].get('weight', 1)
        
        #line_width = weight / max_weight * 5 if max_weight > 0 else 1 
        line_width = weight / max_weight_global * 5 if max_weight_global > 0 else 1
        
        start_index = current_edge_index * 3
        end_index = start_index + 2
        
        trace_edge = go.Scatter(
            x=edge_x[start_index:end_index],
            y=edge_y[start_index:end_index],
            line=dict(width=line_width, color='black'),
            hoverinfo='text',
            mode='lines',
            opacity=0.7,
            text=[f"{legend_left}: {edge[0]} - {legend_right}: {edge[1]}<br>重み: {weight/10:.1f}"],
            showlegend=False
        )
        edge_traces.append(trace_edge)
        current_edge_index += 1
    node_x = [pos[node][0] for node in B.nodes()]
    node_y = [pos[node][1] for node in B.nodes()]
    x_min, x_max = min(node_x), max(node_x)
    y_max = max(node_y)
    node_labels = list(B.nodes())
    hover_text_nodes = [f"ID: {node}" for node in B.nodes()]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_labels,
        textposition="middle center",
        textfont=dict(color='black', size=12, weight='bold'),
        
        marker=dict(
            showscale=False,
            size=40,
            color='white',
            line_width=2,
            line_color='black'
        ),
        hovertext=hover_text_nodes,
        showlegend=False
    )
    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(
                        title={
                            'text': f'{title}',
                            'x': 0.5, 
                            'xanchor': 'center',
                            'font': {'size': 16} 
                        },
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        annotations=[
                            go.layout.Annotation(
                                text=f"<b>{legend_left}</b>",
                                x=x_min,
                                y=y_max + 0.1,
                                xref="x",
                                yref="y",
                                showarrow=False,
                                font=dict(size=16, color="black"),
                                xanchor='left',
                                yanchor='bottom'
                            ),
                            go.layout.Annotation(
                                text=f"<b>{legend_right}</b>",
                                x=x_max,
                                y=y_max + 0.1,
                                xref="x",
                                yref="y",
                                showarrow=False,
                                font=dict(size=16, color="black"),
                                xanchor='right',
                                yanchor='bottom'
                            )
                        ]
                    )
    )
    x_min, x_max = min(node_x), max(node_x)
    y_min, y_max = min(node_y), max(node_y)

    fig.update_layout(
        xaxis_range=[x_min - 0.2, x_max + 0.2],
        yaxis_range=[y_min - 0.2, y_max + 0.2]
    )

    return fig

def gen_graphs(path_out: str, bees_1, bees_2, th=0):
    figs = dict()    
    #x_coords = np.random.normal(loc=1, scale=0.08, size=len([i.distance_sum for i in bees.values()]))
    
    df_0 = pd.DataFrame({'各個体の総移動距離': [i.distance_sum for i in bees_1.values()], 'Category': "5SP"})
    df_1 = pd.DataFrame({'各個体の総移動距離': [i.distance_sum for i in bees_2.values()], 'Category': "PBS"})

    # --- 統計量の計算とプリント ---
    for df, label in zip([df_0, df_1], ['Category A', 'Category B']):
        data = df['各個体の総移動距離']
        
        mean_val = data.mean()          # 平均値
        median_val = data.median()      # 中央値 (第2四分位数)
        q1 = data.quantile(0.25)        # 第1四分位数
        q3 = data.quantile(0.75)        # 第3四分位数
        
        print(f"--- {label} 統計量 ---")
        print(f"平均値:   {mean_val:.2f}")
        print(f"中央値:   {median_val:.2f}")
        print(f"第1四分位数: {q1:.2f}")
        print(f"第3四分位数: {q3:.2f}")
        print("-" * 20)
        
    # --- 描画処理 (既存コード) ---
    # ※結合して1つのデータフレームにすると描画がスムーズです
    df_all = pd.concat([df_0, df_1])
    sns.boxplot(x='Category', y='各個体の総移動距離', data=df_all, color='lightblue', ax=plt.gca(), showmeans=True, meanline=True)
    sns.stripplot(x='Category', y='各個体の総移動距離', data=df_all, color='darkgreen', s=7, alpha=0.6, jitter=0.0001, ax=plt.gca())

    plt.savefig(f"{path_out}TotalDistanceTraveled.png")
    plt.cla()

    time_caring_0 = [sum([e.duration for e in bee.event_caring if e.duration > 0]) for bee in bees_1.values()]
    time_caring_1 = [sum([e.duration for e in bee.event_caring if e.duration > 0]) for bee in bees_1.values()]
    df_0 = pd.DataFrame({'各個体の総育児時間': time_caring_0, 'Category': "5SP"})
    df_1 = pd.DataFrame({'各個体の総育児時間': time_caring_1, 'Category': "PBS"})
    sns.boxplot(x='Category', y='各個体の総育児時間', data=df_0, color='lightblue', ax=plt.gca())
    sns.stripplot(x='Category',y='各個体の総育児時間', data=df_0, color='darkgreen', s=7, alpha=0.6, jitter=0.0001, ax=plt.gca())
    sns.boxplot(x='Category', y='各個体の総育児時間', data=df_1, color='lightblue', ax=plt.gca())
    sns.stripplot(x='Category',y='各個体の総育児時間', data=df_1, color='darkgreen', s=7, alpha=0.6, jitter=0.0001, ax=plt.gca())
    plt.savefig(f"{path_out}TotalRearingTime.png")

    distance_data_0 = [i.distance_sum for i in bees_1.values()]
    distance_data_1 = [i.distance_sum for i in bees_2.values()]
    data_distance = {
        '各個体の総移動距離': distance_data_0 + distance_data_1,
        'Category': ["5SP"] * len(distance_data_0) + ["PBS"] * len(distance_data_1),
        "ID": list(bees_1.keys()) + list(bees_2.keys())
    }
    df_distance = pd.DataFrame(data_distance)
    export_to_csv(df_distance, path_out, "Individual_TotalDistanceTraveled.csv")
    fig_distance = go.Figure()
    for category in df_distance['Category'].unique():
        df_subset = df_distance[df_distance['Category'] == category]
        fig_distance.add_trace(go.Box(
            y=df_subset['各個体の総移動距離'],
            name=category,
            boxmean=True,
            boxpoints='all',  
            pointpos=0,
            jitter=0.0001,
            marker=dict(
                color='darkgreen',
                size=7,
                opacity=0.6,
                line=dict(width=0)
            ),
            line=dict(color='darkblue', width=1),
            fillcolor='lightblue',
            customdata=df_subset[["ID", "各個体の総移動距離"]],
            hovertemplate=(
            "<b>個体ID: %{customdata[0]}</b><br>" +
            "移動距離: %{customdata[1]}<br>" +
            "<extra></extra>"
            )
        )
    )

    fig_distance.update_layout(
        title={
            'text': '各個体の総移動距離',
            'x': 0.5, 
            'xanchor': 'center'
        },
        yaxis_title='各個体の総移動距離',
        showlegend=False,
    )

    fig_distance.write_html(f"{path_out}TotalDistanceTraveled.html")
    figs["TotalDistanceTraveled"] = fig_distance

    time_caring_0 = [sum([e.duration for e in bee.event_caring if e.duration > th]) for bee in bees_1.values()]
    time_caring_1 = [sum([e.duration for e in bee.event_caring if e.duration > th]) for bee in bees_2.values()]
    data_caring = {
        '各個体の総育児時間': time_caring_0 + time_caring_1,
        'Category': ["5SP"] * len(time_caring_0) + ["PBS"] * len(time_caring_1),
        'ID': list(bees_1.keys()) + list(bees_2.keys())
    }
    df_caring = pd.DataFrame(data_caring)
    export_to_csv(df_caring, path_out, "Individual_TotalRearingTime.csv")
    fig_caring = go.Figure()
    for category in df_caring['Category'].unique():
        df_subset = df_caring[df_caring['Category'] == category]
        fig_caring.add_trace(go.Box(
            x=df_subset['Category'], 
            y=df_subset['各個体の総育児時間'],
            name=category,
            boxpoints='all',  
            pointpos=0,       
            jitter=0.0001,       
            marker=dict(
                color='darkgreen', 
                size=7,
                opacity=0.6,
                line=dict(width=0)
            ),
            line=dict(color='darkblue', width=1), 
            fillcolor='lightblue',
            
            customdata=df_subset[['ID', '各個体の総育児時間']],
            hovertemplate=(
                "<b>個体ID: %{customdata[0]}</b><br>" +
                "育児時間: %{customdata[1]}<br>" +
                "<extra></extra>"
            )
        ))
    fig_caring.update_layout(
        title={
            'text': '各個体の総育児時間',
            'x': 0.5, 
            'xanchor': 'center'
        },
        yaxis_title='各個体の総育児時間',
        showlegend=False,
        #boxmode='group'
    )
    fig_caring.write_html(f"{path_out}TotalRearingTime.html")
    figs["TotalRearingTime"] = fig_caring


    time_trophallaxis_0 = [sum([e.duration for e in bee.event_trophallaxis if e.duration > th]) for bee in bees_1.values()]
    time_trophallaxis_1 = [sum([e.duration for e in bee.event_trophallaxis if e.duration > th]) for bee in bees_2.values()]
    data_trophallaxis = {
        '頭ー頭・頭ー腹 相互作用の総発生回数': time_trophallaxis_0 + time_trophallaxis_1,
        'Category': ["5SP"] * len(time_trophallaxis_0) + ["PBS"] * len(time_trophallaxis_1),
        'ID': list(bees_1.keys()) + list(bees_2.keys())
    }
    df_trophallaxis = pd.DataFrame(data_trophallaxis)
    export_to_csv(df_trophallaxis, path_out, "Individual_TotalTrophallaxisTime.csv")
    fig_trophallaxis = go.Figure()
    for category in df_trophallaxis['Category'].unique():
        df_subset = df_trophallaxis[df_trophallaxis['Category'] == category]
        
        fig_trophallaxis.add_trace(go.Box(
            x=df_subset['Category'], 
            y=df_subset['頭ー頭・頭ー腹 相互作用の総発生回数'],
            name=category,
            boxpoints='all',  
            pointpos=0,       
            jitter=0.0001,       
            marker=dict(
                color='darkgreen', 
                size=7,
                opacity=0.6,
                line=dict(width=0)
            ),
            line=dict(color='darkblue', width=1), 
            fillcolor='lightblue',
            
            customdata=df_subset[['ID', '頭ー頭・頭ー腹 相互作用の総発生回数']],
            hovertemplate=(
                "<b>個体ID: %{customdata[0]}</b><br>" +
                "育児時間: %{customdata[1]}<br>" +
                "<extra></extra>"
            )
        ))

    fig_trophallaxis.update_layout(
        title={
            'text': '頭ー頭・頭ー腹 相互作用の総発生回数',
            'x': 0.5, 
            'xanchor': 'center'
        },
        yaxis_title='頭ー頭・頭ー腹 相互作用の総発生回数',
        showlegend=False,
    )
    fig_trophallaxis.write_html(f"{path_out}TotalTrophallaxisTime.html")
    figs["TotalTrophallaxisTime"] = fig_trophallaxis
    
    node_bees_flora = []
    node_hives_flora = []
    edges_flora = []
    node_bees_noflora = []
    node_hives_noflora = []
    edges_noflora = []
    for bee in bees_1.values():
        data_caring = collections.Counter([e.id_hive for e in bee.event_caring if e.duration > th]).most_common()
        if len(data_caring) != 0:
            node_bees_flora.append(bee.id)
            for d in data_caring:
                node_hives_flora.append(str(d[0]))
                edges_flora.append((bee.id, str(d[0]), d[1]))

    for bee in bees_2.values():
        data_caring = collections.Counter([e.id_hive for e in bee.event_caring if e.duration > th]).most_common()
        if len(data_caring) != 0:
            node_bees_noflora.append(bee.id)
            for d in data_caring:
                node_hives_noflora.append(str(d[0]))
                edges_noflora.append((bee.id, str(d[0]), d[1]))
            
    if len(edges_flora) != 0 and len(edges_noflora) != 0:
        all_weights = [e[2] for e in edges_flora] + [e[2] for e in edges_noflora]
        global_max_weight = max(all_weights) if all_weights else 1

        fig_caring_network_flora = gen_bipartite_network(node_bees_flora, "ハチID", node_hives_flora, "幼虫ID", edges_flora, global_max_weight, title="ハチと幼虫の相互作用の評価")
        fig_caring_network_noflora = gen_bipartite_network(node_bees_noflora, "ハチID", node_hives_noflora, "幼虫ID", edges_noflora, global_max_weight, title="ハチと幼虫の相互作用の評価")
    else:
        fig_caring_network_flora = go.Figure()
        fig_caring_network_noflora = go.Figure()       

    if edges_flora:
        df_edges_flora = pd.DataFrame(edges_flora, columns=["BeeID", "HiveID", "Count"])
        export_to_csv(df_edges_flora, path_out, "Caring_Edges_Flora.csv")
    if edges_noflora:
        df_edges_noflora = pd.DataFrame(edges_noflora, columns=["BeeID", "HiveID", "Count"])
        export_to_csv(df_edges_noflora, path_out, "Caring_Edges_NoFlora.csv")

    combined_fig = make_subplots(
        rows=1, 
        cols=2, 
        subplot_titles=("A", "B")
    )
    for trace in fig_caring_network_flora.data:
        combined_fig.add_trace(trace, row=1, col=1)
    for trace in fig_caring_network_noflora.data:
        combined_fig.add_trace(trace, row=1, col=2)

    combined_fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    combined_fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    if fig_caring_network_flora.layout.annotations:
        for anno in fig_caring_network_flora.layout.annotations:
            anno_dict = anno.to_plotly_json()
            anno_dict.pop('xref', None) 
            anno_dict.pop('yref', None)
            combined_fig.add_annotation(
                **anno_dict,
                xref="x1",
                yref="y1",
                row=1, 
                col=1
            )
    if fig_caring_network_noflora.layout.annotations:
        for anno in fig_caring_network_noflora.layout.annotations:
            anno_dict = anno.to_plotly_json()
            anno_dict.pop('xref', None) 
            anno_dict.pop('yref', None)
            combined_fig.add_annotation(
                **anno_dict,
                xref="x2",
                yref="y2",
                row=1, 
                col=2
            )
        combined_fig.update_layout(
            title_text="ハチと幼虫の相互作用",
            title_x=0.5,
        )
    combined_fig.write_html(f"{path_out}Caring_Network.html")
    figs["Caring_Network"] = combined_fig
    
    edges_flora = []
    bee_ids_flora = set()
    hive_ids_flora = set()
    for bee in bees_1.values():
        data_caring = collections.Counter([e.id_hive for e in bee.event_caring if e.duration > th]).most_common()
        if len(data_caring) != 0:
            bee_ids_flora.add(bee.id)
            for d in data_caring:
                hive_id_str = str(d[0])
                hive_ids_flora.add(hive_id_str)
                edges_flora.append((bee.id, hive_id_str, d[1]))

    df_adj_flora = create_caring_adj_matrix(edges_flora, bee_ids_flora, hive_ids_flora)
    export_to_csv(df_adj_flora, path_out, "Caring_AdjMatrix_Flora.csv")

    edges_noflora = []
    bee_ids_noflora = set()
    hive_ids_noflora = set()
    for bee in bees_2.values():
        data_caring = collections.Counter([e.id_hive for e in bee.event_caring if e.duration > th]).most_common()
        if len(data_caring) != 0:
            bee_ids_noflora.add(bee.id)
            for d in data_caring:
                hive_id_str = str(d[0])
                hive_ids_noflora.add(hive_id_str)
                edges_noflora.append((bee.id, hive_id_str, d[1]))

    df_adj_noflora = create_caring_adj_matrix(edges_noflora, bee_ids_noflora, hive_ids_noflora)
    export_to_csv(df_adj_noflora, path_out, "Caring_AdjMatrix_NoFlora.csv")
    
    z_min_combined = 0
    z_max_combined = 0
    if df_adj_flora.size > 0:
        z_max_combined = max(z_max_combined, df_adj_flora.values.max())
    if df_adj_noflora.size > 0:
        z_max_combined = max(z_max_combined, df_adj_noflora.values.max())

    heatmap_trace_flora = go.Heatmap(
        z=df_adj_flora.values,
        x=df_adj_flora.columns,
        y=df_adj_flora.index,
        colorscale='Blues',
        zmin=z_min_combined,
        zmax=z_max_combined,
        texttemplate="%{z}",
        showscale=False 
    )

    heatmap_trace_noflora = go.Heatmap(
        z=df_adj_noflora.values,
        x=df_adj_noflora.columns,
        y=df_adj_noflora.index,
        colorscale='Blues',
        zmin=z_min_combined,
        zmax=z_max_combined,
        texttemplate="%{z}",
        showscale=True,
        colorbar=dict(
            title=dict(
                text="相互作用回数",
                side="right"
            ),
            x=1.05, 
            len=0.9
        )
    )

    combined_fig = make_subplots(
        rows=1, 
        cols=2, 
        subplot_titles=("A. Flora群: ハチと幼虫の相互作用", "B. No Flora群: ハチと幼虫の相互作用"),
        shared_yaxes=False,
        shared_xaxes=False
    )

    combined_fig.add_trace(heatmap_trace_flora, row=1, col=1)
    combined_fig.add_trace(heatmap_trace_noflora, row=1, col=2)
    
    combined_fig.update_xaxes(title_text="幼虫ID", row=1, col=1)
    combined_fig.update_yaxes(title_text="ハチID", row=1, col=1)
    combined_fig.update_xaxes(title_text="幼虫ID", row=1, col=2)
    combined_fig.update_yaxes(title_text="ハチID", row=1, col=2)
    
    combined_fig.update_layout(
        title_text="ハチと幼虫の相互作用",
        title_x=0.5,
        coloraxis=dict(colorscale='Blues', cmin=z_min_combined, cmax=z_max_combined)
    )
    figs["Caring_Heatmap"] = combined_fig
    
    edges_flora = []
    pair_added_flora = []
    for bee in bees_1.values():
        data_trophallaxis = collections.Counter([e.id_pair for e in bee.event_trophallaxis if e.duration > th]).most_common()
        if len(data_trophallaxis) != 0:
            for d in data_trophallaxis:
                if set((bee.id, d[0])) not in pair_added_flora:
                    pair_added_flora.append(set((bee.id, d[0])))
                    edges_flora.append((bee.id, d[0], d[1]))
    edges_noflora = []
    pair_added_noflora = []
    for bee in bees_2.values():
        data_trophallaxis = collections.Counter([e.id_pair for e in bee.event_trophallaxis if e.duration > th]).most_common()
        if len(data_trophallaxis) != 0:
            for d in data_trophallaxis:
                if set((bee.id, d[0])) not in pair_added_noflora:
                    pair_added_noflora.append(set((bee.id, d[0])))
                    edges_noflora.append((bee.id, d[0], d[1]))   
                    
    if edges_flora:
        df_edges_flora_t = pd.DataFrame(edges_flora, columns=["BeeID_A", "BeeID_B", "Count"])
        export_to_csv(df_edges_flora_t, path_out, "Trophallaxis_Edges_Flora.csv")
    if edges_noflora:
        df_edges_noflora_t = pd.DataFrame(edges_noflora, columns=["BeeID_A", "BeeID_B", "Count"])
        export_to_csv(df_edges_noflora_t, path_out, "Trophallaxis_Edges_NoFlora.csv")                    

    all_trophallaxis_weights = [e[2] for e in edges_flora] + [e[2] for e in edges_noflora]
    global_max_trophallaxis_weight = max(max(all_trophallaxis_weights) if all_trophallaxis_weights else 10000, 3000)
    #global_max_trophallaxis_weight = 10000
    fig_trophallaxis_network_flora = gen_network(edges_flora, global_max_trophallaxis_weight, "AAA")
    fig_trophallaxis_network_noflora = gen_network(edges_noflora, global_max_trophallaxis_weight, "AAA")
    combined_fig = make_subplots(
        rows=1, 
        cols=2, 
        subplot_titles=("A", "B")
    )
    for trace in fig_trophallaxis_network_flora.data:
        combined_fig.add_trace(trace, row=1, col=1)
    for trace in fig_trophallaxis_network_noflora.data:
        combined_fig.add_trace(trace, row=1, col=2)
    combined_fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    combined_fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    if fig_trophallaxis_network_flora.layout.annotations:
        for anno in fig_trophallaxis_network_flora.layout.annotations:
            anno_dict = anno.to_plotly_json()
            anno_dict.pop('xref', None) 
            anno_dict.pop('yref', None)
            combined_fig.add_annotation(
                **anno_dict,
                xref="x1",
                yref="y1",
                row=1, 
                col=1
            )
    if fig_trophallaxis_network_noflora.layout.annotations:
        for anno in fig_trophallaxis_network_noflora.layout.annotations:
            anno_dict = anno.to_plotly_json()
            anno_dict.pop('xref', None) 
            anno_dict.pop('yref', None)
            combined_fig.add_annotation(
                **anno_dict,
                xref="x2",
                yref="y2",
                row=1, 
                col=2
            )
    combined_fig.update_layout(
        title_text="個体間相互作用",
        title_x=0.5,
    )
    combined_fig.write_html(f"{path_out}Trophallaxis_Network.html")
    figs["Trophallaxis_Network"] = combined_fig

    edges_flora = []
    pair_added = []
    for bee in bees_1.values():
        data_trophallaxis = collections.Counter([e.id_pair for e in bee.event_trophallaxis if e.duration > th]).most_common()
        if len(data_trophallaxis) != 0:
            for d in data_trophallaxis:
                if set((bee.id, d[0])) not in pair_added:
                    pair_added.append(set((bee.id, d[0])))
                    edges_flora.append((bee.id, d[0], d[1]))

    bee_ids_flora = set(bees_1.keys()) 
    df_adj_flora = create_adj_matrix(edges_flora, bee_ids_flora)
    export_to_csv(df_adj_flora, path_out, "Trophallaxis_AdjMatrix_Flora.csv")
    
    edges_noflora = []
    pair_added = []
    for bee in bees_2.values():
        data_trophallaxis = collections.Counter([e.id_pair for e in bee.event_trophallaxis if e.duration > th]).most_common()
        if len(data_trophallaxis) != 0:
            for d in data_trophallaxis:
                if set((bee.id, d[0])) not in pair_added:
                    pair_added.append(set((bee.id, d[0])))
                    edges_noflora.append((bee.id, d[0], d[1]))

    bee_ids_noflora = set(bees_2.keys())
    df_adj_noflora = create_adj_matrix(edges_noflora, bee_ids_noflora)
    export_to_csv(df_adj_noflora, path_out, "Trophallaxis_AdjMatrix_NoFlora.csv")

    z_min_combined = 0
    z_max_combined = max(df_adj_flora.values.max(), df_adj_noflora.values.max())

    heatmap_trace_flora = go.Heatmap(
        z=df_adj_flora.values,
        x=df_adj_flora.columns,
        y=df_adj_flora.index,
        colorscale='Blues',
        zmin=z_min_combined,
        zmax=z_max_combined,
        texttemplate="%{z}",
        showscale=False 
    )

    heatmap_trace_noflora = go.Heatmap(
        z=df_adj_noflora.values,
        x=df_adj_noflora.columns,
        y=df_adj_noflora.index,
        colorscale='Blues',
        zmin=z_min_combined,
        zmax=z_max_combined,
        texttemplate="%{z}",
        showscale=True,
        colorbar=dict(
            title=dict(
                text="相互作用回数",
                side="right"
            ),
            x=1.05, 
            len=0.9
        )
    )

    combined_fig = make_subplots(
        rows=1, 
        cols=2, 
        subplot_titles=("A. Flora群", "B. No Flora群"),
        shared_yaxes=False,
        shared_xaxes=False
    )

    combined_fig.add_trace(heatmap_trace_flora, row=1, col=1)
    combined_fig.add_trace(heatmap_trace_noflora, row=1, col=2)

    combined_fig.update_xaxes(title_text="個体ID (受け手)", row=1, col=1)
    combined_fig.update_yaxes(title_text="個体ID (渡し手)", row=1, col=1)
    combined_fig.update_xaxes(title_text="個体ID (受け手)", row=1, col=2)
    combined_fig.update_yaxes(title_text="個体ID (渡し手)", row=1, col=2)


    combined_fig.update_layout(
        title_text="個体間相互作用",
        title_x=0.5,
        coloraxis=dict(colorscale='Viridis', cmin=z_min_combined, cmax=z_max_combined)
    )

    path_out = ""
    combined_fig.write_html(f"{path_out}Trophallaxis_Heatmap.html")
    figs["Trophallaxis_Heatmap"] = combined_fig    
    
    with open(f"{path_out}figs.pkl", mode='wb') as f:
        pickle.dump(figs, f)
    return figs

if __name__ == "__main__":
    with open("/kpsort/output/flora2/data_graph.pkl", "rb") as f:
        data_flora = pickle.load(f) 
    with open("/kpsort/output/noflora2/data_graph.pkl", "rb") as f:
        data_noflora = pickle.load(f)
        
    bee_flora = data_flora["bees"]
    bee_noflora = data_noflora["bees"]

    #Bee = data["Bee"]
    #gen_graphs("/kpsort/test/", bee_flora, bee_noflora, th=18)
    edges = []
    pair_added = []
    for bee in bee_noflora.values():
        data_trophallaxis = collections.Counter([e.id_pair for e in bee.event_trophallaxis if e.duration > 5]).most_common()
        if len(data_trophallaxis) != 0:
            for d in data_trophallaxis:
                if set((bee.id, d[0])) not in pair_added:
                    pair_added.append(set((bee.id, d[0])))
                    edges.append((bee.id, d[0], d[1]))
    fig = gen_network(edges, "AAA")
    fig.write_html(f"test.html")