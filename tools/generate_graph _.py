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

def gen_network(edges, title=""):
    G = nx.Graph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=w * 10)
    if not G.edges():
        return go.Figure()
    
    num_nodes = len(G.nodes())
    k_value = 1.0 / (num_nodes ** 0.5) * 1.5
    
    pos = nx.spring_layout(
        G, 
        k=k_value * 1000,          # ノード間の理想的な距離を調整 (デフォルトは 1/sqrt(N))
                           # 値を大きくするとノードが広がり、小さくすると密集する
        iterations=100,    # イテレーション回数を増やすと、レイアウトがより安定する
        seed=42,           # レイアウトの再現性のため乱数シードを設定
        center=[0.5, 0.5], # グラフ全体がPlotlyの表示範囲の中央にくるように誘導
        scale=0.8          # レイアウトの全体的なスケールを調整（描画範囲に収める）
    )

    # 4. 重みの正規化と線の太さの計算
    edge_weights_dict = nx.get_edge_attributes(G, 'weight')
    all_weights = list(edge_weights_dict.values())
    max_weight = max(all_weights) if all_weights else 1
    
    # 線の太さリスト（Plotlyではトレースごとに指定するため、少しロジックが変わる）
    
    # 5. エッジトレースの作成
    edge_traces = []
    
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        
        # 線の太さの正規化 (最大 5 にスケーリング)
        scaled_width = (weight / max_weight) * 5
        
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        trace_edge = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=scaled_width, color='black'),
            hoverinfo='text',
            mode='lines',
            opacity=0.7,
            # ホバーテキスト: weightを10で割って元の回数に戻す
            text=[f"相互作用: {u} - {v}<br>回数: {weight/10:.1f}"],
            showlegend=False
        )
        edge_traces.append(trace_edge)

    # 6. ノードトレースの作成
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    # ノードサイズはここでは一律（必要に応じてノードの接続数などで変更可能）
    node_sizes = [15] * len(G.nodes())
    node_text = [f"ID: {node}" for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=list(G.nodes()), # ノードIDを直接ラベルとして表示
        textposition='middle center',
        marker=dict(
            showscale=False,
            size=40,
            color='white', # ノードの色を統一
            line_width=2,
            line_color='black'
        ),
        textfont=dict(size=12, color='black'),
        hovertext=node_text,
        showlegend=False
    )

    # 7. Figureオブジェクトの構築
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

def gen_bipartite_network(nodes_left, legend_left, nodes_right, legend_right, edges, title=""):
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
    max_weight = max(nx.get_edge_attributes(B, 'weight').values()) if nx.get_edge_attributes(B, 'weight') else 1

    for edge in B.edges(data=True):
        x0, y0 = pos[edge[0]] # ノードUの座標
        x1, y1 = pos[edge[1]] # ノードVの座標
        
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        weight = edge[2].get('weight', 1)
        hover_text_edges.append(f"相互作用の重み: {weight/10:.1f}") # 元の重みに戻して表示
    edge_traces = []
    current_edge_index = 0

    for edge in B.edges(data=True):
        weight = edge[2].get('weight', 1)
        
        line_width = weight / max_weight * 5 if max_weight > 0 else 1 
        
        start_index = current_edge_index * 3
        end_index = start_index + 2
        
        trace_edge = go.Scatter(
            x=edge_x[start_index:end_index],
            y=edge_y[start_index:end_index],
            line=dict(width=line_width, color='black'),
            hoverinfo='text',
            mode='lines',
            opacity=0.7,
            text=[f"{legend_left}: {edge[0]} - {legend_right}: {edge[1]}<br>重み: {weight/10:.1f}"], # 重みを元のスケールで表示
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
    
    df_0 = pd.DataFrame({'各個体の総移動距離': [i.distance_sum for i in bees_1.values()], 'Category': '腸内細菌あり'})
    df_1 = pd.DataFrame({'各個体の総移動距離': [i.distance_sum for i in bees_2.values()], 'Category': '腸内細菌なし'})
    sns.boxplot(x='Category', y='各個体の総移動距離', data=df_0, color='lightblue', ax=plt.gca())
    sns.stripplot(x='Category',y='各個体の総移動距離', data=df_0, color='darkgreen', s=7, alpha=0.6, jitter=0.0001, ax=plt.gca())
    sns.boxplot(x='Category', y='各個体の総移動距離', data=df_1, color='lightblue', ax=plt.gca())
    sns.stripplot(x='Category',y='各個体の総移動距離', data=df_1, color='darkgreen', s=7, alpha=0.6, jitter=0.0001, ax=plt.gca())
    plt.savefig(f"{path_out}TotalDistanceTraveled.png")

    plt.cla()


    time_caring_0 = [sum([e.duration for e in bee.event_caring if e.duration > 0]) for bee in bees_1.values()]
    time_caring_1 = [sum([e.duration for e in bee.event_caring if e.duration > 0]) for bee in bees_1.values()]
    df_0 = pd.DataFrame({'各個体の総育児時間': time_caring_0, 'Category': '腸内細菌あり'})
    df_1 = pd.DataFrame({'各個体の総育児時間': time_caring_1, 'Category': '腸内細菌なし'})
    sns.boxplot(x='Category', y='各個体の総育児時間', data=df_0, color='lightblue', ax=plt.gca())
    sns.stripplot(x='Category',y='各個体の総育児時間', data=df_0, color='darkgreen', s=7, alpha=0.6, jitter=0.0001, ax=plt.gca())
    sns.boxplot(x='Category', y='各個体の総育児時間', data=df_1, color='lightblue', ax=plt.gca())
    sns.stripplot(x='Category',y='各個体の総育児時間', data=df_1, color='darkgreen', s=7, alpha=0.6, jitter=0.0001, ax=plt.gca())
    plt.savefig(f"{path_out}TotalRearingTime.png")

    distance_data_0 = [i.distance_sum for i in bees_1.values()]
    distance_data_1 = [i.distance_sum for i in bees_2.values()]
    data_distance = {
        '各個体の総移動距離': distance_data_0 + distance_data_1,
        'Category': ['腸内細菌あり'] * len(distance_data_0) + ['腸内細菌なし'] * len(distance_data_1),
        "ID": list(bees_1.keys()) + list(bees_2.keys())
    }
    df_distance = pd.DataFrame(data_distance)
    fig_distance = go.Figure()
    for category in df_distance['Category'].unique():
        df_subset = df_distance[df_distance['Category'] == category]
        fig_distance.add_trace(go.Box(
            y=df_subset['各個体の総移動距離'],
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
        'Category': ['腸内細菌あり'] * len(time_caring_0) + ['腸内細菌なし'] * len(time_caring_1),
        'ID': list(bees_1.keys()) + list(bees_2.keys())
    }
    df_caring = pd.DataFrame(data_caring)

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
        'Category': ['腸内細菌あり'] * len(time_trophallaxis_0) + ['腸内細菌なし'] * len(time_trophallaxis_1),
        'ID': list(bees_1.keys()) + list(bees_2.keys())
    }
    df_trophallaxis = pd.DataFrame(data_trophallaxis)
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
    

    node_bees = []
    node_hives = []
    edges = []
    for bee in bees_1.values():
        data_caring = collections.Counter([e.id_hive for e in bee.event_caring if e.duration > th]).most_common()
        if len(data_caring) != 0:
            node_bees.append(bee.id)
            for d in data_caring:
                node_hives.append(str(d[0]))
                edges.append((bee.id, str(d[0]), d[1]))
    if len(edges) != 0:
        fig_caring_network_flora = gen_bipartite_network(node_bees, "ハチID", node_hives, "幼虫ID", edges, title="ハチと幼虫の相互作用の評価")
    else:
        fig_caring_network_flora = go.Figure()
    node_bees.clear()
    node_hives.clear()
    edges.clear()
    for bee in bees_2.values():
        data_caring = collections.Counter([e.id_hive for e in bee.event_caring if e.duration > th]).most_common()
        if len(data_caring) != 0:
            node_bees.append(bee.id)
            for d in data_caring:
                node_hives.append(str(d[0]))
                edges.append((bee.id, str(d[0]), d[1]))
    if len(edges) != 0:        
        fig_caring_network_noflora = gen_bipartite_network(node_bees, "ハチID", node_hives, "幼虫ID", edges, title="ハチと幼虫の相互作用の評価")
    else:
        fig_caring_network_noflora = go.Figure()

    combined_fig = make_subplots(
        rows=1, 
        cols=2, 
        subplot_titles=("腸内細菌あり", "腸内細菌なし")
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
    

    edges = []
    pair_added = []
    for bee in bees_1.values():
        data_trophallaxis = collections.Counter([e.id_pair for e in bee.event_trophallaxis if e.duration > th]).most_common()
        if len(data_trophallaxis) != 0:
            for d in data_trophallaxis:
                if set((bee.id, d[0])) not in pair_added:
                    pair_added.append(set((bee.id, d[0])))
                    edges.append((bee.id, d[0], d[1]))
    fig_trophallaxis_network_flora = gen_network(edges, "AAA")
    edges = []
    pair_added = []
    for bee in bees_2.values():
        data_trophallaxis = collections.Counter([e.id_pair for e in bee.event_trophallaxis if e.duration > th]).most_common()
        if len(data_trophallaxis) != 0:
            for d in data_trophallaxis:
                if set((bee.id, d[0])) not in pair_added:
                    pair_added.append(set((bee.id, d[0])))
                    edges.append((bee.id, d[0], d[1]))
    fig_trophallaxis_network_noflora = gen_network(edges, "AAA")
    combined_fig = make_subplots(
        rows=1, 
        cols=2, 
        subplot_titles=("腸内細菌あり", "腸内細菌なし")
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