import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import base64
import new_calculate as calc
from sklearn.metrics import roc_curve
import textwrap
 
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
 
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True
 
"""Design of web page"""
 
app.layout = html.Div(
    style={"height": "100%"},  # add more details
    children=[
        html.Div(id='output-data-upload-2', style={'display': 'none'}),
        html.Div(id='output-data-upload-3', style={'display': 'none'}),
        html.Div(id='scores', style={'display': 'none'}),
 
        html.Div(
            [
                html.H1(
                    'Xgboost Test Stage Viewer',
                    id='title',
                    className="ten columns",
                    style={"margin-left": "3%"}  # change if necessary
                ),
          
            ],
            className='banner row'
        ),
        html.Div(
            className='buttons row',
            children=[html.Div(
                className="twelve columns",
                children=[
                    html.Div([
                        dcc.Upload(id='aux',
                                   children=html.Div([
                                       'Drag and Drop or ',
                                       html.A('select .aux file')
                                   ]),
                                   style={
                                       'width': '80%',
                                       'height': '60px',
                                       'lineHeight': '60px',
                                       'borderWidth': '1px',
                                       'borderStyle': 'solid',
                                       'borderRadius': '5px',
                                       'textAlign': 'center',
                                       'margin': '10px',
                                       'padding': '10px'
                                   }, ),
                        html.Div(id='aux_confirm')
                    ], className="three columns"),
                    html.Div([
                        dcc.Upload(id='csv',
                                   children=html.Div([
                                       'Drag and Drop or ',
                                       html.A('select .csv file')
                                   ]),
                                   style={
                                       'width': '80%',
                                       'height': '60px',
                                       'lineHeight': '60px',
                                       'borderWidth': '1px',
                                       'borderStyle': 'solid',
                                       'borderRadius': '5px',
                                       'textAlign': 'center',
                                       'margin': '10px',
                                       'padding': '10px'
                                   }, ),
                        html.Div(id='csv_confirm')
                    ], className="three columns"),
                    html.Div([
                        dcc.Upload(id='bin',
                                   children=html.Div([
                                       'Drag and Drop or ',
                                       html.A('select .bin file')
                                   ]),
                                   style={
                                       'width': '80%',
                                       'height': '60px',
                                       'lineHeight': '60px',
                                       'borderWidth': '1px',
                                       'borderStyle': 'solid',
                                       'borderRadius': '5px',
                                       'textAlign': 'center',
                                       'margin': '10px',
                                       'padding': '10px',
                                       'BackgroundColor': 'white'
                                   }, ),
                        html.Div(id='bin_confirm')
                    ], className="three columns"),
                    html.Div([
                        html.Button(
                            id='predict',
                            n_clicks=0,
                            children=["Predict"],
                            style={
                                'width': '80%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'solid',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px',
                                'padding': '10px',
                                'BackgroundColor': 'white'
                            }
                        ),
                    ], className='three columns')
                ]
            )
 
            ]
        ),
        html.Div(
            className='stats row',
            children=[
                html.Div(
                    className='twelve columns',
                    children=[
                        html.Div(
                            className='nine columns',
                            children=[
                                html.Div(id='Mygraph_2', children=[
                                    dcc.Graph(
                                        id='Mygraph',
                                        figure={'data': []},
                                        style={
                                            "height": "600px"
                                        }
                                    ),
                                ]
                                         )
                            ]
                        ),
                        html.Div(
                            className='three columns',
                            id = 'stat'
                        )
                    ]
                )
            ]
        ),
        html.Div(
            className='stats row 2',
            children=[
                html.Div(
                    className='twelve columns',
                    children=[
                        html.Div(
                            className='nine columns',
                            children=[
                                html.Div(id='Mygraph2.0', children=[
                                    dcc.Graph(
                                        id='Mygraph2',
                                        figure={'data': []},
                                        style={
                                            "height": "600px"
                                        }
                                    ),
                                ]
                                         )
                            ]
                        ),
                        html.Div(
                            className='three columns',
                            id = 'stat2'
                        )
                    ]
                )
            ]
        )
 
    ]
)
 
# store data from .csv in pandas Dataframe
 
def get_csv(filename):
    df = pd.DataFrame()
    if filename is not None:
        df = pd.read_csv(filename, header=0, usecols=[1, 7, 8, 9, 17])
    else:
        raise PreventUpdate
    return df
 
# store data from .aux in pandas Dataframe
 
def get_aux(filename):
    df = pd.DataFrame()
    if filename is not None:
        df = pd.read_csv(filename, header=None, usecols=[0, 1])
    else:
        raise PreventUpdate
    return df
 
# upload csv file
 
@app.callback([Output('output-data-upload-2', 'children'), Output('csv_confirm', 'children')],
              [Input('csv', 'filename')])
def update_csv(filename):
    if filename is None:
        raise PreventUpdate
    else:
        df = get_csv(filename)
        df = df.to_json()
    return df, 'File uploaded successfully'
 
# upload aux file
 
@app.callback([Output('output-data-upload-3', 'children'), Output('aux_confirm', 'children')],
              [Input('aux', 'filename')])
def update_aux(filename):
    if filename is None:
        raise PreventUpdate
    else:
        data = get_aux(filename)
        data = data.to_json()
    return data, 'File uploaded successfully'
 
# upload bin file
 
@app.callback(Output('bin_confirm','children'),
              [Input('bin','filename')])
def update_bin(filename):
    if filename is None:
        raise PreventUpdate
    else:
        return 'File uploaded successfully'
 
 
"""
TEST STAGE:
Getting data from .aux, .csv and .bin files, 
making predictions, showing statistics (graphs)
"""
 
@app.callback([Output('Mygraph','figure'), Output('Mygraph2', 'figure'), Output('scores', 'children')],
              [Input('predict', 'n_clicks')],
              [State('output-data-upload-2', 'children'),
               State('output-data-upload-3', 'children'),
               State('bin', 'filename')]
              )
def perform_xgb(n_clicks, csv, aux, filename):
    if csv is None:
        if aux is None:
            if filename is None:
                raise PreventUpdate
    else:
        if n_clicks > 0:
            df = pd.read_json(csv)
            cut_ranges = df.query('column10 != 0')[['column2', 'column8', 'column9']]
            features_index = df.query('column18 != 0').index
            aux = pd.read_json(aux)
            path_list = aux[1]
            label_list = aux[0].to_numpy()
            index0 = [i for i, x in enumerate(label_list) if x == 0]
            index1 = [i for i, x in enumerate(label_list) if x == 1]
            path_0 = list(path_list[index0])
            path_1 = list(path_list[index1])
            label_0 = list(label_list[index0])
            label_1 = list(label_list[index1])
            features = []
            labels = []
            plot_data = {}
            model = xgb.Booster()
            model.load_model(filename)
            # ROC-AUC (Benign to simulant)
            fig1 = go.Figure()
            for ind, path in enumerate(path_1):
                features, labels, target_id, n_input, n_simple = calc.read_data(path_0, label_0, features_index, cut_ranges)
                s_features, s_labels, s_target_id, s_n_input, s_n_simple = calc.dat_keeper(path, label_1[ind], features_index, cut_ranges)
                n_features = np.concatenate([features, s_features], axis=0)
                n_labels = np.concatenate([labels, s_labels], axis=0)
                dtest = xgb.DMatrix(n_features)
                scores = model.predict(dtest)
                fpr, tpr, thr = roc_curve(n_labels, scores)
                trace_name = path.split('/')[-1]
                legend_str = '<br>'.join(textwrap.wrap(trace_name, width=50))
                # roc-auc for each benign file
                fig1.add_trace(go.Scatter(x=fpr, y=tpr, name=legend_str, hovertext=np.ndarray.round(thr, decimals=4)))  # trace 2
                fig1.update_layout(
                    title=go.layout.Title(
                        text="ROC-AUC (benign to simulants)", 
                        # xref="paper",
                        x=0,
                        xanchor="auto",
                        font=dict(
                            family="Courier New, monospace",
                            size=18,
                            color="#000000"
                        )),
                    xaxis=go.layout.XAxis(
                        title=go.layout.xaxis.Title(
                            text="FPR",
                            font=dict(
                                family="Courier New, monospace",
                                size=18,
                                color="#000000"
                            )
                        ),
                        hoverformat=".3f"
                    ),
                    yaxis=go.layout.YAxis(
                        title=go.layout.yaxis.Title(
                            text="TPR",
                            font=dict(
                                family="Courier New, monospace",
                                size=18,
                                color="#000000"
                            )
                        ),
                        hoverformat=".3f"
                    )
                )
                plot_data = {
                    'scores': json.dumps(np.ndarray.tolist(scores)),
                    # 'scores0': json.dumps(list(map(float, scores0))),  # can't work with list?????
                    # 'scores1': json.dumps(list(map(float, scores1))),
                    # 'y0': json.dumps(np.ndarray.tolist(y0)),
                    # 'y1': json.dumps(np.ndarray.tolist(y1)),
                    # 'input_targets': json.dumps(np.ndarray.tolist(input_targets)),
                    # 'simple_targets': json.dumps(np.ndarray.tolist(simple_targets))
                }
            features_1, labels_1, target_id_1, n_input_1, n_simple_1 = calc.read_data(path_1, label_1, features_index, cut_ranges)
            all_features = np.concatenate([features, features_1], axis=0)
            all_labels = np.concatenate([labels, labels_1], axis=0)
            all_test = xgb.DMatrix(all_features)
            all_scores = model.predict(all_test)
            fpr, tpr, thr = roc_curve(all_labels, all_scores)
            # roc-auc for all benign files
            fig1.add_trace(go.Scatter(x=fpr, y=tpr, name='all', marker_color='#0000FF', text=thr))
            # ROC-AUC (Simulant to benign)
            fig2 = go.Figure()
            for ind, path in enumerate(path_0):
                features, labels, target_id, n_input, n_simple = calc.read_data(path_1, label_1, features_index, cut_ranges)
                s_features, s_labels, s_target_id, s_n_input, s_n_simple = calc.dat_keeper(path, label_0[ind], features_index, cut_ranges)
                n_features = np.concatenate([features, s_features], axis=0)
                n_labels = np.concatenate([labels, s_labels], axis=0)
                dtest = xgb.DMatrix(n_features)
                scores = model.predict(dtest)
                fpr2, tpr2, thr2 = roc_curve(n_labels, scores)
                trace_name = path.split('/')[-1]
                legend_str = '<br>'.join(textwrap.wrap(trace_name, width=50))
                # roc-auc for each simulant to benign
                fig2.add_trace(go.Scatter(x=fpr2, y=tpr2, name=legend_str, text=thr2))  # trace 2
                # fig.add_trace(go.Scatter(x=thr, y=y1, name=path, marker_color='#0000FF'))  # trace 1
                fig2.update_layout(
                    title=go.layout.Title(
                        text="ROC-AUC (simulant to benigns)",  
                        # xref="paper",
                        x=0,
                        xanchor="auto",
                        font=dict(
                            family="Courier New, monospace",
                            size=18,
                            color="#000000"
                        )),
                    xaxis=go.layout.XAxis(
                        title=go.layout.xaxis.Title(
                            text="FPR",
                            font=dict(
                                family="Courier New, monospace",
                                size=18,
                                color="#000000"
                            )
                        ),
                        hoverformat=".3f"
                    ),
                    yaxis=go.layout.YAxis(
                        title=go.layout.yaxis.Title(
                            text="TPR",
                            font=dict(
                                family="Courier New, monospace",
                                size=18,
                                color="#000000"
                            )
                        ),
                        hoverformat=".3f"
                    )
                )
            features_0, labels_0, target_id_0, n_input_0, n_simple_0 = calc.read_data(path_0, label_0, features_index, cut_ranges)
            all_features = np.concatenate([features, features_0], axis=0)
            all_labels = np.concatenate([labels, labels_0], axis=0)
            all_test = xgb.DMatrix(all_features)
            all_scores = model.predict(all_test)
            # roc-auc for all simulant files
            fpr, tpr, thr = roc_curve(all_labels, all_scores)
            fig2.add_trace(go.Scatter(x=fpr, y=tpr, text=thr, name='all', marker_color='#0000FF'))
            return fig1, fig2, json.dumps(plot_data)
        else:
            raise PreventUpdate
 
# showing statistics (number of targets)
 
# @app.callback(
#     Output('stat', 'children'),
#     [Input('Mygraph', 'clickData'),
#      Input('scores', 'children')])
# def display_click_data(clickData, children):  # scores/particular_score???
#     x = 0
#     out_t = 0
#     if clickData is None:
#         raise PreventUpdate
#     storage = json.loads(children)
#     x = clickData['points'][0]['x']
#     scores = json.loads(storage['scores'])
#     scores0 = json.loads(storage['scores0'])
#     scores1 = json.loads(storage['scores1'])
#     in_t = json.loads(storage['input_targets'])
#     sim_t = json.loads(storage['simple_targets'])
#     curve_number = clickData['points'][0]['curveNumber']
#     if curve_number == 0:
#         out_t = calc.output_target(scores, scores0, x)
#     elif curve_number == 1:
#         out_t = calc.output_target(scores, scores1, x)
#     return html.Div(id='stat', children=[
#             html.P(
#                 "Statistics:",
#                 # style={
#                 #     "font-weight": "bold",
#                 #     "margin-top": "15px",
#                 #     "margin-bottom": "0px",
#                 # },
#             ),
#             html.Div(f"Curve number: {curve_number}"),
#             html.Div(f"Input targets: {len(in_t)}"),
#             html.Div(f"Simple targets: {len(sim_t)}"),
#             html.Div(f"Threshold: {x}"),
#             html.Div(f"Output targets: {out_t}")
#         ])
 
 
if __name__ == '__main__':
    app.run_server(debug=True)
