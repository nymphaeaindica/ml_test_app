"""Test app version for running from ubuntu/macos terminal"""
import xgboost as xgb
import numpy as np
import pandas as pd
import argparse
import sys
import calc
import plotly
import plotly.graph_objs as go
import datetime
from decimal import Decimal

columns = np.round(np.arange(0, 1, 0.01), decimals=3)

def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name')
    return parser


if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    xml_path = namespace.config_name
    if not xml_path:
        sys.exit('Specify the full path to the file "settings.xml"')
    global_params, other = calc.parsconf(xml_path)
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n' + '%s' % now + '  Testing stage ...    ')
    print('Configuration file was parsed')
    aux = pd.read_csv(global_params['working_dir'] + global_params['test_aux_file_name'], header=None, sep=',',
                      dtype=str)
    del aux[2]
    configcsv = pd.read_csv(global_params['working_dir'] + global_params['csv_file_name'], header=0,
                            usecols=[1, 7, 8, 9, 17])
    cut_ranges = configcsv.query('column10 != 0')[['column2', 'column8', 'column9']]
    features_index = configcsv.query('column18 != 0').index
    threshold = np.arange(0, 1, 0.1)
    model = xgb.Booster()
    model.load_model(global_params['working_dir'] + global_params['model_file_name'])
    simulant_x = []
    benign_y = []
    fig1 = go.Figure()
    fig2 = go.Figure()
    output_target = []
    input = []
    simple = []
    print('Predicting scores...')
    for ind, row in aux.iterrows():
            features, labels, target_id, n_input, n_simple = calc.dat_keeper(row[1], row[0], features_index, cut_ranges)
            dtest = xgb.DMatrix(features)
            scores = model.predict(dtest)
            scores_dict = calc.prediction2dict(scores, target_id)
            target = calc.predict_target(columns, scores_dict)
            output_target.append(target)
            input.append(n_input)
            simple.append(n_simple)
    output_df = pd.DataFrame(output_target, columns=columns.astype(dtype=str))
    output_df['input_target'] = input
    output_df['label'] = aux[0]
    del output_target
    sim = []
    benign = []
    for i in columns.astype(dtype=str):
        all_input_1 = output_df[output_df['label'] != '0']['input_target'].sum(min_count=0)
        tmp1 = output_df[output_df['label'] != '0'][i].sum()
        x1 = (tmp1 / all_input_1)
        sim.append(x1)
        all_input_0 = output_df[output_df['label'] == '0']['input_target'].sum(min_count=0)
        tmp0 = output_df[output_df['label'] =='0'][i].sum()
        x0 = (tmp0 / all_input_0)
        benign.append(x0)
    m_1 = []
    n_0 = []
    m_1_path = calc.split_path(aux[aux[0] != '0'][1])
    n_0_path = calc.split_path(aux[aux[0] == '0'][1])
    for index, row in output_df[output_df['label'] != '0'].iterrows():
        stat = []
        for i in columns.astype(dtype=str):
            x = output_df[i][index] / output_df['input_target'][index]
            stat.append(x)
        m_1.append(stat)
    for index, row in output_df[output_df['label'] == '0'].iterrows():
        stat = []
        for i in columns.astype(dtype=str):
            x = output_df[i][index] / output_df['input_target'][index]
            stat.append(x)
        n_0.append(stat)
    print('Plotting graphs...')
    for i, j in zip(n_0, n_0_path):
        # fig1.add_trace(go.Scatter(x=n_0[i], y=sim, name=n_0_path[j]))
        fig1.add_trace(go.Scatter(x=i, y=sim, name=j))
    for i, j in zip(m_1, m_1_path):
        fig2.add_trace(go.Scatter(x=benign, y=i, name=j))
    fig1.add_trace(go.Scatter(x=benign, y=sim, name='all files', line=dict(color='firebrick', width=4)))
    fig2.add_trace(go.Scatter(x=benign, y=sim, name='all files', line=dict(color='firebrick', width=4)))
    fig1.update_layout(
        title=go.layout.Title(
            text="Class 0 details",
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
    fig2.update_layout(
        title=go.layout.Title(
            text="Class 1 details",
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
    fig1.show()
    fig2.show()
    if other['fixed_threshold'] is not None:
        print('Importing to csv...')
        import_data = pd.DataFrame()
        import_data['label'] = aux[0]
        import_data['filename'] = calc.split_path(aux[1])
        import_data['input target'] = output_df['input_target']
        import_data['simple target'] = simple
        fixed_thr = ''.join(str(i) for i in other['fixed_threshold'])
        import_data['fixed threshold'] = output_df[fixed_thr]
        # n = import_data['fixed threshold'].sum()
        # N = import_data['input target'].sum()
        n = import_data[import_data['label'] == '0']['fixed threshold'].sum(min_count=0)
        m = import_data[import_data['label'] != '0']['fixed threshold'].sum(min_count=0)
        n_sim = import_data[import_data['label'] == '0']['simple target'].sum(min_count=0)
        m_sim = import_data[import_data['label'] != '0']['simple target'].sum(min_count=0)
        N = import_data[import_data['label'] == '0']['input target'].sum(min_count=0)
        M = import_data[import_data['label'] != '0']['input target'].sum(min_count=0)
        n_div = n / N
        m_div = m / M
        import_data['n / N'] = import_data['fixed threshold'].div(import_data['input target'], axis=0)
        import_data = import_data.append({'label': '0', 'filename':'total','input target':N, 'simple target':n_sim,'fixed threshold':n,'n / N':n_div },ignore_index=True)
        import_data = import_data.append({'label': '1', 'filename': 'total', 'input target': M, 'simple target': m_sim, 'fixed threshold': m, 'n / N': m_div}, ignore_index=True)
        import_data.to_csv(global_params['working_dir'] + global_params['model_file_name'].split('.bin')[0] + '.csv', index=False, sep=',')
    print('Saving graphs to html...')
    fig1_save = plotly.offline.plot(fig1, filename=global_params['working_dir'] + global_params['model_file_name'].split('.')[0] + '_' + 'class_0_details.html', auto_open=False)
    fig2_save = plotly.offline.plot(fig2, filename=global_params['working_dir'] + global_params['model_file_name'].split('.')[0] + '_' + 'class_1_details.html', auto_open=False)
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n' + '%s' % now + '  Testing stage is over.')

