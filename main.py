from gpchar import GPChar
import pdb
import time
import threading

from dash import Dash, dcc, html, Input, Output, callback, ALL
import plotly.graph_objects as go

import numpy as np
import matplotlib.pyplot as plt

def f(x, a, b):
    print("Running f", x)
    time.sleep(10)
    m = np.array([
        [np.cos(a)*np.cos(b), np.sin(a)*np.cos(b), np.cos(b)],
        [-np.sin(a)*np.cos(b), np.sin(a)*np.cos(b), -np.cos(b)]])
    print("Run done ", x)
    return np.dot(m, np.sin(2*x)) + 100

def plt_version():
    gpc = GPChar(f, [(0,1), (2,4), (-np.pi, np.pi)], ["x_1", "x_2", "x_3"], ["y_1", "y_2"], keyword_args={"a": 1, "b": 2})

    fig, ax = plt.subplots()
    xs, ys, stds = gpc.get_1d_prediction(2, np.array([0.5, 3, 0]))
    ax.plot(xs, ys[:,0], label="Ys")
    ax.fill_between(xs, ys[:,0]-stds[:,0], ys[:,0]+stds[:,0], alpha=0.5)
    plt.legend()
    plt.show()

def dash_version():
    bounds = [(0,1), (2,4), (-np.pi, np.pi)]
    inputs = ["x0", "x1", "x2"]
    outputs = ["y0", "y1"]
    gpc = GPChar(f, bounds, 3, 2, keyword_args={"a": 1, "b": 2})

    sliders = [
        dcc.Slider(
            b[0],
            b[1],
            step=(b[1]-b[0])/100,
            value=(b[1]+b[0])/2,
            marks={x: f"{x:.3g}" for x in np.arange(b[0], b[1], (b[1]-b[0])/10)},
            id={"type": 'slider', "index":i}
        )
        for i, b in enumerate(bounds)
    ]

    app = Dash(__name__)

    app.layout = html.Div([
        html.Div([
            dcc.Dropdown(
                inputs,
                'x0',
                id='input-dropdown',
                #style={'float': 'left','margin': 'auto'}
                style={'width': '20%', 'display': 'inline-block'},
            ),
            dcc.Dropdown(
                outputs,
                'y0',
                id='output-dropdown',
                #style={'float': 'right','margin': 'auto'}
                style={'width': '20%', 'display': 'inline-block'},
            )
        #], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
        #], style={'height': '10%', 'width': '48%', 'float': 'right', 'display': 'inline-block'}),
        ]),
        dcc.Graph(id='graph-with-slider')
    ] + sliders
    )


    @callback(
        Output('graph-with-slider', 'figure'),
        Input('input-dropdown', 'value'),
        Input('output-dropdown', 'value'),
        Input({"type": "slider", "index": ALL}, "value")
    )
    def update_figure(x_dim, y_dim, values):

        xs, ys, stds = gpc.get_1d_prediction(inputs.index(x_dim), np.array(values))
        fig = go.Figure([
            go.Scatter(
                name='Mean',
                x=xs,
                y=ys[:,outputs.index(y_dim)],
                mode='lines',
                line=dict(color='rgb(31, 119, 180)'),
                showlegend=False
            ),
            go.Scatter(
                name='Upper Bound',
                x=xs,
                y=ys[:,outputs.index(y_dim)] + stds[:,outputs.index(y_dim)],
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ),
            go.Scatter(
                name='Lower Bound',
                x=xs,
                y=ys[:,outputs.index(y_dim)] - stds[:,outputs.index(y_dim)],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False
            )
        ])

        fig.update_layout(transition_duration=100)

        return fig

    app.run(debug=True)

def dash_2d_version():
    bounds = [(0,1), (2,4), (-np.pi, np.pi)]
    inputs = ["x0", "x1", "x2"]
    outputs = ["y0", "y1"]
    gpc = GPChar(f, bounds, 3, 2, "data.csv", keyword_args={"a": 1, "b": 2})
    gpc.acquire_random_evaluations_in_thread(3)

    sliders = [
        dcc.Slider(
            b[0],
            b[1],
            step=(b[1]-b[0])/100,
            value=(b[1]+b[0])/2,
            marks={x: f"{x:.3g}" for x in np.arange(b[0], b[1], (b[1]-b[0])/10)},
            id={"type": 'slider', "index":i}
        )
        for i, b in enumerate(bounds)
    ]

    app = Dash(__name__)

    app.layout = html.Div([
        html.Div([
            dcc.Dropdown(
                inputs,
                'x0',
                id='input1-dropdown',
                #style={'float': 'left','margin': 'auto'}
                style={'width': '20%', 'display': 'inline-block'},
            ),
            dcc.Dropdown(
                inputs,
                'x1',
                id='input2-dropdown',
                #style={'float': 'left','margin': 'auto'}
                style={'width': '20%', 'display': 'inline-block'},
            ),
            dcc.Dropdown(
                outputs,
                'y0',
                id='output-dropdown',
                #style={'float': 'right','margin': 'auto'}
                style={'width': '20%', 'display': 'inline-block'},
            )
        ]),
        dcc.Graph(id='1d-graph',
                  style={'width': '33%', 'display': 'inline-block'}),
        dcc.Graph(id='mean-contour',
                  style={'width': '33%', 'display': 'inline-block'}),
        dcc.Graph(id='std-contour',
                  style={'width': '33%', 'display': 'inline-block'}),
    ] + sliders
    )


    @app.callback(
        Output('1d-graph', 'figure'),
        Output('mean-contour', 'figure'),
        Output('std-contour', 'figure'),
        Input('input1-dropdown', 'value'),
        Input('input2-dropdown', 'value'),
        Input('output-dropdown', 'value'),
        Input({"type": "slider", "index": ALL}, "value")
    )
    def update_figures(x1_dim, x2_dim, y_dim, values):
        #pdb.set_trace()
        #breakpoint()
        xs, ys, stds = gpc.get_1d_prediction(inputs.index(x1_dim), np.array(values))
        oned_fig = go.Figure([
            go.Scatter(
                name='Mean',
                x=xs,
                y=ys[:,outputs.index(y_dim)],
                mode='lines',
                line=dict(color='rgb(31, 119, 180)'),
                showlegend=False
            ),
            go.Scatter(
                name='Upper Bound',
                x=xs,
                y=ys[:,outputs.index(y_dim)] + stds[:,outputs.index(y_dim)],
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ),
            go.Scatter(
                name='Lower Bound',
                x=xs,
                y=ys[:,outputs.index(y_dim)] - stds[:,outputs.index(y_dim)],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False
            )
        ])

        oned_fig.update_layout(
            transition_duration=100,
            xaxis_title=x1_dim,
            yaxis_title=y_dim
        )


        x1s, x2s, ys, stds = gpc.get_2d_prediction(inputs.index(x1_dim), inputs.index(x2_dim), np.array(values))
        mean_zs = np.reshape(ys[:,outputs.index(y_dim)], (100,100))
        std_zs = np.reshape(stds[:,outputs.index(y_dim)], (100,100))

        mean_fig = go.Figure([
            go.Contour(
                name='Mean',
                x=x1s,
                y=x2s,
                z=mean_zs,
                colorbar=dict(
                    title=y_dim,
                    titleside='right',
                    titlefont=dict(
                        size=14,
                        family='Arial, sans-serif')
                ),
                line=dict(width=0),
                ncontours=20,
            ),
            go.Scatter(
                name='1d-marker-line',
                x=x1s,
                y=np.full(x1s.shape, values[inputs.index(x2_dim)]),
            )
        ])

        mean_fig.update_layout(
            transition_duration=100,
            xaxis_title=x1_dim,
            yaxis_title=x2_dim
        )

        std_fig = go.Figure([
            go.Contour(
                name='Uncertainty',
                x=x1s,
                y=x2s,
                z=std_zs,
                colorbar=dict(
                    title=y_dim + " uncertainty",
                    titleside='right',
                    titlefont=dict(
                        size=14,
                        family='Arial, sans-serif')
                ),
                colorscale='Reds',
                line=dict(width=0),
                zmin=0,
                zmax=np.max(std_zs),
                ncontours=20,
            ),
            go.Scatter(
                name='1d-marker-line',
                x=x1s,
                y=np.full(x1s.shape, values[inputs.index(x2_dim)]),
            )
        ])

        std_fig.update_layout(
            transition_duration=100,
            xaxis_title=x1_dim,
            yaxis_title=x2_dim
        )

        return oned_fig, mean_fig, std_fig

    app.run(debug=True,use_reloader=False)

if __name__ == '__main__':
    dash_2d_version()
    #dash_version()
