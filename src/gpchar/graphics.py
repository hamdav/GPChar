import pdb
import time
import threading

from dash import Dash, dcc, html, Input, Output, callback, ALL
import plotly.graph_objects as go
import plotly.express as px

import numpy as np
import matplotlib.pyplot as plt

def make_color_transparent(col, transparency=0.3):
    return 'rgba' + col[3:-1]  + ', ' + str(transparency) + ')'


def launch_dash_app(gpc, bounds, input_names, output_names):

    # Create sliders for each input dimension
    sliders = [
        html.Div([
            input_names[i],
            dcc.Slider(
                b[0],
                b[1],
                step=(b[1]-b[0])/100,
                value=(b[1]+b[0])/2,
                # Marks don't work if it is a float of an integer value... Very stupid... https://github.com/plotly/dash-core-components/issues/159
                marks={x+((b[1]-b[0])/10000 if x < (b[0]+b[1])/2 else -(b[1]-b[0])/10000): f"{x:.3g}" for x in np.arange(b[0], b[1], (b[1]-b[0])/10)},
                id={"type": 'slider', "index":i},
            )
        ])
        for i, b in enumerate(bounds)
    ]

    app = Dash(__name__)

    app.layout = html.Div([
        # Dropdowns for input and output dimensions
        html.Div(children = [
            "Input 1: ",
            dcc.Dropdown(
                input_names,
                input_names[0],
                id='input1-dropdown',
                style={'width': '20%', 'display': 'inline-block'},
            ),
            "Input 2: ",
            dcc.Dropdown(
                input_names,
                input_names[1],
                id='input2-dropdown',
                style={'width': '20%', 'display': 'inline-block'},
            ),
            "Output(s):  ",
            dcc.Dropdown(
                output_names,
                output_names[0],
                multi=True,
                id='output-dropdown',
                style={'width': '50%', 'display': 'inline-block'},
            )
        ], style=dict(display='flex')),
        # Graphs
        dcc.Graph(id='1d-graph',
                  style={'width': '33%', 'display': 'inline-block'}),
        dcc.Graph(id='mean-contour',
                  style={'width': '33%', 'display': 'inline-block'}),
        dcc.Graph(id='std-contour',
                  style={'width': '33%', 'display': 'inline-block'}),
    ] + sliders
    )


    # Whenever a slider or dropdown changes, call this function
    @app.callback(
        Output('1d-graph', 'figure'),
        Output('mean-contour', 'figure'),
        Output('std-contour', 'figure'),
        Input('input1-dropdown', 'value'),
        Input('input2-dropdown', 'value'),
        Input('output-dropdown', 'value'),
        Input({"type": "slider", "index": ALL}, "value")
    )
    def update_figures(x1_dim, x2_dim, y_dims, values):

        # If only one option is selected, dash gives me a string with the option
        # otherwise, dash gives me a list. This just avoids duplicating code
        # and makes it always a list
        if isinstance(y_dims, str):
            y_dims = [y_dims]

        main_y_dim = y_dims[0]

        # Update the 1D graph
        xs, ys, stds = gpc.get_1d_prediction(input_names.index(x1_dim), np.array(values))

        # Made these up, with inspiration from matplotlib, but I like red first...
        colors = [
            "rgb(255,120,14)",
            "rgb(22,120,220)",
            "rgb(30,200,30)",
            "rgb(222,44,44)",
            "rgb(120,85,77)",
            "rgb(222,100,230)",
            "rgb(180,180,180)",
            "rgb(222,222,33)",
            "rgb(27,222,233)",
        ]
        scatterplots = []
        for i, y_dim in enumerate(y_dims):
            y = ys[:,output_names.index(y_dim)] if len(ys.shape)==2 else ys
            std = stds[:,output_names.index(y_dim)] if len(stds.shape)==2 else stds
            scatterplots.append(
                go.Scatter(
                    name=y_dim,
                    x=xs,
                    y=y,
                    mode='lines',
                    line=dict(color=colors[i]),
                )
            )
            scatterplots.append(
                go.Scatter(
                    name='Upper Bound',
                    x=xs,
                    y=y+std,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                )
            )
            scatterplots.append(
                go.Scatter(
                    name='Lower Bound',
                    x=xs,
                    y=y-std,
                    line=dict(width=0),
                    mode='lines',
                    fillcolor=make_color_transparent(colors[i]),
                    fill='tonexty',
                    showlegend=False
                )
            )

        oned_fig = go.Figure(scatterplots)

        oned_fig.update_layout(
            transition_duration=100,
            xaxis_title=x1_dim,
            yaxis_title=main_y_dim
        )


        # Update the 2D contour plots
        x1s, x2s, ys, stds = gpc.get_2d_prediction(input_names.index(x1_dim), input_names.index(x2_dim), np.array(values))
        y = ys[:,output_names.index(main_y_dim)] if len(ys.shape)==2 else ys
        std = stds[:,output_names.index(main_y_dim)] if len(stds.shape)==2 else stds

        mean_zs = np.reshape(y, (100,100))
        std_zs = np.reshape(std, (100,100))

        mean_fig = go.Figure([
            go.Contour(
                name='Mean',
                x=x1s,
                y=x2s,
                z=mean_zs,
                colorbar=dict(
                    title=main_y_dim,
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
                y=np.full(x1s.shape, values[input_names.index(x2_dim)]),
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
                    title=main_y_dim + " uncertainty",
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
                y=np.full(x1s.shape, values[input_names.index(x2_dim)]),
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
    from gpchar import GPChar

    def f(x, a, b):
        print("Running f", x)
        time.sleep(10)
        m = np.array([
            [np.cos(a)*np.cos(b), np.sin(a)*np.cos(b), np.cos(b)],
            [-np.sin(a)*np.cos(b), np.sin(a)*np.cos(b), -np.cos(b)]])
        print("Run done ", x)
        return np.dot(m, np.sin(2*x)) + 100

    bounds = [(0,1), (2,4), (-np.pi, np.pi)]
    inputs = ["x0", "x1", "x2"]
    outputs = ["y0", "y1"]
    gpc = GPChar(f, bounds, 3, 2, "data.csv", keyword_args={"a": 1, "b": 2})
    launch_dash_app(gpc, bounds, inputs, outputs)
