import pdb
import time
import threading

from dash import Dash, dcc, html, Input, Output, callback, ALL
import plotly.graph_objects as go
import plotly.express as px

import numpy as np
import matplotlib.pyplot as plt
from .gpchar import GPChar
#from gpchar import GPChar
from flask.helpers import get_root_path

def make_color_transparent(col, transparency=0.3):
    return 'rgba' + col[3:-1]  + ', ' + str(transparency) + ')'

def launch_dash_app_in_thread(gpc: GPChar, bounds: list[tuple], input_names: list[str], output_names: list[str]) -> threading.Thread:
    thread = threading.Thread(target=launch_dash_app, args=(gpc, bounds, input_names, output_names))
    thread.start()
    return thread

def launch_dash_app(gpc: GPChar, bounds: list[tuple], input_names: list[str], output_names: list[str]) -> None:
    app = create_dash_app(gpc, bounds, input_names, output_names)
    app.run(debug=True,use_reloader=False)

def create_dash_app(gpc: GPChar, bounds: list[tuple], input_names: list[str], output_names: list[str]) -> None:

    # Create sliders for each input dimension
    # Each slider is a
    # div containing a
    #   div with dimension name and value,
    #   and a div with the slider object
    sliders = [
        html.Div(
            children=[
                html.Div([
                    html.Span(input_names[i]+": "),
                    html.Span(
                        f"{(b[1]+b[0])/2:.3g}",
                        id={"type": 'slidertext', "index": i},
                        className="variable_number",
                    ),
                ], style=dict(width='100px'),),
                html.Div(
                    children=[
                        dcc.Slider(
                            b[0],
                            b[1],
                            step=(b[1]-b[0])/100,
                            value=(b[1]+b[0])/2,
                            # Marks don't work if it is a float of an integer value... Very stupid... https://github.com/plotly/dash-core-components/issues/159
                            marks={x+((b[1]-b[0])/10000 if x < (b[0]+b[1])/2 else -(b[1]-b[0])/10000): f"{x:.3g}" for x in np.arange(b[0], b[1], (b[1]-b[0])/10)},
                            id={"type": 'slider', "index":i},
                        )
                    ],
                    className="sliderdiv",
                ),
            ],
            className="vcenter hcenter",
        )
        for i, b in enumerate(bounds)
    ]

    # Create the dash app. assets_folder is where the css files are stored
    # any files in this folder are automatically served
    app = Dash(__name__, assets_folder="assets")

    app.layout = html.Div(
        children=[
            # Dropdowns for input and output dimensions
            # This div has three children, each one a div
            # containing a description text and a dropdown object
            html.Div(
                children = [
                    html.Div([
                        html.Span("Input 1: "),
                        dcc.Dropdown(
                            input_names,
                            input_names[0],
                            id='input1-dropdown',
                            className='input_dropdown',
                        ),
                    ], className="vcenter"),
                    html.Div([
                        html.Span("Input 2: "),
                        dcc.Dropdown(
                            input_names,
                            input_names[1],
                            id='input2-dropdown',
                            className='input_dropdown',
                        ),
                    ], className="vcenter"),
                    html.Div([
                        html.Span("Output(s):  "),
                        dcc.Dropdown(
                            output_names,
                            [output_names[0]],
                            multi=True,
                            id='output-dropdown',
                            className='output_dropdown',
                        )
                    ], className="vcenter"),
                ],
                className="round_corners vcenter gapped",
            ),
            # Graphs
            dcc.Graph(id='1d-graph',
                      style={'width': '33%', 'display': 'inline-block'}),
            dcc.Graph(id='mean-contour',
                      style={'width': '33%', 'display': 'inline-block'}),
            dcc.Graph(id='std-contour',
                      style={'width': '33%', 'display': 'inline-block'}),
            # Sliders
            html.Div(
                children=sliders,
                className="round_corners",
            )
        ]
    )


    # Whenever a slider or dropdown changes, call this function
    @app.callback(
        Output('1d-graph', 'figure'),
        Output('mean-contour', 'figure'),
        Output('std-contour', 'figure'),
        Output({"type": "slidertext", "index": ALL}, "children"),
        Input('input1-dropdown', 'value'),
        Input('input2-dropdown', 'value'),
        Input('output-dropdown', 'value'),
        Input({"type": "slider", "index": ALL}, "value")
    )
    def update_figures(x1_dim, x2_dim, y_dims, values):

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
                    line=dict(width=0, color=colors[i]),
                    showlegend=False
                )
            )
            scatterplots.append(
                go.Scatter(
                    name='Lower Bound',
                    x=xs,
                    y=y-std,
                    line=dict(width=0, color=colors[i]),
                    mode='lines',
                    fillcolor=make_color_transparent(colors[i]),
                    fill='tonexty',
                    showlegend=False
                )
            )

        # WARNING: Non-general code incomming... I just want to make a plot and I'm lazy, but this certainly shouldn't go into the package code
        if x1_dim == "k" and main_y_dim[0] == 'f':
            scatterplots.append(
                go.Scatter(
                    name="Cont.",
                    x=xs,
                    y=1/(2*np.pi) * (xs * np.pi / (1e-9*values[input_names.index("a")])) * 299792458 / 1.45,
                    line=dict(dash="dash", color="black"),
                    mode='lines',
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

        # Update the texts for the dimension sliders
        slidertexts = [f"{x:.3g}" for x in values]

        return oned_fig, mean_fig, std_fig, slidertexts

    return app

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
