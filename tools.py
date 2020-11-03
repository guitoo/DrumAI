import ipywidgets
from ipywidgets import Output, VBox
from IPython.display import display
import IPython
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from librosa.core import load

def display_embedding(x, y, classes, files):
    embedding_df = pd.DataFrame({
        'x': x,
        'y': y,
        'class': classes,
        'file': files    
                         })
    
    scatter = px.scatter(
        embedding_df,
        x='x',
        y='y',
        color='class',
        hover_data=['file']
    )
    figure = go.FigureWidget(scatter)

    out = Output()

    @out.capture(clear_output=False, wait=True)
    def play_sound(trace, points, selector):
        if len(points.point_inds) != 1:
           return
        #print(points)
        out.clear_output()
        path = trace.customdata[points.point_inds[0]][0]
        #print(path)
        sound, sr = load(path)
        IPython.display.display(IPython.display.Audio(sound, rate=sr, autoplay=True))

    for trace in figure.data:
        trace.on_click(play_sound)


    box = VBox([figure, out])
    display(box)