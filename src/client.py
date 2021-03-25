import os
import threading
import tkinter.filedialog
import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedStyle
from playsound import playsound
# from typing import Callable

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import seaborn as sns
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from pathlib import Path

import requests

import os

from sound.features import hardness, depth

ENDPOINT_URL="http://127.0.0.1:8002"
CLASSES = ['Snare','Kick', 'Hat', 'Tom', 'Cymbal', 'Clap', 'Cowbell', 'Conga', 'Shaken']

timbral_features = pd.DataFrame(columns=['hardness', 'depth', 'class'])


class timbral_space(FigureCanvasTkAgg):

    def __init__(self, parent):
        self.figure = self.create_figure()
        super().__init__(self.figure, master=parent)
        self.draw()
        self.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        self.mpl_connect('button_press_event', self.event_key_press)

    
    def create_figure(self) -> Figure:
        # generate some data
        # matrix = np.random.randint(20, size=(10, 10))
        # plot the data
        figure = Figure(figsize=(6, 6))
        self.ax = figure.subplots()
        # sns.heatmap(matrix, square=True, cbar=False, ax=self.ax )
        sns.scatterplot(x='hardness', y='depth', hue='class', data=timbral_features, ax=self.ax )
        return figure


    def redraw_figure(self):
        self.figure = self.create_figure()
        self.draw()

    def event_key_press(self, event):
        def closest(point, points):
            dist = cdist([point], points)
            return points[dist.argmin()], dist.argmin()
        print("you pressed {}".format(event))
        print("you pressed {}".format(event.__dict__))
        xydata = event.xdata, event.ydata
        points = [ (x, y) for x,y in zip(timbral_features['hardness'], timbral_features['depth'])]
        coords, index = closest(xydata, points)
        path = timbral_features.iloc[index].name
        print(path)
        play_sound(path)
        # self.redraw_figure()


def play_sound(path):
    print(path)
    threading.Thread(target=playsound, args=(str(path),), daemon=True).start()

window = tk.Tk()

style = ThemedStyle(window)
style.set_theme("breeze")

# s = ttk.Style()
# print(s.theme_names())
# s.theme_use("clearlooks")

label = ttk.Label(text="User")
user_entry = ttk.Entry()

label.pack()
user_entry.pack()

sample_dict = {}

file_types = ['*.wav', '*.mp3', '*.flac', '*.aiff' ]
def import_samples():
    global sample_dict
    directory = tkinter.filedialog.askdirectory()
    print(directory)

    files = []
    for file_type in file_types:
        for path in Path(directory).rglob(file_type):
            files.append(path)
            print(path.name)

    predict_url = ENDPOINT_URL + '/predict'
    new_samples = {}
    for file in files:
        print(file)

        with open(file, 'rb') as f:
            r = requests.post(url=predict_url, files={'samplefile': f})
        print(r.content)
        if r.status_code != 200:
            continue
        result= r.json()
        new_samples[file] = result["class"]
        timbral_features.at[file] = hardness(str(file)), depth(str(file)), result["class"]
        # print(timbral_features)

    for file, class_ in new_samples.items():
        print(f"file: {file}\nclass: {class_}")

    sample_dict.update(new_samples)
    update_tree(new_samples)


import_button = ttk.Button(
    text="Import",
    command=import_samples
)

import_button.pack()

path_iid = {}

treeview = ttk.Treeview(window)
treeview.heading("#0",text="Fichier",anchor=tk.W)
parents = {}
for class_ in CLASSES:
    parents[class_] = treeview.insert("", 'end', class_, text=class_)

def OnTreeDoubleClick(event):
    iid = treeview.identify('item', event.x,event.y)
    # iid = treeview.item(item, "iid")
    print(iid)
    if iid in CLASSES:
        pass
    else:
        play_sound(path_iid[iid])

treeview.bind("<Double-1>", OnTreeDoubleClick)

treeview.pack()


def update_tree(new_samples):
    # global sample_dict
    global treeview
    
    for file, class_ in new_samples.items():
        iid = treeview.insert(parents[class_], "end", text=os.path.basename(file))
        path_iid[iid] = file

canvas = timbral_space(window)
refresh_button = ttk.Button(
    text="Update",
    command=canvas.redraw_figure
)
refresh_button.pack()

window.mainloop()