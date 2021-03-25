import os
import threading
import tkinter.filedialog
import tkinter as tk
from tkinter import font
from tkinter import ttk, TOP, BOTH, X, N, LEFT, RIGHT, StringVar, Y, SUNKEN, GROOVE, DoubleVar, IntVar, BOTTOM
from ttkthemes import ThemedStyle
import queue
# import tksvg
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

from sound.features import hardness, depth, brightness, roughness, warmth, sharpness, boominess

TIMBRAL_F = {
    'hardness': hardness,
    'depth': depth,
    'brightness': brightness,
    'roughness': roughness,
    'warmth': warmth,
    'sharpness': sharpness,
    'boominess': boominess,
}

sns.set(rc={
    'axes.facecolor':'#33393b',
    'figure.facecolor':'#33393b',
    'grid.color': '#5F6A6E',
    'axes.edgecolor': 'white',
    'text.color': 'white',
    'axes.titlecolor': 'white',
    'ytick.color': 'white',
    'xtick.color': 'white',
    'axes.labelcolor': 'white',
    })



ENDPOINT_URL="http://127.0.0.1:8002"
CLASSES = ['Snare','Kick', 'Hat', 'Tom', 'Cymbal', 'Clap', 'Cowbell', 'Conga', 'Shaken']

timbral_features = pd.DataFrame(columns= list(TIMBRAL_F.keys()) + ['class'])


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
        sns.scatterplot(x=x_axis.get(), y=y_axis.get(), hue='class', data=timbral_features, ax=self.ax )
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
        points = [ (x, y) for x,y in zip(timbral_features[x_axis.get()], timbral_features[y_axis.get()])]
        coords, index = closest(xydata, points)
        path = timbral_features.iloc[index].name
        print(path)
        play_sound(path)
        # self.redraw_figure()


def play_sound(path):
    print(path)
    threading.Thread(target=playsound, args=(str(path),), daemon=True).start()

window = tk.Tk()

window.tk.eval("""
    set base_theme_dir ../awthemes-10.3.0/

    package ifneeded awthemes 10.3.0 \
        [list source [file join $base_theme_dir awthemes.tcl]]
    package ifneeded colorutils 4.8 \
        [list source [file join $base_theme_dir colorutils.tcl]]
    package ifneeded awdark 7.11 \
        [list source [file join $base_theme_dir awdark.tcl]]
    # ... (you can add the other themes from the package if you want
    """)
window.tk.call("package", "require", 'awdark')
x_axis = StringVar(window)
x_axis.set('hardness')
y_axis = StringVar(window)
y_axis.set('depth')

style = ThemedStyle(window)
style.set_theme("awdark")

my_font = font.Font(family="terminus", size=16)
print(font.families())
style.configure(".", font=my_font)

root = ttk.Frame(window)
root.pack(side=TOP,fill=BOTH, expand=True)
# progress_frame = ttk.Frame(root, height=15)
# progress_frame.pack_propagate(0)
# progress_frame.pack(side=BOTTOM, fill=X, expand=True)
main_frame = ttk.Frame(root)
main_frame.pack(side=TOP, fill=BOTH, expand=True)
left_frame = ttk.Frame(main_frame, width=250)
left_frame.pack_propagate(0)
left_frame.pack(side=LEFT, fill=Y)
right_frame = ttk.Frame(main_frame)
right_frame.pack(side=LEFT, fill=Y)
user_frame = ttk.Frame(left_frame)
user_frame.pack(side=TOP)

progress_var = IntVar(window)
progress_var.set(100)
progress = ttk.Progressbar(root, maximum=100, variable=progress_var)
progress.pack(side=BOTTOM,fill=X, expand=True)

label = ttk.Label(user_frame, text="User")
user_entry = ttk.Entry(user_frame)

label.pack(side=LEFT)
user_entry.pack(side=LEFT)

sample_dict = {}

file_types = ['*.wav', '*.mp3', '*.flac', '*.aiff' ]


queue = queue.Queue()
global thread

def import_samples_async(files):
    global thread
    thread = threading.Thread(target=import_samples_thread, args=(files,), daemon=True)
    window.after(500, update_progress)
    thread.start()

def import_samples_thread(files):
    progress_var.set(0)
    predict_url = ENDPOINT_URL + '/predict'
    for i, file in enumerate(files):
        print(file)

        with open(file, 'rb') as f:
            r = requests.post(url=predict_url, files={'samplefile': f})
        # print(r.content)
        progress_var.set(100*i/len(files))
        # print(progress_var.get())
        if r.status_code != 200:
            continue
        result= r.json()
        # new_samples[file] = result["class"]
        df  = pd.DataFrame(columns= list(TIMBRAL_F.keys()) + ['class'])
        for feature, transform in TIMBRAL_F.items():
            print(feature)
            df.at[file,feature] = transform(str(file))
        df.at[file,'class'] = result["class"]
        

        queue.put({
            'id': i,
            'df': df,
            })
    progress_var.set(100)

def update_progress():
    global timbral_features
    if not thread.is_alive() and queue.empty():
        # print('END')
        return

    while not queue.empty():
        data = queue.get()
        timbral_features = timbral_features.append(data['df'])
        file = data['df'].index[0]
        class_ = data['df']['class'][0]
        update_tree({file:class_})
        # print(timbral_features)
    canvas.redraw_figure()

    window.after(500, update_progress)

def import_samples():
    global sample_dict
    directory = tkinter.filedialog.askdirectory()
    print(directory)

    files = []
    for file_type in file_types:
        for path in Path(directory).rglob(file_type):
            files.append(path)
            print(path.name)

    import_samples_async(files)

    # predict_url = ENDPOINT_URL + '/predict'
    # new_samples = {}
    # progress_var.set(0)
    # for i, file in enumerate(files):
    #     print(file)

    #     with open(file, 'rb') as f:
    #         r = requests.post(url=predict_url, files={'samplefile': f})
    #     print(r.content)
    #     progress_var.set(100*i/len(files))
    #     print(progress_var.get())
    #     if r.status_code != 200:
    #         continue
    #     result= r.json()
    #     new_samples[file] = result["class"]
    #     for feature, transform in TIMBRAL_F.items():
    #         print(feature)
    #         timbral_features.at[file,feature] = transform(str(file))
    #     timbral_features.at[file,'class'] = result["class"]
    #     # timbral_features.at[file] = hardness(str(file)), depth(str(file)), result["class"]
    #     # print(timbral_features)

    # for file, class_ in new_samples.items():
    #     print(f"file: {file}\nclass: {class_}")

    # sample_dict.update(new_samples)
    # progress_var.set(100)
    
    # update_tree(new_samples)
    # canvas.redraw_figure()
    


import_button = ttk.Button(
    left_frame,
    text="Import",
    command=import_samples
)

import_button.pack()

path_iid = {}

treeview = ttk.Treeview(left_frame)
treeview.heading("#0",text="Files",anchor=tk.W)
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

treeview.pack(side=TOP, fill=BOTH, expand=True)


def update_tree(new_samples):
    # global sample_dict
    global treeview
    
    for file, class_ in new_samples.items():
        iid = treeview.insert(parents[class_], "end", text=os.path.basename(file))
        path_iid[iid] = file

timbral_frame = ttk.Frame(right_frame, relief=SUNKEN, borderwidth=5)
timbral_frame.pack(side=LEFT)


canvas = timbral_space(timbral_frame)
# refresh_button = ttk.Button(
#     text="Update",
#     command=canvas.redraw_figure
# )
# refresh_button.pack()

class TimbralRadioFrame(ttk.Frame):

    def __init__(self, root):
        super().__init__(root)

        left = ttk.Frame(root)
        left.pack(side=TOP)
        right = ttk.Frame(root)
        right.pack(side=TOP)
        x_label = ttk.Label(left, text="X axis")
        x_label.pack()
        y_label = ttk.Label(right, text="Y axis")
        y_label.pack()
        for feat in TIMBRAL_F:
            rb = ttk.Radiobutton(left, command=canvas.redraw_figure, text=feat, value=feat, variable=x_axis)
            rb.pack(anchor='w')

        for feat in TIMBRAL_F:
            rb = ttk.Radiobutton(right, command=canvas.redraw_figure, text=feat, value=feat, variable=y_axis)
            rb.pack(anchor='w')

radios = TimbralRadioFrame(right_frame)
radios.pack(side=LEFT, expand=False)


window.mainloop()