import os
import tkinter.filedialog
import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedStyle

from pathlib import Path

import requests

ENDPOINT_URL="http://127.0.0.1:8002"

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

file_types = ['*.wav', '*.mp3', '*.flac', '*.aiff' ]
def import_samples():
    directory = tkinter.filedialog.askdirectory()
    print(directory)

    files = []
    for file_type in file_types:
        for path in Path(directory).rglob(file_type):
            files.append(path)
            print(path.name)

    predict_url = ENDPOINT_URL + '/predict'
    classes=[]
    for file in files:
        print(predict_url)

        with open(file, 'rb') as f:
            r = requests.post(url=predict_url, files={'samplefile': f})
        print(r.content)
        result= r.json()
        classes.append(result["class"])

    for file, class_ in zip(files, classes):
        print(f"file: {file}\nclass: {class_}")


import_button = ttk.Button(
    text="Import",
    command=import_samples
)

import_button.pack()




window.mainloop()