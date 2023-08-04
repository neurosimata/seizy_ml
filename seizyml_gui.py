# -*- coding: utf-8 -*-
import tkinter as tk
import subprocess

class CommandButton(tk.Button):
    def __init__(self, master=None, command=None, entry=None, **kwargs):
        super().__init__(master, command=self.execute_command, **kwargs)
        self.cli_command = command
        self.entry = entry

    def execute_command(self):
        if self.cli_command == "setpath" and self.entry:
            path = self.entry.get()
            subprocess.run(["python", "cli.py", self.cli_command, path])
        else:
            subprocess.run(["python", "cli.py", self.cli_command])

root = tk.Tk()
root.title("CLI GUI")
root.configure(bg='white')

commands = ["setpath", "filecheck", "preprocess", "predict", "verify", "extractproperties"]
entry = None

for command in commands:
    frame = tk.Frame(root, bg='white')
    frame.pack(fill=tk.X, padx=10, pady=5)

    if command == "setpath":
        entry = tk.Entry(frame)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=5)

    button = CommandButton(frame, text=command, command=command, entry=entry, bg='lightblue')
    button.pack(side=tk.LEFT, padx=10, pady=5)

    entry = None

root.mainloop()
