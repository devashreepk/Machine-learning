#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 13:48:26 2025

@author: devashreepk
"""

import tkinter as tk

window = tk.Tk()
window.title("Tkinter Test")
window.geometry("200x100")

label = tk.Label(window, text="Tkinter is working!")
label.pack()

window.mainloop()
