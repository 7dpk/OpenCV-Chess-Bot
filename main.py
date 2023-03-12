from configparser import ConfigParser
import tkinter as tk
import chess
import chess.engine
from bot import play

engine = chess.engine.SimpleEngine.popen_uci('stockfish')
depth = 12
time_control = 0.1
hash = 128
thread = 2
play_by_depth_bool = False

'''Functions to handle buttons clicks'''


def thread_increase():
    value = int(thread_entry.get())
    thread_entry.delete(0, tk.END)
    thread_entry.insert(0, str(value + 1))


def thread_decrease():
    value = int(thread_entry.get())
    if value == 1:
        return
    thread_entry.delete(0, tk.END)
    thread_entry.insert(0, str(value - 1))


def hash_increase():
    value = int(hash_entry.get())
    hash_entry.delete(0, tk.END)
    hash_entry.insert(0, str(value + 128))


def hash_decrease():
    value = int(hash_entry.get())
    if value <= 129:
        return
    hash_entry.delete(0, tk.END)
    hash_entry.insert(0, str(value - 128))


def depth_increase():
    value = int(depth_entry.get())
    depth_entry.delete(0, tk.END)
    depth_entry.insert(0, str(value + 1))


def depth_decrease():
    value = int(depth_entry.get())
    if value == 1:
        return
    depth_entry.delete(0, tk.END)
    depth_entry.insert(0, str(value - 1))


def time_increase():
    value = float(time_entry.get())
    time_entry.delete(0, tk.END)
    time_entry.insert(0, str(value + 0.1))


def time_decrease():
    value = float(time_entry.get())
    if value <= 0.1:
        return
    time_entry.delete(0, tk.END)
    time_entry.insert(0, str(value - 0.1))


def play_by_depth():
    global play_by_depth_bool
    play_by_depth_bool = True
    global depth
    depth = int(depth_entry.get())
    global hash
    hash = int(hash_entry.get())
    global thread
    thread = int(thread_entry.get())
    return play(chess.Board(), engine, thread, hash,
                depth, time_control, play_by_depth_bool)


def play_by_time():
    global play_by_depth_bool
    play_by_depth_bool = False
    global depth
    depth = int(depth_entry.get())
    global hash
    hash = int(hash_entry.get())
    global thread
    thread = int(thread_entry.get())
    global time_control
    time_control = float(time_entry.get())
    return play(chess.Board(), engine, thread, hash,
                depth, time_control, play_by_depth_bool)


parser = ConfigParser()
parser.read('default.ini')

window = tk.Tk()
window.title('CB 0.1.3')
window.rowconfigure(1, minsize=500, weight=1)
window.columnconfigure(0, minsize=500, weight=1)

welcome_text = tk.Label(
    master=window, text='Welcome!!! Loading default settings from default.config')
'''making a another separate frame for controls'''
controls = tk.Frame(window)
controls.rowconfigure(0, minsize=250, weight=1)
controls.columnconfigure([0, 1], minsize=250, weight=1)

''' creating engine parameter and time control frame '''
frame_engine = tk.Frame(controls, relief=tk.RAISED, borderwidth=3)
frame_time_control = tk.Frame(controls, relief=tk.RAISED, borderwidth=3)
frame_engine.rowconfigure([1, 2], weight=1)
frame_engine.columnconfigure(0, weight=1)
frame_time_control.rowconfigure([1, 2], weight=1)
frame_time_control.columnconfigure(0, weight=1)


'''engine and time control label'''
engine_label = tk.Label(master=frame_engine, text='Engine Parameters')
time_label = tk.Label(master=frame_time_control, text='Time Control')

'''thread frame and hash frame'''
frame_thread = tk.Frame(master=frame_engine, relief=tk.RAISED, borderwidth=3)
frame_hash = tk.Frame(master=frame_engine, relief=tk.RAISED, borderwidth=3)
frame_thread.rowconfigure([1], weight=1)
frame_thread.columnconfigure([0, 1, 2], weight=1)
frame_hash.rowconfigure([1], weight=1)
frame_hash.columnconfigure([0, 1, 2], weight=1)

'''depth and time frame'''
frame_depth = tk.Frame(master=frame_time_control,
                       relief=tk.RAISED, borderwidth=3)
frame_time = tk.Frame(master=frame_time_control,
                      relief=tk.RAISED, borderwidth=3)
frame_depth.rowconfigure([1], weight=1)
frame_depth.rowconfigure([2], weight=1)

frame_depth.columnconfigure([0, 1, 2], weight=1)
frame_time.rowconfigure([1], weight=1)
frame_time.rowconfigure([2], weight=1)
frame_time.columnconfigure([0, 1, 2], weight=1)


'''labels for thread hash depth and time'''
tk.Label(master=frame_thread, text='Threads', font=(
    "Calibri", 12), justify="center").grid(row=0, column=0, sticky='n')
tk.Label(master=frame_hash, text='Hash', font=(
    "Calibri", 12), justify="center").grid(row=0, column=0, sticky='n')
tk.Label(master=frame_depth, text='Depth', font=(
    "Calibri", 12), justify="center").grid(row=0, column=0, sticky='n')
tk.Label(master=frame_time, text='Time', font=(
    "Calibri", 12), justify="center").grid(row=0, column=0, sticky='n')


'''Buttons and Entry box for thread hash time depth'''
tk.Button(master=frame_thread, text='-', command=thread_decrease, font=(
    "Calibri", 24), justify="center", height=3, width=3).grid(row=1, column=0, sticky='nsew')
thread_entry = tk.Entry(master=frame_thread, font=(
    "Calibri", 12), justify="center", width=2)
thread_entry.grid(row=1, column=1, sticky='nsew')
tk.Button(master=frame_thread, text='+', command=thread_increase, font=(
    "Calibri", 24), justify="center", height=3, width=3).grid(row=1, column=2, sticky='nsew')

tk.Button(master=frame_hash, text='-', command=hash_decrease, font=(
    "Calibri", 24), justify="center", height=3, width=3).grid(row=1, column=0, sticky='nsew')
hash_entry = tk.Entry(master=frame_hash, font=(
    "Calibri", 12), justify="center", width=2)
hash_entry.grid(row=1, column=1, sticky='nsew')
tk.Button(master=frame_hash, text='+', command=hash_increase, font=(
    "Calibri", 24), justify="center", height=3, width=3).grid(row=1, column=2, sticky='nsew')

tk.Button(master=frame_depth, text='-', command=depth_decrease, font=(
    "Calibri", 24), justify="center", height=3, width=3).grid(row=1, column=0, sticky='nsew')
depth_entry = tk.Entry(master=frame_depth, font=(
    "Calibri", 12), justify="center", width=2)
depth_entry.grid(row=1, column=1, sticky='nsew')
tk.Button(master=frame_depth, text='+', command=depth_increase, font=(
    "Calibri", 24), justify="center", height=3, width=3).grid(row=1, column=2, sticky='nsew')
tk.Button(master=frame_depth, text='Play by depth', command=play_by_depth, font=(
    "Calibri", 10), justify="center", height=3, width=3).grid(row=2, column=2, sticky='ew')

tk.Button(master=frame_time, text='-', command=time_decrease, font=(
    "Calibri", 24), justify="center", height=3, width=3).grid(row=1, column=0, sticky='nsew')
time_entry = tk.Entry(master=frame_time, font=(
    "Calibri", 12), justify="center", width=2)
time_entry.grid(row=1, column=1, sticky='nsew')
tk.Button(master=frame_time, text='+', command=time_increase, font=(
    "Calibri", 24), justify="center", height=3, width=3).grid(row=1, column=2, sticky='nsew')
tk.Button(master=frame_time, text='Play by time', command=play_by_time, font=(
    "Calibri", 10), justify="center", height=3, width=3).grid(row=2, column=2, sticky='ew')


'''End of making buttons and entry boxes'''

welcome_text.grid(row=0)
controls.grid(row=1, sticky='nsew')

frame_engine.grid(row=0, column=0, sticky='nsew')
frame_time_control.grid(row=0, column=1, sticky='nsew')

time_label.grid(row=0, column=0, sticky='nsew')
engine_label.grid(row=0, column=0, sticky='nsew')

frame_thread.grid(row=1, column=0, sticky='nsew')
frame_hash.grid(row=2, column=0, sticky='nsew')

frame_depth.grid(row=1, column=0, sticky='nsew')
frame_time.grid(row=2, column=0, sticky='nsew')


'''setting up default values of entries'''
thread_entry.delete(0, tk.END)
thread_entry.insert(0, parser.get('Engine Default Settings', 'thread'))

hash_entry.delete(0, tk.END)
hash_entry.insert(0, parser.get('Engine Default Settings', 'hash'))

depth_entry.delete(0, tk.END)
depth_entry.insert(0, parser.get('Time Control', 'depth'))

time_entry.delete(0, tk.END)
time_entry.insert(0, parser.get('Time Control', 'time'))

window.mainloop()
