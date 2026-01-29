import tkinter as tk
from tkinter import ttk
import ctypes
import platform
import math
import random
import json
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

if platform.system() == 'Windows':
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

class PhonexQEngine:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.lib = self._load_library()
        self.sys_ptr = None
        self._init_c_types()
        self.reset()
        
    def _load_library(self):
        try:
            lib_path = os.path.abspath("./phonexq_core.so")
            if os.path.exists(lib_path):
                return ctypes.CDLL(lib_path)
        except:
            pass
        return None
            
    def _init_c_types(self):
        class Complex(ctypes.Structure):
            _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]
        self.Complex = Complex
        if self.lib:
            self.lib.init_system.restype = ctypes.c_void_p
            self.lib.get_probability.restype = ctypes.c_double
        
    def reset(self):
        if self.lib and self.sys_ptr:
            self.lib.free_system(self.sys_ptr)
        if self.lib:
            self.sys_ptr = self.lib.init_system(self.num_qubits)
        
    def apply_gate(self, gate_name, target, control=None):
        if not self.lib:
            return
        gates = {
            'X': (0, 1, 1, 0),
            'Y': (0, -1j, 1j, 0),
            'Z': (1, 0, 0, -1),
            'H': (1/math.sqrt(2), 1/math.sqrt(2), 1/math.sqrt(2), -1/math.sqrt(2)),
            'S': (1, 0, 0, 1j),
            'T': (1, 0, 0, complex(math.cos(math.pi/4), math.sin(math.pi/4)))
        }
        if gate_name in gates:
            k = gates[gate_name]
            args = [self.Complex(x.real, x.imag) for x in k]
            if control is not None:
                self.lib.apply_controlled_gate(self.sys_ptr, control, target, *args)
            else:
                self.lib.apply_gate(self.sys_ptr, target, *args)
                
    def get_probabilities(self):
        if not self.lib:
            size = 1 << self.num_qubits
            return [1.0/size] * size
        size = 1 << self.num_qubits
        return [self.lib.get_probability(self.sys_ptr, i) for i in range(size)]

class CircuitCanvas(tk.Canvas):
    def __init__(self, parent, num_qubits=4, max_steps=12, **kwargs):
        super().__init__(parent, bg='#1a1a1a', highlightthickness=0, **kwargs)
        self.num_qubits = num_qubits
        self.max_steps = max_steps
        self.qubit_spacing = 60
        self.step_width = 80
        self.padding = 40
        self.gates = {}
        self.selected_gate = 'H'
        self.animating = False
        
        self.bind("<Configure>", lambda e: self._draw_grid())
        self.bind("<Button-1>", self._on_click)
        
    def _draw_grid(self):
        self.delete("grid")
        width = self.winfo_width()
        height = self.winfo_height()
        
        for q in range(self.num_qubits):
            y = self.padding + q * self.qubit_spacing
            self.create_line(self.padding, y, width - self.padding, y, fill="#444", width=2, tags="grid")
            self.create_text(20, y, text=f"|q{q}⟩", fill="#fff", font=("Consolas", 12), tags="grid")
            
        for t in range(self.max_steps):
            x = self.padding + t * self.step_width + self.step_width//2
            self.create_line(x, self.padding - 20, x, height - self.padding, fill="#222", width=1, tags="grid")
            
    def _on_click(self, event):
        if self.animating:
            return
        t = int((event.x - self.padding) // self.step_width)
        q = int((event.y - self.padding + self.qubit_spacing//2) // self.qubit_spacing)
        if 0 <= t < self.max_steps and 0 <= q < self.num_qubits:
            self.toggle_gate(q, t)
            
    def toggle_gate(self, q, t, force_gate=None):
        key = (q, t)
        if force_gate:
            self.gates[key] = force_gate
        elif key in self.gates:
            del self.gates[key]
        else:
            self.gates[key] = self.selected_gate
        self._draw_circuit()
        
    def _draw_circuit(self):
        self.delete("gate")
        
        cnot_cols = {}
        for (q, t), gate in self.gates.items():
            if gate == 'CNOT':
                cnot_cols.setdefault(t, []).append(q)
                
        for t, qubits in cnot_cols.items():
            if len(qubits) == 2:
                x = self.padding + t * self.step_width + self.step_width//2
                y1 = self.padding + min(qubits) * self.qubit_spacing
                y2 = self.padding + max(qubits) * self.qubit_spacing
                self.create_line(x, y1, x, y2, fill="#0ff", width=3, tags="gate")
                
        for (q, t), gate in self.gates.items():
            x = self.padding + t * self.step_width + self.step_width//2
            y = self.padding + q * self.qubit_spacing
            colors = {'H': '#ff6b6b', 'X': '#4ecdc4', 'Y': '#ffe66d', 'Z': '#95e1d3', 
                     'S': '#f38181', 'T': '#aa96da', 'CNOT': '#0ff', 'M': '#fff'}
            color = colors.get(gate, '#fff')
            
            if gate == 'CNOT':
                is_control = t in cnot_cols and q == min(cnot_cols[t])
                if is_control:
                    self.create_oval(x-8, y-8, x+8, y+8, fill=color, outline="", tags="gate")
                else:
                    self.create_oval(x-12, y-12, x+12, y+12, fill="#1a1a1a", outline=color, width=3, tags="gate")
                    self.create_line(x-8, y, x+8, y, fill=color, width=2, tags="gate")
                    self.create_line(x, y-8, x, y+8, fill=color, width=2, tags="gate")
            elif gate == 'M':
                self.create_rectangle(x-20, y-15, x+20, y+15, fill="#333", outline="#fff", tags="gate")
                self.create_text(x, y, text="⚡", fill="#fff", font=("Arial", 14), tags="gate")
            else:
                self.create_rectangle(x-20, y-15, x+20, y+15, fill=color, outline="#fff", width=2, tags="gate")
                self.create_text(x, y, text=gate, fill="#fff", font=("Helvetica", 14, "bold"), tags="gate")
                
    def highlight_step(self, t):
        self.delete("highlight")
        if 0 <= t < self.max_steps:
            x1 = self.padding + t * self.step_width + 5
            x2 = x1 + self.step_width - 10
            y1 = self.padding - 30
            y2 = self.padding + self.num_qubits * self.qubit_spacing - 10
            self.create_rectangle(x1, y1, x2, y2, outline="#0f0", width=3, tags="highlight")
            
    def load_preset(self, name):
        self.gates.clear()
        presets = {
            "Bell State": [(0, 0, 'H'), (0, 1, 'CNOT'), (1, 1, 'CNOT')],
            "GHZ": [(0, 0, 'H'), (0, 1, 'CNOT'), (1, 1, 'CNOT'), (0, 2, 'CNOT'), (2, 2, 'CNOT')],
            "QFT": [(0, 0, 'H'), (1, 1, 'S'), (0, 1, 'CNOT'), (1, 2, 'H'), (2, 2, 'T'), (0, 2, 'CNOT')],
            "Grover": [(0, 0, 'H'), (1, 0, 'H'), (0, 1, 'Z'), (0, 2, 'CNOT'), (2, 2, 'CNOT'), (0, 3, 'H')]
        }
        if name in presets:
            for q, t, g in presets[name]:
                self.gates[(q, t)] = g
        self._draw_circuit()

class QuantumGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PhonexQ Architect")
        self.root.geometry("1400x900")
        self.root.configure(bg='#0d0d0d')
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure(".", background="#0d0d0d", foreground="#fff")
        self.style.configure("TButton", background="#2d2d2d", foreground="#fff", padding=10)
        self.style.map("TButton", background=[('active', '#3d3d3d')])
        
        self.num_qubits = 4
        self.current_step = 0
        self.running = False
        
        self.engine = PhonexQEngine(self.num_qubits)
        self._build_ui()
        
    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        toolbar = ttk.Frame(main)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(toolbar, text="⚛ PhonexQ", font=("Helvetica", 20, "bold")).pack(side=tk.LEFT)
        
        gate_frame = ttk.Frame(toolbar)
        gate_frame.pack(side=tk.LEFT, padx=20)
        
        self.gate_var = tk.StringVar(value='H')
        for g in ['H', 'X', 'Y', 'Z', 'S', 'T', 'CNOT', 'M']:
            ttk.Radiobutton(gate_frame, text=g, variable=self.gate_var, value=g,
                          command=self._update_gate).pack(side=tk.LEFT)
            
        preset_frame = ttk.Frame(toolbar)
        preset_frame.pack(side=tk.LEFT, padx=10)
        self.preset_var = tk.StringVar()
        presets = ttk.Combobox(preset_frame, textvariable=self.preset_var, 
                              values=['Bell State', 'GHZ', 'QFT', 'Grover'], width=12)
        presets.pack(side=tk.LEFT)
        presets.bind('<<ComboboxSelected>>', lambda e: self.canvas.load_preset(self.preset_var.get()))
        
        ttk.Button(toolbar, text="▶ Run", command=self.run).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="⏹ Stop", command=self.stop).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="⏮ Clear", command=self.clear).pack(side=tk.LEFT, padx=5)
        
        content = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True)
        
        left = ttk.Frame(content)
        content.add(left, weight=2)
        
        self.canvas = CircuitCanvas(left, num_qubits=self.num_qubits, max_steps=12)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        right = ttk.Notebook(content)
        content.add(right, weight=1)
        
        self._create_plot_tab(right, "Probabilities")
        self._create_plot_tab(right, "State Vector")
        
        self.fig, self.ax = plt.subplots(figsize=(6, 4), facecolor='#1a1a1a')
        self.ax.set_facecolor('#1a1a1a')
        self.ax.tick_params(colors='white')
        for spine in self.ax.spines.values():
            spine.set_color('white')
            
        self.plot_canvas = FigureCanvasTkAgg(self.fig, right.winfo_children()[0])
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def _create_plot_tab(self, notebook, title):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=title)
        return frame
        
    def _update_gate(self):
        self.canvas.selected_gate = self.gate_var.get()
        
    def run(self):
        if self.running:
            return
        self.running = True
        self.canvas.animating = True
        self.current_step = 0
        self.engine.reset()
        self._step()
        
    def _step(self):
        if not self.running or self.current_step >= self.canvas.max_steps:
            self.running = False
            self.canvas.animating = False
            self.canvas.delete("highlight")
            return
            
        self.canvas.highlight_step(self.current_step)
        
        for q in range(self.num_qubits):
            gate = self.canvas.gates.get((q, self.current_step))
            if gate and gate != 'CNOT':
                self.engine.apply_gate(gate if gate != 'M' else 'Z', q)
                
        for t, gate in self.canvas.gates.items():
            if t[1] == self.current_step and gate == 'CNOT':
                for q2 in range(self.num_qubits):
                    if (q2, self.current_step) in self.canvas.gates and q2 != t[0]:
                        self.engine.apply_gate('X', max(t[0], q2), control=min(t[0], q2))
                        break
                        
        self._update_plot()
        self.current_step += 1
        self.root.after(300, self._step)
        
    def _update_plot(self):
        probs = self.engine.get_probabilities()
        states = [format(i, f'0{self.num_qubits}b') for i in range(len(probs))]
        
        self.ax.clear()
        self.ax.set_facecolor('#1a1a1a')
        self.ax.tick_params(colors='white')
        for spine in self.ax.spines.values():
            spine.set_color('white')
            
        colors = plt.cm.viridis(np.array(probs) / (max(probs) or 1))
        self.ax.bar(states, probs, color=colors)
        self.ax.set_ylim(0, 1.1)
        self.ax.set_title(f'Step {self.current_step}', color='white')
        self.ax.set_xlabel('State', color='white')
        self.ax.set_ylabel('Probability', color='white')
        plt.setp(self.ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        self.plot_canvas.draw()
        
    def stop(self):
        self.running = False
        self.canvas.animating = False
        
    def clear(self):
        self.stop()
        self.current_step = 0
        self.canvas.gates.clear()
        self.canvas.delete("gate", "highlight")
        self.engine.reset()
        self.ax.clear()
        self.plot_canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = QuantumGUI(root)
    root.mainloop()