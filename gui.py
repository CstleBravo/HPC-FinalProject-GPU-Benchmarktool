import tkinter as tk
from tkinter import ttk, scrolledtext

import gpubenchmark as bench

def run_selected_benchmarks():
    output_box.delete(1.0, tk.END)  # Clear previous output
    
    for line in bench.get_device_info():
        output_box.insert(tk.END, line + "\n")
    output_box.insert(tk.END, "\n")
    
    selected = benchmark_var.get()
    if selected == 'Vector Addition':
        results = bench.run_vector_benchmark()
    elif selected == 'Matrix Multiplication':
        results = bench.run_matmul_benchmark()
    elif selected == 'Conv2D':
        results = bench.run_conv2d_benchmark()
    elif selected == 'Device Comparison':
        results = bench.compare_devices(1024)
    else:
        results = bench.unified_benchmark()
    
    for line in results:
        output_box.insert(tk.END, line + '\n')


def clear_output():
    output_box.delete(1.0, tk.END)
    

def build_gui():
    global benchmark_var, output_box

    root = tk.Tk()
    root.title('GPU Benchmarking')
    root.geometry('800x600')

    top_frame = ttk.Frame(root)
    top_frame.pack(pady=10)

    title_label = ttk.Label(top_frame, text='GPU Benchmarking', font=('Arial', 16, 'bold'))
    title_label.pack()

    control_frame = ttk.Frame(root)
    control_frame.pack(pady=10)

    tk.Label(control_frame, text='Select Benchmark:').pack(side=tk.LEFT, padx=5)
    benchmark_var = tk.StringVar(value='Unified Benchmark')
    task_menu = ttk.Combobox(control_frame, textvariable=benchmark_var, state='readonly')
    task_menu['values'] = ('Vector Addition', 'Matrix Multiplication', 'Conv2D', 'Device Comparison', 'Unified Benchmark')
    task_menu.current(4)
    task_menu.pack(side=tk.LEFT, padx=5)

    run_button = ttk.Button(control_frame, text='Run Benchmark', command=run_selected_benchmarks)
    run_button.pack(side=tk.LEFT, padx=5)
    clear_button = ttk.Button(control_frame, text='Clear Output', command=clear_output)
    clear_button.pack(side=tk.LEFT, padx=5)

    output_box = scrolledtext.ScrolledText(root, width=100, height=30, font=('Courier', 10))
    output_box.pack(padx=10, pady=10)

    return root


if __name__ == '__main__':
    root = build_gui()
    root.mainloop()
