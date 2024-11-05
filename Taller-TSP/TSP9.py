import networkx as nx
import numpy as np
import heapq
from collections import defaultdict
import time
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def create_graph_from_edges(node_names, edges):
    graph = nx.Graph()
    graph.add_nodes_from(node_names)
    for edge in edges:
        u, v, weight = edge
        graph.add_edge(u, v, weight=weight)
    return graph

def get_edges_from_input(entry_widgets):
    edges = []
    for i, (entry1, entry2, weight_entry) in enumerate(entry_widgets):
        try:
            u = entry1.get()
            v = entry2.get()
            cost = float(weight_entry.get())
            edges.append((u, v, cost))
        except ValueError:
            print(f"Error: Asegúrate de que todas las entradas sean válidas. Entrada inválida en la fila {i + 1}.")
            return None  # Detenemos el proceso si hay un error
    return edges

def initialize_pheromone_map(graph, initial_pheromone=1.0):
    pheromone_map = defaultdict(lambda: defaultdict(lambda: initial_pheromone))
    return pheromone_map

# (Funciones de heurísticas y algoritmo de búsqueda de A* permanecen iguales)

def run_tsp_and_display(graph, start_node, display_area):
    pheromone_map = initialize_pheromone_map(graph)
    heuristics = {
        "MST": mst_heuristic,
        "ACO": ant_colony_heuristic,
        "Combined": combined_heuristic
    }
    results = []

    for heuristic_name, heuristic_func in heuristics.items():
        start_time = time.time()
        path, cost = astar_tsp(graph, start_node, heuristic_func, pheromone_map)
        elapsed_time = time.time() - start_time
        results.append((heuristic_name, cost, elapsed_time))

        display_area.insert(END, f"{heuristic_name} Heuristic - Cost: {cost}, Time: {elapsed_time:.4f}s\n")

    display_area.insert(END, "\n")
    return results

def show_edge_input_interface(root, num_nodes):
    edges_window = Toplevel(root)
    edges_window.title("Ingresar Conexiones")

    Label(edges_window, text="Ingrese cada conexión (nodo1, nodo2, peso):").pack()

    entry_widgets = []
    for i in range(num_nodes * (num_nodes - 1) // 2):
        frame = Frame(edges_window)
        frame.pack()
        entry1 = Entry(frame, width=10)
        entry2 = Entry(frame, width=10)
        weight_entry = Entry(frame, width=5)
        entry1.grid(row=0, column=0)
        entry2.grid(row=0, column=1)
        weight_entry.grid(row=0, column=2)
        entry_widgets.append((entry1, entry2, weight_entry))

    Button(edges_window, text="Ejecutar TSP", command=lambda: execute_tsp_from_input(edges_window, root, entry_widgets)).pack(pady=10)

def execute_tsp_from_input(edges_window, root, entry_widgets):
    edges = get_edges_from_input(entry_widgets)
    if edges is None:
        return

    # Obtiene los nombres únicos de los nodos a partir de los datos ingresados
    node_names = list(set([edge[0] for edge in edges] + [edge[1] for edge in edges]))
    graph = create_graph_from_edges(node_names, edges)
    edges_window.destroy()
    display_results(root, graph, node_names[0])

def display_results(root, graph, start_node):
    result_window = Toplevel(root)
    result_window.title("Resultados TSP")

    Label(result_window, text="Resultados del TSP con el grafo ingresado").pack()

    text_area = Text(result_window, width=50, height=15)
    text_area.pack()
    
    results = run_tsp_and_display(graph, start_node, text_area)

    fig, ax = plt.subplots(figsize=(5, 5))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, ax=ax, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    canvas = FigureCanvasTkAgg(fig, master=result_window)
    canvas.draw()
    canvas.get_tk_widget().pack()

    fig, ax = plt.subplots()
    heuristics, costs, times = zip(*results)
    x = np.arange(len(heuristics))
    width = 0.35
    ax.bar(x - width/2, costs, width, label='Cost')
    ax.bar(x + width/2, times, width, label='Time')
    ax.set_xlabel("Heuristic")
    ax.set_title("Comparison of Costs and Times")
    ax.set_xticks(x)
    ax.set_xticklabels(heuristics)
    ax.legend()

    canvas_bar = FigureCanvasTkAgg(fig, master=result_window)
    canvas_bar.draw()
    canvas_bar.get_tk_widget().pack()

def main():
    root = Tk()
    root.title("TSP Heuristic Comparison")

    Label(root, text="TSP Heuristic Comparison Tool", font=("Helvetica", 16)).pack(pady=10)

    Label(root, text="Seleccione el número de nodos:").pack()
    num_nodes = IntVar(value=5)
    node_entry = ttk.Entry(root, textvariable=num_nodes)
    node_entry.pack()

    Button(root, text="Ingresar Conexiones", command=lambda: show_edge_input_interface(root, num_nodes.get())).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
