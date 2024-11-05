import networkx as nx
import numpy as np
import heapq
from collections import defaultdict
import time
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Función para generar un grafo a partir de las conexiones ingresadas
def create_graph_from_edges(num_nodes, edges):
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    for edge in edges:
        u, v, weight = edge
        graph.add_edge(u, v, weight=weight)
    return graph

# Función para recoger las conexiones y crear el grafo
def get_edges_from_input(num_nodes, entry_widgets):
    edges = []
    for i in range(num_nodes * (num_nodes - 1) // 2):
        u = int(entry_widgets[i][0].get())
        v = int(entry_widgets[i][1].get())
        weight = float(entry_widgets[i][2].get())
        edges.append((u, v, weight))
    return edges

# Heurísticas
def mst_heuristic(graph, current_node, unvisited_nodes):
    subgraph = graph.subgraph(unvisited_nodes)
    mst = nx.minimum_spanning_tree(subgraph, algorithm="prim")
    mst_cost = mst.size(weight='weight')
    min_edge_cost = min(
        graph[current_node][n]["weight"] for n in unvisited_nodes if graph.has_edge(current_node, n)
    )
    return mst_cost + min_edge_cost

def initialize_pheromone_map(graph, initial_pheromone=1.0):
    pheromone_map = defaultdict(lambda: defaultdict(lambda: initial_pheromone))
    return pheromone_map

def ant_colony_heuristic(graph, current_node, unvisited_nodes, pheromone_map, alpha=1.0, beta=1.0):
    total_pheromone_cost = 0
    for node in unvisited_nodes:
        pheromone = pheromone_map[current_node][node]
        distance = graph[current_node][node]['weight']
        total_pheromone_cost += (pheromone ** alpha) * ((1.0 / distance) ** beta)
    return total_pheromone_cost / len(unvisited_nodes)

def combined_heuristic(graph, current_node, unvisited_nodes, pheromone_map, alpha=0.5):
    mst_cost = mst_heuristic(graph, current_node, unvisited_nodes)
    aco_cost = ant_colony_heuristic(graph, current_node, unvisited_nodes, pheromone_map)
    return alpha * mst_cost + (1 - alpha) * aco_cost

def astar_tsp(graph, start_node, heuristic_func, pheromone_map=None):
    num_nodes = graph.number_of_nodes()
    frontier = []
    heapq.heappush(frontier, (0, start_node, [start_node], 0))
    best_cost = float('inf')
    best_path = None

    while frontier:
        estimated_cost, current_node, path, path_cost = heapq.heappop(frontier)
        
        if len(path) == num_nodes:
            total_cost = path_cost + graph[current_node][start_node]["weight"]
            if total_cost < best_cost:
                best_cost = total_cost
                best_path = path + [start_node]
            continue

        unvisited_nodes = set(graph.nodes) - set(path)

        if heuristic_func == mst_heuristic:
            heuristic_cost = mst_heuristic(graph, current_node, unvisited_nodes)
        elif heuristic_func == ant_colony_heuristic:
            heuristic_cost = ant_colony_heuristic(graph, current_node, unvisited_nodes, pheromone_map)
        else:
            heuristic_cost = combined_heuristic(graph, current_node, unvisited_nodes, pheromone_map)

        for neighbor in unvisited_nodes:
            edge_cost = graph[current_node][neighbor]["weight"]
            new_path_cost = path_cost + edge_cost
            new_estimated_cost = new_path_cost + heuristic_cost
            heapq.heappush(frontier, (new_estimated_cost, neighbor, path + [neighbor], new_path_cost))

    return best_path, best_cost

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

# Función para mostrar la interfaz de entrada de bordes y ejecutar el TSP
def show_edge_input_interface(root, num_nodes):
    edges_window = Toplevel(root)
    edges_window.title("Ingresar Conexiones")

    Label(edges_window, text="Ingrese cada conexión (nodo1, nodo2, peso):").grid(row=0, column=0, columnspan=3)

    entry_widgets = []
    for i in range(num_nodes * (num_nodes - 1) // 2):
        entry1 = Entry(edges_window, width=5)
        entry2 = Entry(edges_window, width=5)
        weight_entry = Entry(edges_window, width=5)
        entry1.grid(row=i+1, column=0)
        entry2.grid(row=i+1, column=1)
        weight_entry.grid(row=i+1, column=2)
        entry_widgets.append((entry1, entry2, weight_entry))

    Button(edges_window, text="Ejecutar TSP", command=lambda: execute_tsp_from_input(edges_window, root, num_nodes, entry_widgets)).grid(row=num_nodes+1, column=0, columnspan=3)

def execute_tsp_from_input(edges_window, root, num_nodes, entry_widgets):
    edges = get_edges_from_input(num_nodes, entry_widgets)
    graph = create_graph_from_edges(num_nodes, edges)
    edges_window.destroy()
    display_results(root, graph)

def display_results(root, graph):
    start_node = 0

    result_window = Toplevel(root)
    result_window.title("Resultados TSP")

    Label(result_window, text="Resultados del TSP con el grafo ingresado").pack()

    text_area = Text(result_window, width=50, height=15)
    text_area.pack()
    
    results = run_tsp_and_display(graph, start_node, text_area)

    # Mostrar grafo generado
    fig, ax = plt.subplots(figsize=(5, 5))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, ax=ax, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    canvas = FigureCanvasTkAgg(fig, master=result_window)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Gráfica de barras para resultados
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

    Label(root, text="Select Number of Nodes:").pack()
    num_nodes = IntVar(value=5)
    node_entry = ttk.Entry(root, textvariable=num_nodes)
    node_entry.pack()

    Button(root, text="Input Connections", command=lambda: show_edge_input_interface(root, num_nodes.get())).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
