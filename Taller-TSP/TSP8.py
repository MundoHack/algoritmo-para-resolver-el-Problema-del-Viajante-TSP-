import networkx as nx
import numpy as np
import heapq
from collections import defaultdict
import time
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def create_graph_from_edges(num_nodes, edges):
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    for edge in edges:
        u, v, weight = edge
        graph.add_edge(u, v, weight=weight)
    return graph

def get_edges_from_input(num_nodes, entry_widgets):
    edges = []
    for i, (entry1, entry2, weight_entry) in enumerate(entry_widgets):
        try:
            u = int(entry1.get())
            v = int(entry2.get())
            cost = float(weight_entry.get())
            edges.append((u, v, cost))
        except ValueError:
            print(f"Error: Asegúrate de que todas las entradas sean numéricas. Entrada inválida en la fila {i + 1}.")
            return None  # Detenemos el proceso si hay un error
    return edges

def mst_heuristic(graph, current_node, unvisited_nodes):
    subgraph = graph.subgraph(unvisited_nodes)
    mst = nx.minimum_spanning_tree(subgraph, algorithm="prim")
    mst_cost = mst.size(weight='weight')

    min_edge_costs = [
        graph[current_node][n]["weight"] for n in unvisited_nodes if graph.has_edge(current_node, n)
    ]
    min_edge_cost = min(min_edge_costs) if min_edge_costs else float('inf')
    return mst_cost + min_edge_cost

def initialize_pheromone_map(graph, initial_pheromone=1.0):
    pheromone_map = defaultdict(lambda: defaultdict(lambda: initial_pheromone))
    return pheromone_map

def ant_colony_heuristic(graph, current_node, unvisited_nodes, pheromone_map):
    heuristic_cost = 0
    for node in unvisited_nodes:
        # Verificar si existe la arista antes de acceder al peso
        if graph.has_edge(current_node, node):
            distance = graph[current_node][node]['weight']
            pheromone = pheromone_map.get((current_node, node), 1.0)
            heuristic_cost += distance / pheromone
        else:
            # Si la arista no existe, se puede asignar un costo alto o ignorarla
            heuristic_cost += float('inf')  # Representa una distancia muy alta
    return heuristic_cost


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
            # Verificar si existe la arista para evitar KeyError
            if graph.has_edge(current_node, start_node):
                total_cost = path_cost + graph[current_node][start_node]["weight"]
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_path = path + [start_node]
            continue

        unvisited_nodes = set(graph.nodes) - set(path)

        # Elegir la heurística adecuada
        if heuristic_func == mst_heuristic:
            heuristic_cost = mst_heuristic(graph, current_node, unvisited_nodes)
        elif heuristic_func == ant_colony_heuristic:
            heuristic_cost = ant_colony_heuristic(graph, current_node, unvisited_nodes, pheromone_map)
        else:
            heuristic_cost = combined_heuristic(graph, current_node, unvisited_nodes, pheromone_map)

        for neighbor in unvisited_nodes:
            # Verificar si existe la arista para evitar KeyError
            if graph.has_edge(current_node, neighbor):
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

def show_edge_input_interface(root, num_nodes):
    edges_window = Toplevel(root)
    edges_window.title("Ingresar Conexiones")

    Label(edges_window, text="Ingrese cada conexión (nodo1, nodo2, peso):").pack()

    entry_widgets = []
    for i in range(num_nodes * (num_nodes - 1) // 2):
        frame = Frame(edges_window)
        frame.pack()
        entry1 = Entry(frame, width=5)
        entry2 = Entry(frame, width=5)
        weight_entry = Entry(frame, width=5)
        entry1.grid(row=0, column=0)
        entry2.grid(row=0, column=1)
        weight_entry.grid(row=0, column=2)
        entry_widgets.append((entry1, entry2, weight_entry))

    Button(edges_window, text="Ejecutar TSP", command=lambda: execute_tsp_from_input(edges_window, root, num_nodes, entry_widgets)).pack(pady=10)

def execute_tsp_from_input(edges_window, root, num_nodes, entry_widgets):
    edges = get_edges_from_input(num_nodes, entry_widgets)
    if edges is None:
        return  # Se detiene si hay un error en los datos de entrada
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

    Label(root, text="Select Number of Nodes:").pack()
    num_nodes = IntVar(value=5)
    node_entry = ttk.Entry(root, textvariable=num_nodes)
    node_entry.pack()

    Button(root, text="Input Connections", command=lambda: show_edge_input_interface(root, num_nodes.get())).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
