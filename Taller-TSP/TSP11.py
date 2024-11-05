import networkx as nx
import numpy as np
import heapq
from collections import defaultdict
import time
import matplotlib.pyplot as plt
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def create_graph_from_edges(node_names, edges):
    graph = nx.Graph()
    for node in node_names:
        graph.add_node(node)

    for edge in edges:
        u, v, weight = edge
        graph.add_edge(u, v, weight=weight)

    return graph

def get_edges_from_input(entry_widgets, node_names):
    edges = []
    for i, (entry1, entry2, weight_entry) in enumerate(entry_widgets):
        try:
            u = entry1.get().strip()
            v = entry2.get().strip()
            cost = float(weight_entry.get())
            if u and v and u in node_names and v in node_names:
                edges.append((u, v, cost))
        except ValueError:
            print(f"Error: Asegúrate de que todas las entradas sean válidas. Entrada inválida en la fila {i + 1}.")
            return None, None  # Detenemos el proceso si hay un error

    return edges  # Devolver lista de conexiones

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
        if graph.has_edge(current_node, node):
            distance = graph[current_node][node]['weight']
            pheromone = pheromone_map.get((current_node, node), 1.0)
            heuristic_cost += distance / pheromone
        else:
            heuristic_cost += float('inf')
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
            if graph.has_edge(current_node, path[0]):
                total_cost = path_cost + graph[current_node][path[0]]["weight"]
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_path = path + [path[0]]
            continue

        unvisited_nodes = set(graph.nodes) - set(path)

        if heuristic_func == mst_heuristic:
            heuristic_cost = mst_heuristic(graph, current_node, unvisited_nodes)
        elif heuristic_func == ant_colony_heuristic:
            heuristic_cost = ant_colony_heuristic(graph, current_node, unvisited_nodes, pheromone_map)
        else:
            heuristic_cost = combined_heuristic(graph, current_node, unvisited_nodes, pheromone_map)

        for neighbor in unvisited_nodes:
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

def show_edge_input_interface(root):
    edges_window = Toplevel(root)
    edges_window.title("Ingresar Conexiones")

    Label(edges_window, text="Ingrese la cantidad de nodos:").pack()
    node_count_entry = Entry(edges_window)
    node_count_entry.pack()

    def create_node_entries():
        try:
            node_count = int(node_count_entry.get())
            if node_count <= 0:
                raise ValueError("El número de nodos debe ser mayor que 0.")
            
            # Crear entradas para los nombres de los nodos
            node_names_frame = Frame(edges_window)
            node_names_frame.pack(pady=10)

            Label(node_names_frame, text="Ingrese nombres para los nodos:").grid(row=0, column=0, columnspan=2)

            node_entries = []
            for i in range(node_count):
                entry = Entry(node_names_frame, width=15)
                entry.grid(row=i + 1, column=0, padx=5)
                node_entries.append(entry)

            # Botón para habilitar entradas de conexión después de ingresar nombres
            Button(node_names_frame, text="Siguiente", command=lambda: create_connection_entries(node_entries)).grid(row=node_count + 1, column=0, pady=10)

        except ValueError:
            print("Por favor, introduce un número válido de nodos.")

    def create_connection_entries(node_entries):
        node_names = [entry.get().strip() for entry in node_entries]

        if any(name == "" for name in node_names):
            print("Todos los nombres de los nodos deben ser completados.")
            return
        
        # Crear entradas para las conexiones
        connection_frame = Frame(edges_window)
        connection_frame.pack(pady=10)

        Label(connection_frame, text="Ingrese cada conexión (nodo1, nodo2, peso):").pack()

        entry_widgets = []
        for i in range(10):  # Hasta 10 conexiones; puedes ajustar según tus necesidades
            frame = Frame(connection_frame)
            frame.pack()
            entry1 = Entry(frame, width=10)  # Nodo 1 (nombre)
            entry2 = Entry(frame, width=10)  # Nodo 2 (nombre)
            weight_entry = Entry(frame, width=5)  # Peso
            entry1.grid(row=0, column=0)
            entry2.grid(row=0, column=1)
            weight_entry.grid(row=0, column=2)
            entry_widgets.append((entry1, entry2, weight_entry))

        Button(connection_frame, text="Ejecutar TSP", command=lambda: execute_tsp_from_input(edges_window, root, entry_widgets, node_names)).pack(pady=10)

    node_count_entry.bind('<Return>', lambda event: create_node_entries())
    node_count_entry.pack(pady=10)

def execute_tsp_from_input(edges_window, root, entry_widgets, node_names):
    edges = get_edges_from_input(entry_widgets, node_names)
    if edges is None:
        return
    graph = create_graph_from_edges(node_names, edges)
    edges_window.destroy()
    display_results(root, graph)

def display_results(root, graph):
    start_node = list(graph.nodes)[0]  # Tomar el primer nodo como nodo inicial

    result_window = Toplevel(root)
    result_window.title("Resultados TSP")

    Label(result_window, text="Resultados del TSP con el grafo ingresado").pack()

    text_area = Text(result_window, width=50, height=15)
    text_area.pack()
    
    results = run_tsp_and_display(graph, start_node, text_area)

    fig, ax = plt.subplots(figsize=(5, 5))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, ax=ax, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, 'weight'), ax=ax)
    
    canvas = FigureCanvasTkAgg(fig, master=result_window)
    canvas.draw()
    canvas.get_tk_widget().pack()

def main():
    root = Tk()
    root.title("TSP con Algoritmos Heurísticos")

    Label(root, text="Algoritmo de Viaje del Viajante").pack(pady=10)
    Button(root, text="Ingresar Conexiones", command=lambda: show_edge_input_interface(root)).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
