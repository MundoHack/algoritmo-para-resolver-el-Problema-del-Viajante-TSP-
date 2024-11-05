import networkx as nx
import heapq
from collections import defaultdict
import time
from tkinter import *
from tkinter import Toplevel, Label
from itertools import permutations
from itertools import combinations
import matplotlib.pyplot as plt

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
            return None

    return edges

def calculate_tour_cost(graph, tour):
    total_cost = 0
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]  # Volver al nodo inicial al final
        if graph.has_edge(u, v):
            total_cost += graph[u][v]['weight']
        else:
            return float('inf')  # Ruta no válida
    return total_cost

def find_best_tour(graph, node_names):
    min_cost = float('inf')
    best_tour = None
    
    for tour in permutations(node_names):
        cost = calculate_tour_cost(graph, tour)
        if cost < min_cost:
            min_cost = cost
            best_tour = tour
    
    return best_tour, min_cost

def greedy_tsp(graph, start_node):
    visited = {start_node}
    tour = [start_node]
    current_node = start_node

    while len(visited) < len(graph.nodes):
        neighbors = {node: graph[current_node][node]['weight'] for node in graph.neighbors(current_node) if node not in visited}
        if not neighbors:
            break
        next_node = min(neighbors, key=neighbors.get)
        visited.add(next_node)
        tour.append(next_node)
        current_node = next_node

    # Regresar al nodo inicial
    tour.append(start_node)
    return tour, calculate_tour_cost(graph, tour)

def plot_graph(graph, path, title):
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(8, 5))
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold')
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    
    if path:
        path_edges = list(zip(list(path), list(path[1:]) + [path[0]]))  # Convertir a lista
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='r', width=3)
    
    plt.title(title)
    plt.show()

def run_tsp_and_display(graph, start_node, display_area):
    # Búsqueda exhaustiva
    best_tour, best_cost = find_best_tour(graph, list(graph.nodes))
    result_text = f"Búsqueda Exhaustiva - Mejor ruta: {' -> '.join(best_tour)}, Costo: {best_cost}"
    display_area.insert(END, result_text + "\n")
    plot_graph(graph, best_tour, "Mejor Ruta TSP - Búsqueda Exhaustiva")

    # Heurística Greedy
    greedy_tour, greedy_cost = greedy_tsp(graph, start_node)
    result_text = f"Heurística Greedy - Mejor ruta: {' -> '.join(greedy_tour)}, Costo: {greedy_cost}"
    display_area.insert(END, result_text + "\n")
    plot_graph(graph, greedy_tour, "Ruta TSP - Heurística Greedy")

def show_edge_input_interface(root):
    edges_window = Toplevel(root)
    edges_window.title("Ingresar Conexiones")

    Label(edges_window, text="Ingrese la cantidad de nodos:").grid(row=0, column=0, padx=10, pady=5)
    node_count_entry = Entry(edges_window)
    node_count_entry.grid(row=0, column=1, padx=10, pady=5)

    def create_node_entries():
        try:
            node_count = int(node_count_entry.get())
            if node_count <= 0:
                raise ValueError("El número de nodos debe ser mayor que 0.")
            
            node_names_frame = Frame(edges_window)
            node_names_frame.grid(row=1, column=0, padx=10, pady=10)

            Label(node_names_frame, text="Ingrese nombres para los nodos:").grid(row=0, column=0, columnspan=2)

            node_entries = []
            for i in range(node_count):
                entry = Entry(node_names_frame, width=15)
                entry.grid(row=i + 1, column=0, padx=5, pady=5)
                node_entries.append(entry)

            Button(node_names_frame, text="Siguiente", command=lambda: create_connection_entries(node_entries)).grid(row=node_count + 1, column=0, pady=10)

        except ValueError:
            print("Por favor, introduce un número válido de nodos.")

    def create_connection_entries(node_entries):
        node_names = [entry.get().strip() for entry in node_entries]

        if any(name == "" for name in node_names):
            print("Todos los nombres de los nodos deben ser completados.")
            return
        
        connection_frame = Frame(edges_window)
        connection_frame.grid(row=1, column=1, padx=20, pady=10, sticky='n')

        Label(connection_frame, text="Ingrese cada conexión (nodo1, nodo2, peso):").grid(row=0, column=0, columnspan=3)

        entry_widgets = []
        for i, (node1, node2) in enumerate(combinations(node_names, 2)):
            frame = Frame(connection_frame)
            frame.grid(row=i + 1, column=0, sticky='w')
            entry1 = Entry(frame, width=10)
            entry2 = Entry(frame, width=10)
            weight_entry = Entry(frame, width=5)
            entry1.insert(0, node1)
            entry2.insert(0, node2)
            entry1.config(state='readonly')
            entry2.config(state='readonly')
            entry1.grid(row=0, column=0)
            entry2.grid(row=0, column=1)
            weight_entry.grid(row=0, column=2)
            entry_widgets.append((entry1, entry2, weight_entry))

        Button(connection_frame, text="Ejecutar TSP", command=lambda: execute_tsp_from_input(edges_window, root, entry_widgets, node_names)).grid(row=len(entry_widgets) + 1, column=0, pady=10)

    node_count_entry.bind("<Return>", lambda event: create_node_entries())
    node_count_entry.grid(row=0, column=1, padx=10, pady=5)

def execute_tsp_from_input(edges_window, root, entry_widgets, node_names):
    edges = get_edges_from_input(entry_widgets, node_names)
    if edges is None:
        return

    graph = create_graph_from_edges(node_names, edges)

    display_window = Toplevel(root)
    display_window.title("Resultados del TSP")
    display_area = Text(display_window, width=60, height=20)
    display_area.pack(pady=10)

    run_tsp_and_display(graph, node_names[0], display_area)

def main():
    root = Tk()
    root.title("Problema del Vendedor Viajero")
    Button(root, text="Ingresar Conexiones", command=lambda: show_edge_input_interface(root)).pack(pady=10)
    root.mainloop()

if __name__ == "__main__":
    main()
