import json
import networkx as nx
import heapq
from collections import defaultdict
import time
from tkinter import *
from tkinter import Toplevel, Label, Scrollbar, Frame, filedialog, messagebox
from itertools import combinations
import matplotlib.pyplot as plt

# Funciones para manejar el archivo JSON
def save_connections_to_file(node_names, edges):
    data = {"nodes": node_names, "edges": edges}
    file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
    if file_path:
        with open(file_path, "w") as f:
            json.dump(data, f)
        messagebox.showinfo("Guardar Conexiones", f"Conexiones guardadas en {file_path}")

def load_connections_from_file():
    file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
    if file_path:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data["nodes"], data["edges"]
    return None, None

# Funciones previas del código (sin cambios)
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
            print(f"Error: Entrada inválida en la fila {i + 1}.")
            return None
    return edges

# Implementación de heurísticas y TSP (sin cambios)
def mst_heuristic(graph, current_node, unvisited_nodes):
    subgraph = graph.subgraph(unvisited_nodes)
    mst = nx.minimum_spanning_tree(subgraph)
    mst_cost = mst.size(weight='weight')
    min_edge_costs = [
        graph[current_node][n]["weight"] for n in unvisited_nodes if graph.has_edge(current_node, n)
    ]
    min_edge_cost = min(min_edge_costs) if min_edge_costs else float('inf')
    return mst_cost + min_edge_cost

def initialize_pheromone_map(graph, initial_pheromone=1.0):
    pheromone_map = defaultdict(lambda: defaultdict(lambda: initial_pheromone))
    return pheromone_map

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
        heuristic_cost = mst_heuristic(graph, current_node, unvisited_nodes)
        for neighbor in unvisited_nodes:
            if graph.has_edge(current_node, neighbor):
                edge_cost = graph[current_node][neighbor]["weight"]
                new_path_cost = path_cost + edge_cost
                new_estimated_cost = new_path_cost + heuristic_cost
                heapq.heappush(frontier, (new_estimated_cost, neighbor, path + [neighbor], new_path_cost))
    return best_path, best_cost

def plot_graph(graph, path, title):
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(8, 5))
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold')
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    if path:
        path_edges = list(zip(path, path[1:] + [path[0]]))
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='r', width=3)
    plt.title(title)
    plt.show()

def run_tsp_and_display(graph, start_node, display_area):
    pheromone_map = initialize_pheromone_map(graph)
    start_time = time.time()
    path, cost = astar_tsp(graph, start_node, mst_heuristic, pheromone_map)
    elapsed_time = time.time() - start_time
    result_text = f"Costo de la ruta: {cost:.2f}, Tiempo: {elapsed_time:.4f}s"
    display_area.insert(END, result_text + "\n")
    path_text = "Ruta óptima: " + " -> ".join(path)
    display_area.insert(END, path_text + "\n")
    plot_graph(graph, path, result_text)
    display_area.insert(END, "\n")

# Interfaz para ingresar conexiones
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
        canvas = Canvas(connection_frame)
        scrollbar = Scrollbar(connection_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0)
        scrollbar.grid(row=0, column=1, sticky="ns")
        Label(scrollable_frame, text="Ingrese cada conexión (nodo1, nodo2, peso):").grid(row=0, column=0, columnspan=3)
        entry_widgets = []
        for i, (node1, node2) in enumerate(combinations(node_names, 2)):
            frame = Frame(scrollable_frame)
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
        Button(connection_frame, text="Guardar Conexiones", command=lambda: save_connections_to_file(node_names, get_edges_from_input(entry_widgets, node_names))).grid(row=len(entry_widgets) + 2, column=0, pady=5)
        Button(connection_frame, text="Cargar Conexiones", command=lambda: load_and_display_connections(edges_window, root)).grid(row=len(entry_widgets) + 3, column=0, pady=5)

    Button(edges_window, text="Crear nodos", command=create_node_entries).grid(row=1, column=0, pady=10)

def execute_tsp_from_input(window, root, entry_widgets, node_names):
    edges = get_edges_from_input(entry_widgets, node_names)
    if edges is None:
        return
    window.destroy()
    graph = create_graph_from_edges(node_names, edges)
    start_node = node_names[0]
    display_tsp_results(root, graph, start_node)

def load_and_display_connections(window, root):
    node_names, edges = load_connections_from_file()
    if node_names is None or edges is None:
        messagebox.showerror("Error", "No se pudo cargar el archivo.")
        return
    window.destroy()
    graph = create_graph_from_edges(node_names, edges)
    start_node = node_names[0]
    display_tsp_results(root, graph, start_node)

def display_tsp_results(root, graph, start_node):
    result_window = Toplevel(root)
    result_window.title("Resultados del TSP")
    display_frame = Frame(result_window)
    display_frame.pack(fill=BOTH, expand=True)
    scrollbar = Scrollbar(display_frame)
    scrollbar.pack(side=RIGHT, fill=Y)
    display_area = Text(display_frame, height=20, width=80, yscrollcommand=scrollbar.set)
    display_area.pack(side=LEFT, fill=BOTH, expand=True)
    scrollbar.config(command=display_area.yview)
    display_area.insert(END, f"Ejecutando TSP desde el nodo: {start_node}\n")
    run_tsp_and_display(graph, start_node, display_area)

# Código para ejecutar la interfaz gráfica
if __name__ == "__main__":
    root = Tk()
    root.title("Algoritmo del Vendedor Viajero (TSP)")
    menu = Menu(root)
    root.config(menu=menu)
    file_menu = Menu(menu)
    menu.add_cascade(label="Archivo", menu=file_menu)
    file_menu.add_command(label="Ingresar Conexiones", command=lambda: show_edge_input_interface(root))
    file_menu.add_separator()
    file_menu.add_command(label="Salir", command=root.quit)
    root.mainloop()
