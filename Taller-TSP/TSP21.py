import json
import networkx as nx
import heapq
from collections import defaultdict
import time
from tkinter import *
from tkinter import filedialog, messagebox, Toplevel, Frame, Label
from itertools import combinations
import matplotlib.pyplot as plt

# Funciones para manejo de archivos JSON
def save_connections_to_file(node_names, edges):
    data = {"nodes": node_names, "edges": edges}
    file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
    if file_path:
        with open(file_path, "w") as f:
            json.dump(data, f)
        messagebox.showinfo("Guardar Conexiones", f"Conexiones guardadas en {file_path}")

def load_connections_from_file(display_area):
    file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
    if file_path:
        with open(file_path, "r") as f:
            data = json.load(f)
        node_names = data["nodes"]
        edges = data["edges"]
        graph = create_graph_from_edges(node_names, edges)
        
        # Ejecuta el proceso de TSP después de cargar los datos
        display_area.delete(1.0, END)  # Limpiar el área de visualización
        run_tsp_and_display(graph, node_names[0], display_area)
    else:
        messagebox.showinfo("Carga de Conexiones", "No se seleccionó ningún archivo.")

# Funciones para crear grafo y obtener conexiones desde la interfaz
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
    for entry1, entry2, weight_entry in entry_widgets:
        try:
            u = entry1.get().strip()
            v = entry2.get().strip()
            cost = float(weight_entry.get())
            if u and v and u in node_names and v in node_names:
                edges.append((u, v, cost))
        except ValueError:
            print(f"Error en la entrada.")
            return None
    return edges

# Implementación de heurísticas
def mst_heuristic(graph, current_node, unvisited_nodes):
    subgraph = graph.subgraph(unvisited_nodes)
    mst = nx.minimum_spanning_tree(subgraph)
    mst_cost = mst.size(weight='weight')
    min_edge_costs = [
        graph[current_node][n]["weight"] for n in unvisited_nodes if graph.has_edge(current_node, n)
    ]
    min_edge_cost = min(min_edge_costs) if min_edge_costs else float('inf')
    return mst_cost + min_edge_cost

def aco_heuristic(graph, start_node, unvisited_nodes):
    return mst_heuristic(graph, start_node, unvisited_nodes)

def combined_heuristic(graph, current_node, unvisited_nodes):
    return (mst_heuristic(graph, current_node, unvisited_nodes) + 
            aco_heuristic(graph, current_node, unvisited_nodes)) / 2

# A* para resolver TSP
def astar_tsp(graph, start_node, heuristic_func):
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
        heuristic_cost = heuristic_func(graph, current_node, unvisited_nodes)
        for neighbor in unvisited_nodes:
            if graph.has_edge(current_node, neighbor):
                edge_cost = graph[current_node][neighbor]["weight"]
                new_path_cost = path_cost + edge_cost
                new_estimated_cost = new_path_cost + heuristic_cost
                heapq.heappush(frontier, (new_estimated_cost, neighbor, path + [neighbor], new_path_cost))
    return best_path, best_cost

# Función para mostrar resultados en la interfaz y graficar
def run_tsp_and_display(graph, start_node, display_area):
    heuristics = {
        "MST": mst_heuristic,
        "ACO": aco_heuristic,
        "Combined": combined_heuristic
    }
    results = {}
    for name, heuristic in heuristics.items():
        start_time = time.time()
        path, cost = astar_tsp(graph, start_node, heuristic)
        elapsed_time = time.time() - start_time
        results[name] = (cost, elapsed_time, path)

        # Muestra en el área de texto
        result_text = f"{name} Heuristic - Cost: {cost:.2f}, Time: {elapsed_time:.4f}s"
        display_area.insert(END, result_text + "\n")
        path_text = "Best Path: " + " -> ".join(path)
        display_area.insert(END, path_text + "\n")

        # Muestra el gráfico para el camino encontrado
        show_graph(graph, path, name)

    # Resultados finales
    display_area.insert(END, "\nResultados:\n")
    for name, (cost, elapsed_time, _) in results.items():
        display_area.insert(END, f"{name}: Costo = {cost:.2f}, Tiempo = {elapsed_time:.4f}s\n")
    display_area.insert(END, "\n")

# Función para graficar la ruta del TSP encontrada
def show_graph(graph, path, heuristic_name):
    pos = nx.spring_layout(graph)  # Posiciona los nodos para el grafo
    plt.figure(figsize=(8, 6))
    nx.draw(graph, pos, with_labels=True, node_color="lightblue", node_size=500, font_size=10, font_weight="bold")
    edge_labels = {(u, v): f"{d['weight']}" for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color="red")
    
    # Dibuja el mejor camino
    path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color="blue", width=2)
    
    plt.title(f"Ruta TSP usando {heuristic_name} Heurística")
    plt.show()

# Interfaz para ingresar conexiones y botones
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
        Label(scrollable_frame, text="Conexiones (Nodo A - Nodo B - Peso)").grid(row=0, column=0, columnspan=3, pady=10)
        
        entry_widgets = []
        for i, (u, v) in enumerate(combinations(node_names, 2)):
            entry1 = Entry(scrollable_frame, width=10)
            entry1.insert(0, u)
            entry1.grid(row=i + 1, column=0)
            entry2 = Entry(scrollable_frame, width=10)
            entry2.insert(0, v)
            entry2.grid(row=i + 1, column=1)
            weight_entry = Entry(scrollable_frame, width=10)
            weight_entry.grid(row=i + 1, column=2)
            entry_widgets.append((entry1, entry2, weight_entry))
        
        display_area = Text(scrollable_frame, width=50, height=20)
        display_area.grid(row=len(entry_widgets) + 2, column=0, columnspan=3, pady=10)
        
        Button(scrollable_frame, text="Guardar Conexión", command=lambda: save_connections_to_file(node_names, get_edges_from_input(entry_widgets, node_names))).grid(row=len(entry_widgets) + 1, column=0, pady=10)
        Button(scrollable_frame, text="Generar TSP", command=lambda: run_tsp_and_display(create_graph_from_edges(node_names, get_edges_from_input(entry_widgets, node_names)), node_names[0], display_area)).grid(row=len(entry_widgets) + 1, column=1, pady=10)
    
    node_count_entry.bind("<Return>", lambda event: create_node_entries())

# Función principal para la interfaz
def main():
    root = Tk()
    root.title("Ejemplo de TSP con Heurísticas")
    display_area = Text(root, width=50, height=20)
    display_area.pack(pady=10)
    Button(root, text="Cargar Conexiones", command=lambda: load_connections_from_file(display_area)).pack(pady=10)
    Button(root, text="Ingresar Nuevas Conexiones", command=lambda: show_edge_input_interface(root)).pack(pady=10)
    root.mainloop()

if __name__ == "__main__":
    main()
