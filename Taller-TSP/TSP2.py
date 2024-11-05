import networkx as nx
import numpy as np
import heapq
from collections import defaultdict

# Genera un grafo de ejemplo con pesos aleatorios para el TSP
def generate_graph(num_nodes):
    graph = nx.complete_graph(num_nodes)
    for (u, v) in graph.edges():
        graph.edges[u, v]['weight'] = np.random.randint(1, 20)
    return graph

# Heurística 1: Árbol de Expansión Mínima (MST)
def mst_heuristic(graph, current_node, unvisited_nodes):
    subgraph = graph.subgraph(unvisited_nodes)
    mst = nx.minimum_spanning_tree(subgraph, algorithm="prim")
    mst_cost = mst.size(weight='weight')
    
    min_edge_cost = min(
        graph[current_node][n]["weight"] for n in unvisited_nodes if graph.has_edge(current_node, n)
    )
    
    return mst_cost + min_edge_cost

# Heurística 2: Colonia de Hormigas (ACO) - Simulación básica
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

# Combinación de Heurísticas: MST y ACO
def combined_heuristic(graph, current_node, unvisited_nodes, pheromone_map, alpha=0.5):
    mst_cost = mst_heuristic(graph, current_node, unvisited_nodes)
    aco_cost = ant_colony_heuristic(graph, current_node, unvisited_nodes, pheromone_map)
    return alpha * mst_cost + (1 - alpha) * aco_cost

# Algoritmo A* para el TSP
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
        
        # Ajustar el cálculo de heurística para cada caso
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

# Inicialización de ejemplo y ejecución de las tres heurísticas
def run_tsp_examples():
    num_nodes = 5
    graph = generate_graph(num_nodes)
    start_node = 0
    
    # Inicialización de la Heurística ACO
    pheromone_map = initialize_pheromone_map(graph)
    
    # 1. A* con Heurística MST
    mst_path, mst_cost = astar_tsp(graph, start_node, mst_heuristic)
    print(f"MST Heuristic Path: {mst_path}, Cost: {mst_cost}")
    
    # 2. A* con Heurística ACO
    aco_path, aco_cost = astar_tsp(graph, start_node, ant_colony_heuristic, pheromone_map)
    print(f"ACO Heuristic Path: {aco_path}, Cost: {aco_cost}")
    
    # 3. A* con Heurística Combinada MST + ACO
    combined_path, combined_cost = astar_tsp(graph, start_node, combined_heuristic, pheromone_map)
    print(f"Combined Heuristic Path: {combined_path}, Cost: {combined_cost}")

# Ejecución del ejemplo
run_tsp_examples()
