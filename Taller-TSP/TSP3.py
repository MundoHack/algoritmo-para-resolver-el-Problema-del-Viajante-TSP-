import networkx as nx
import numpy as np
import heapq
from collections import defaultdict
import time

# Genera un grafo completo con pesos aleatorios
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

# Función para ejecutar experimentos con diferentes tamaños y complejidades
def run_tsp_experiments():
    node_sizes = [5, 6, 7, 8]  # Tamaños del grafo para probar
    results = []

    for num_nodes in node_sizes:
        print(f"\nRunning TSP for graph with {num_nodes} nodes:")
        graph = generate_graph(num_nodes)
        start_node = 0
        pheromone_map = initialize_pheromone_map(graph)
        
        # Evaluar A* con Heurística MST
        start_time = time.time()
        mst_path, mst_cost = astar_tsp(graph, start_node, mst_heuristic)
        mst_time = time.time() - start_time
        print(f" - MST Heuristic Path: {mst_path}, Cost: {mst_cost}, Time: {mst_time:.4f}s")
        
        # Evaluar A* con Heurística ACO
        start_time = time.time()
        aco_path, aco_cost = astar_tsp(graph, start_node, ant_colony_heuristic, pheromone_map)
        aco_time = time.time() - start_time
        print(f" - ACO Heuristic Path: {aco_path}, Cost: {aco_cost}, Time: {aco_time:.4f}s")
        
        # Evaluar A* con Heurística Combinada MST + ACO
        start_time = time.time()
        combined_path, combined_cost = astar_tsp(graph, start_node, combined_heuristic, pheromone_map)
        combined_time = time.time() - start_time
        print(f" - Combined Heuristic Path: {combined_path}, Cost: {combined_cost}, Time: {combined_time:.4f}s")
        
        # Almacenar los resultados
        results.append({
            "num_nodes": num_nodes,
            "mst": {"path": mst_path, "cost": mst_cost, "time": mst_time},
            "aco": {"path": aco_path, "cost": aco_cost, "time": aco_time},
            "combined": {"path": combined_path, "cost": combined_cost, "time": combined_time}
        })

    # Imprimir resumen de resultados
    print("\nSummary of results:")
    for result in results:
        print(f"Graph with {result['num_nodes']} nodes:")
        print(f"  MST Heuristic - Cost: {result['mst']['cost']}, Time: {result['mst']['time']:.4f}s")
        print(f"  ACO Heuristic - Cost: {result['aco']['cost']}, Time: {result['aco']['time']:.4f}s")
        print(f"  Combined Heuristic - Cost: {result['combined']['cost']}, Time: {result['combined']['time']:.4f}s")

# Ejecutar los experimentos
run_tsp_experiments()
