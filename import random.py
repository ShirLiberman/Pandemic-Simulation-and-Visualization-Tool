import random
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict

class PandemicSimulation:
    def __init__(self, population_size: int, initial_infected: int = 1):
        """
        Initialize the pandemic simulation.

        Args:
            population_size (int): Total number of individuals in the simulation.
            initial_infected (int): Number of initially infected individuals.
        """
        self.population_size = population_size

        # Create a social network graph using the Watts-Strogatz model
        self.social_network = nx.watts_strogatz_graph(
            n=population_size,
            k=4,  # Each node connected to 4 nearest neighbors
            p=0.1  # Probability of rewiring edges
        )

        # Disease parameters
        self.infection_rate = 0.3
        self.recovery_rate = 0.1

        # Simulation state: Initialize all nodes as 'susceptible'
        self.node_states = {node: 'susceptible' for node in self.social_network.nodes()}

        # Initially infect random nodes
        for _ in range(initial_infected):
            node = random.choice(list(self.social_network.nodes()))
            self.node_states[node] = 'infected'

    def bfs_infection_trace(self, start_node: int) -> Dict[int, List[int]]:
        """
        Trace the potential infection path using BFS.

        Args:
            start_node (int): Node to start tracing from.

        Returns:
            Dict mapping each infected node to its infection path.
        """
        infection_paths = {}
        queue = [(start_node, [start_node])]
        visited = set()

        while queue:
            current_node, path = queue.pop(0)

            if current_node in visited:
                continue

            visited.add(current_node)

            # Check neighbors for potential infection
            for neighbor in list(self.social_network.neighbors(current_node)):
                if neighbor not in self.social_network.nodes():
                    continue
                if self.node_states[neighbor] == 'susceptible':
                    # Probabilistic infection spread
                    if random.random() < self.infection_rate:
                        self.node_states[neighbor] = 'infected'
                        new_path = path + [neighbor]
                        infection_paths[neighbor] = new_path
                        queue.append((neighbor, new_path))

        return infection_paths

    def block_infection_chain(self, critical_nodes: List[int]) -> None:
        """
        Block infection by removing or quarantining critical nodes.

        Args:
            critical_nodes (List[int]): Nodes to remove/quarantine.
        """
        for node in critical_nodes:
            if node in self.social_network.nodes():
                # Remove connections for quarantined node
                self.social_network.remove_node(node)
                self.node_states[node] = 'quarantined'

    def run_simulation(self, steps: int = 10) -> Dict[str, List[int]]:
        """
        Run the pandemic simulation.

        Args:
            steps (int): Number of simulation steps.

        Returns:
            Simulation statistics.
        """
        simulation_stats = {
            'susceptible': [sum(1 for state in self.node_states.values() if state == 'susceptible')],
            'infected': [sum(1 for state in self.node_states.values() if state == 'infected')],
            'recovered': [sum(1 for state in self.node_states.values() if state == 'recovered')],
            'quarantined': [sum(1 for state in self.node_states.values() if state == 'quarantined')]
        }

        for step in range(steps):
            current_infected = [node for node, state in self.node_states.items() if state == 'infected']

            for infected_node in current_infected:
                self.bfs_infection_trace(infected_node)

                # Recovery progression
                if random.random() < self.recovery_rate:
                    self.node_states[infected_node] = 'recovered'

            # Update stats
            simulation_stats['susceptible'].append(sum(1 for state in self.node_states.values() if state == 'susceptible'))
            simulation_stats['infected'].append(sum(1 for state in self.node_states.values() if state == 'infected'))
            simulation_stats['recovered'].append(sum(1 for state in self.node_states.values() if state == 'recovered'))
            simulation_stats['quarantined'].append(sum(1 for state in self.node_states.values() if state == 'quarantined'))

        return simulation_stats

    def visualize_results(self, simulation_stats: Dict[str, List[int]]):
        """
        Visualize simulation results.

        Args:
            simulation_stats (Dict): Simulation statistics to plot.
        """
        plt.figure(figsize=(10, 6))
        for category, values in simulation_stats.items():
            plt.plot(values, label=category)

        plt.title('Pandemic Spread Simulation')
        plt.xlabel('Simulation Steps')
        plt.ylabel('Number of Individuals')
        plt.legend()
        plt.tight_layout()
        plt.show()

def main():
    # Create simulation
    simulation = PandemicSimulation(
        population_size=1000,  # Total population
        initial_infected=5  # Starting number of infected individuals
    )

    print("Initial Pandemic Simulation Setup:")
    print(f"Total Population: {simulation.population_size}")
    print(f"Initial Infected: {sum(1 for state in simulation.node_states.values() if state == 'infected')}")

    # Run simulation without intervention
    print("\nRunning Simulation without Intervention...")
    no_intervention_stats = simulation.run_simulation(steps=20)

    # Reset simulation
    simulation = PandemicSimulation(
        population_size=1000,
        initial_infected=5
    )

    # Identify and block critical nodes (top 5% most connected)
    critical_nodes = sorted(
        simulation.social_network.degree,
        key=lambda x: x[1],
        reverse=True
    )[:int(0.05 * simulation.population_size)]
    critical_node_ids = [node for node, degree in critical_nodes]

    # Block infection chain
    print("\nBlocking Infection Chain at Critical Nodes...")
    simulation.block_infection_chain(critical_node_ids)

    # Run simulation with intervention
    intervention_stats = simulation.run_simulation(steps=20)

    # Visualize results
    print("\nVisualizing Simulation Results...")
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.title('No Intervention')
    for category, values in no_intervention_stats.items():
        plt.plot(values, label=category)
    plt.xlabel('Simulation Steps')
    plt.ylabel('Number of Individuals')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title('With Intervention')
    for category, values in intervention_stats.items():
        plt.plot(values, label=category)
    plt.xlabel('Simulation Steps')
    plt.ylabel('Number of Individuals')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
