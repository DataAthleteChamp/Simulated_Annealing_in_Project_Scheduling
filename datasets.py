import numpy as np
import json
import random
import os
from typing import List, Dict
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd


def _create_theoretical_schedule(tasks: List[Dict]) -> Dict[int, tuple]:
    """Create a theoretical schedule for visualization"""
    schedule = {}
    current_time = 0
    task_end_times = {}

    task_ids = range(len(tasks))
    sorted_tasks = sorted(task_ids,
                          key=lambda x: (len(tasks[x]['dependencies']), -tasks[x]['priority']))

    for task_id in sorted_tasks:
        task = tasks[task_id]

        start_time = current_time
        for dep in task['dependencies']:
            if dep in task_end_times:
                start_time = max(start_time, task_end_times[dep])

        end_time = start_time + task['processing_time']
        schedule[task_id] = (start_time, end_time)
        task_end_times[task_id] = end_time

        current_time = min(current_time + task['processing_time'] / 2, end_time)

    return schedule


class MakespanOptimizedGenerator:
    def __init__(self, seed: int = 42):
        """Initialize the dataset generator"""
        np.random.seed(seed)
        random.seed(seed)

        self.output_dir = "advanced_task_scheduling_datasets"
        os.makedirs(self.output_dir, exist_ok=True)

        # Create visualization directory
        self.viz_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)

    def generate_dataset(self, complexity: str) -> Dict:
        """Generate dataset based on complexity level"""
        config = {
            'small': {
                'num_tasks': 100,
                'parallel_groups': 5,
                'max_processing_time': 50,
                'num_resources': 3,
                'dependency_chains': 3,
                'chain_length': 5,
                'resource_limit_multiplier': 4
            },
            'medium': {
                'num_tasks': 500,
                'parallel_groups': 10,
                'max_processing_time': 100,
                'num_resources': 4,
                'dependency_chains': 5,
                'chain_length': 8,
                'resource_limit_multiplier': 6
            },
            'large': {
                'num_tasks': 2000,
                'parallel_groups': 20,
                'max_processing_time': 200,
                'num_resources': 5,
                'dependency_chains': 8,
                'chain_length': 12,
                'resource_limit_multiplier': 8
            }
        }[complexity]

        resources = self._generate_resources(config)
        tasks = self._generate_parallel_tasks(config, resources)

        dataset = {
            'dataset_metadata': {
                'complexity_level': complexity,
                'total_tasks': len(tasks),
                'global_resources': resources,
                'parallel_groups': config['parallel_groups'],
                'generation_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            },
            'tasks': tasks
        }

        return dataset

    def _generate_resources(self, config: Dict) -> Dict:
        """Generate resources that allow parallel execution"""
        resources = {}
        resource_types = [
            ('CPU', (4, 16)),
            ('Memory', (16, 64)),
            ('GPU', (2, 8)),
            ('IO', (10, 50)),
            ('Network', (50, 200))
        ]

        selected_resources = random.sample(resource_types, config['num_resources'])

        for resource_name, (min_val, max_val) in selected_resources:
            base_capacity = random.randint(min_val, max_val)
            resources[resource_name] = base_capacity * config['resource_limit_multiplier']

        return resources

    def _generate_parallel_tasks(self, config: Dict, resources: Dict) -> List[Dict]:
        """Generate tasks with parallel execution opportunities"""
        tasks = []
        task_id = 0

        # Create parallel groups
        for group in range(config['parallel_groups']):
            group_size = random.randint(2, 5)

            for _ in range(group_size):
                if task_id >= config['num_tasks']:
                    break
                task = self._create_task(task_id, config, resources, group)
                tasks.append(task)
                task_id += 1

        # Fill remaining tasks
        while task_id < config['num_tasks']:
            task = self._create_task(task_id, config, resources, None)
            tasks.append(task)
            task_id += 1

        # Add dependency chains
        tasks = self._add_dependency_chains(tasks, config)

        return tasks

    def _create_task(self, task_id: int, config: Dict, resources: Dict, group: int) -> Dict:
        """Create a single task with appropriate characteristics"""
        processing_time = random.randint(config['max_processing_time'] // 5,
                                         config['max_processing_time'])

        resource_requirements = {}
        for resource, max_amount in resources.items():
            if group is not None:
                max_usage = max_amount / (config['parallel_groups'] * 1.5)
            else:
                max_usage = max_amount * 0.5

            requirement = random.uniform(max_usage * 0.2, max_usage)
            resource_requirements[resource] = requirement

        priority = random.uniform(0.5, 1.0)
        if group is not None:
            priority *= 1.5

        return {
            'task_id': task_id,
            'processing_time': processing_time,
            'resource_requirements': resource_requirements,
            'priority': priority,
            'parallel_group': group,
            'dependencies': [],
            'deadline': processing_time * random.uniform(2.0, 4.0)
        }

    def _add_dependency_chains(self, tasks: List[Dict], config: Dict) -> List[Dict]:
        """Add dependency chains that affect makespan"""
        num_tasks = len(tasks)

        for chain in range(config['dependency_chains']):
            chain_tasks = random.sample(range(num_tasks), config['chain_length'])
            chain_tasks.sort()

            for i in range(1, len(chain_tasks)):
                tasks[chain_tasks[i]]['dependencies'].append(chain_tasks[i - 1])
                tasks[chain_tasks[i]]['deadline'] = max(
                    tasks[chain_tasks[i]]['deadline'],
                    tasks[chain_tasks[i - 1]]['deadline'] + tasks[chain_tasks[i]]['processing_time']
                )

        return tasks

    def visualize_dataset(self, dataset: Dict, complexity: str):
        """Generate comprehensive visualizations for the dataset"""
        self._visualize_task_distribution(dataset, complexity)
        self._visualize_parallel_groups(dataset, complexity)
        self._visualize_resource_requirements(dataset, complexity)
        self._visualize_dependency_graph(dataset, complexity)
        self._visualize_theoretical_schedule(dataset, complexity)

    def _visualize_task_distribution(self, dataset: Dict, complexity: str):
        """Visualize distribution of task characteristics"""
        tasks = dataset['tasks']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        processing_times = [task['processing_time'] for task in tasks]
        sns.histplot(processing_times, ax=ax1, bins=30)
        ax1.set_title('Distribution of Processing Times')
        ax1.set_xlabel('Processing Time')

        priorities = [task['priority'] for task in tasks]
        sns.histplot(priorities, ax=ax2, bins=30)
        ax2.set_title('Distribution of Task Priorities')
        ax2.set_xlabel('Priority')

        dep_counts = [len(task['dependencies']) for task in tasks]
        sns.histplot(dep_counts, ax=ax3, bins=max(dep_counts) + 1)
        ax3.set_title('Dependencies per Task')
        ax3.set_xlabel('Number of Dependencies')

        deadlines = [task['deadline'] for task in tasks]
        sns.histplot(deadlines, ax=ax4, bins=30)
        ax4.set_title('Distribution of Deadlines')
        ax4.set_xlabel('Deadline')

        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, f'{complexity}_task_distributions.png'))
        plt.close()

    def _visualize_parallel_groups(self, dataset: Dict, complexity: str):
        """Visualize parallel groups and their characteristics"""
        tasks = dataset['tasks']

        groups = {}
        for task in tasks:
            group = task.get('parallel_group')
            if group not in groups:
                groups[group] = []
            groups[group].append(task)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        group_sizes = [len(tasks) for group, tasks in groups.items() if group is not None]
        sns.barplot(x=range(len(group_sizes)), y=group_sizes, ax=ax1)
        ax1.set_title('Size of Parallel Groups')
        ax1.set_xlabel('Group ID')
        ax1.set_ylabel('Number of Tasks')

        # Resource usage by group
        resource_types = list(dataset['dataset_metadata']['global_resources'].keys())
        group_resources = []

        for group_id, group_tasks in groups.items():
            if group_id is not None:
                avg_resources = {}
                for resource in resource_types:
                    avg_usage = np.mean([task['resource_requirements'][resource]
                                         for task in group_tasks])
                    avg_resources[resource] = avg_usage
                group_resources.append(avg_resources)

        if group_resources:  # Check if there are any parallel groups
            sns.boxplot(data=pd.DataFrame(group_resources), ax=ax2)
            ax2.set_title('Resource Requirements by Parallel Group')
            ax2.set_xlabel('Resource Type')
            ax2.set_ylabel('Average Resource Usage')
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, f'{complexity}_parallel_groups.png'))
        plt.close()

    def _visualize_resource_requirements(self, dataset: Dict, complexity: str):
        """Visualize resource requirements and utilization potential"""
        tasks = dataset['tasks']
        resources = dataset['dataset_metadata']['global_resources']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Resource usage heatmap
        resource_matrix = []
        for task in tasks[:min(50, len(tasks))]:
            row = [task['resource_requirements'][r] for r in resources.keys()]
            resource_matrix.append(row)

        sns.heatmap(resource_matrix,
                    xticklabels=list(resources.keys()),
                    yticklabels=range(len(resource_matrix)),
                    ax=ax1,
                    cmap='YlOrRd')
        ax1.set_title('Resource Requirements Heatmap (First 50 Tasks)')

        # Resource utilization potential
        resource_usage = []
        for resource, limit in resources.items():
            usage = sum(task['resource_requirements'][resource] for task in tasks)
            resource_usage.append((resource, usage, limit))

        resources, usage, limits = zip(*resource_usage)
        x = range(len(resources))

        ax2.bar(x, limits, alpha=0.3, label='Resource Limit')
        ax2.bar(x, usage, alpha=0.7, label='Total Requirements')
        ax2.set_xticks(x)
        ax2.set_xticklabels(resources, rotation=45)
        ax2.set_title('Resource Utilization Potential')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, f'{complexity}_resource_requirements.png'))
        plt.close()

    def _visualize_dependency_graph(self, dataset: Dict, complexity: str):
        """Visualize task dependencies as a directed graph"""
        tasks = dataset['tasks']

        G = nx.DiGraph()

        for task in tasks:
            G.add_node(task['task_id'],
                       processing_time=task['processing_time'],
                       priority=task['priority'])
            for dep in task['dependencies']:
                G.add_edge(dep, task['task_id'])

        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G)

        node_sizes = [G.nodes[node]['processing_time'] * 100 for node in G.nodes()]
        node_colors = [G.nodes[node]['priority'] for node in G.nodes()]

        nx.draw(G, pos,
                node_size=node_sizes,
                node_color=node_colors,
                cmap=plt.cm.viridis,
                with_labels=True,
                arrows=True,
                edge_color='gray',
                alpha=0.6)

        plt.title('Task Dependency Graph\nNode size = Processing Time, Color = Priority')
        plt.savefig(os.path.join(self.viz_dir, f'{complexity}_dependency_graph.png'))
        plt.close()

    def _visualize_theoretical_schedule(self, dataset: Dict, complexity: str):
        """Visualize a theoretical parallel schedule"""
        tasks = dataset['tasks']
        schedule = _create_theoretical_schedule(tasks)

        plt.figure(figsize=(15, 8))

        for task_id, (start_time, end_time) in schedule.items():
            plt.barh(y=task_id,
                     width=end_time - start_time,
                     left=start_time,
                     alpha=0.6,
                     color=plt.cm.viridis(tasks[task_id]['priority']))

            plt.axvline(x=tasks[task_id]['deadline'],
                        ymin=(task_id - 0.4) / len(tasks),
                        ymax=(task_id + 0.4) / len(tasks),
                        color='red',
                        linestyle='--',
                        alpha=0.3)

        plt.title('Theoretical Parallel Schedule\nColor intensity indicates priority')
        plt.xlabel('Time')
        plt.ylabel('Task ID')
        plt.grid(True, alpha=0.3)

        plt.savefig(os.path.join(self.viz_dir, f'{complexity}_theoretical_schedule.png'))
        plt.close()

    def _print_dataset_stats(self, dataset: Dict, complexity: str):
        """Print statistics about the generated dataset"""
        tasks = dataset['tasks']
        resources = dataset['dataset_metadata']['global_resources']

        print(f"\nDataset Statistics ({complexity}):")
        print(f"Number of tasks: {len(tasks)}")
        print(f"Number of parallel groups: {dataset['dataset_metadata']['parallel_groups']}")

        # Parallel group stats
        parallel_tasks = [t for t in tasks if t.get('parallel_group') is not None]
        print(f"Tasks in parallel groups: {len(parallel_tasks)}")

        # Processing time stats
        proc_times = [t['processing_time'] for t in tasks]
        print(f"Processing times: min={min(proc_times)}, max={max(proc_times)}, "
              f"avg={sum(proc_times) / len(proc_times):.2f}")

        # Dependency stats
        dep_counts = [len(t['dependencies']) for t in tasks]
        print(f"Dependencies per task: min={min(dep_counts)}, max={max(dep_counts)}, "
              f"avg={sum(dep_counts) / len(dep_counts):.2f}")

        # Resource stats
        for resource in resources:
            usage = [t['resource_requirements'][resource] for t in tasks]
            print(f"{resource} usage: min={min(usage):.2f}, max={max(usage):.2f}, "
                  f"limit={resources[resource]}")

        # Theoretical bounds
        sequential_time = sum(proc_times)
        parallel_time = sequential_time / dataset['dataset_metadata']['parallel_groups']
        print(f"\nMakespan bounds:")
        print(f"Sequential execution: {sequential_time}")
        print(f"Perfect parallel (theoretical): {parallel_time:.2f}")
        print("=" * 50)

    def generate_and_save_datasets(self):
        """Generate, save, and visualize all datasets"""
        for complexity in ['small', 'medium', 'large']:
            print(f"\nGenerating {complexity} dataset...")
            dataset = self.generate_dataset(complexity)

            # Save dataset
            filename = f'advanced_task_scheduling_{complexity}_dataset.json'
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(dataset, f, indent=2)

            # Generate visualizations
            print(f"Generating visualizations for {complexity} dataset...")
            self.visualize_dataset(dataset, complexity)

            # Print statistics
            self._print_dataset_stats(dataset, complexity)

            print(f"Dataset and visualizations saved in {self.output_dir}")


def main():
    """Main function to run the dataset generator"""
    print("Starting Task Scheduling Dataset Generator...")
    generator = MakespanOptimizedGenerator(seed=42)
    generator.generate_and_save_datasets()
    print("\nDataset generation complete!")
    print("\nGenerated files:")
    print("1. Datasets (JSON):")
    print("   - advanced_task_scheduling_small_dataset.json")
    print("   - advanced_task_scheduling_medium_dataset.json")
    print("   - advanced_task_scheduling_large_dataset.json")
    print("\n2. Visualizations (PNG):")
    print("   - *_task_distributions.png")
    print("   - *_parallel_groups.png")
    print("   - *_resource_requirements.png")
    print("   - *_dependency_graph.png")
    print("   - *_theoretical_schedule.png")


if __name__ == "__main__":
    main()

