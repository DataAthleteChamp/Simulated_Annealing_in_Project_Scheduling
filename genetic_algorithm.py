import numpy as np
import json
import time
from typing import List, Dict, Tuple
import random
import os
from datetime import datetime
import matplotlib.pyplot as plt


class GeneticScheduler:
    def __init__(self, dataset_path: str):
        """Initialize the genetic scheduler"""
        self._load_dataset(dataset_path)
        self._initialize_tracking()
        self._create_output_directories()

    def create_initial_population(self) -> List[List[int]]:
        """Create initial population with heuristic solutions"""
        population = []

        try:
            # Add heuristic solutions
            # 1. Deadline-based
            deadline_order = sorted(range(self.num_tasks),
                                    key=lambda x: self.tasks[x]['deadline'])
            population.append(deadline_order)

            # 2. Priority-based
            priority_order = sorted(range(self.num_tasks),
                                    key=lambda x: -self.tasks[x]['priority'])
            population.append(priority_order)

            # 3. Dependency-based
            dependency_order = sorted(range(self.num_tasks),
                                      key=lambda x: len(self.tasks[x].get('dependencies', [])))
            population.append(dependency_order)

            # Fill rest with random permutations
            while len(population) < self.population_size:
                solution = list(range(self.num_tasks))
                random.shuffle(solution)
                population.append(solution)

            return population

        except Exception as e:
            print(f"Error creating initial population: {str(e)}")
            # Return a simple random population as fallback
            return [list(range(self.num_tasks)) for _ in range(self.population_size)]

    def calculate_fitness(self, schedule: List[int]) -> float:
        """Calculate fitness score (higher is better)"""
        try:
            if len(set(schedule)) != self.num_tasks:
                return float('-inf')

            current_time = 0
            resource_usage = {r: 0 for r in self.global_resources.keys()}
            task_end_times = {}
            penalty = 0
            reward = 0

            for task_idx in schedule:
                task = self.tasks[task_idx]
                start_time = current_time

                # Check dependencies
                for dep in task.get('dependencies', []):
                    if dep in task_end_times:
                        start_time = max(start_time, task_end_times[dep])
                    else:
                        penalty += 1000  # Heavy penalty for unmet dependencies

                # Resource constraints
                for resource, amount in task['resource_requirements'].items():
                    resource_usage[resource] += amount
                    if resource_usage[resource] > self.global_resources[resource]:
                        penalty += 500

                end_time = start_time + task['processing_time']
                task_end_times[task_idx] = end_time

                # Deadline penalty
                if end_time > task['deadline']:
                    penalty += (end_time - task['deadline']) * 10

                # Priority reward
                reward += task['priority'] * 50

                current_time = max(current_time, start_time + 1)

            makespan = max(task_end_times.values())

            # Final fitness score
            fitness = 10000 - penalty + reward - (makespan * 5)
            return float(fitness)

        except Exception as e:
            print(f"Error calculating fitness: {str(e)}")
            return float('-inf')

    def order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Perform order crossover (OX1)"""
        try:
            size = len(parent1)
            # Select substring
            start, end = sorted(random.sample(range(size), 2))

            # Initialize offspring
            child = [-1] * size
            child[start:end] = parent1[start:end]

            # Fill remaining positions
            parent2_idx = 0
            for i in range(size):
                if child[i] == -1:
                    while parent2[parent2_idx] in child:
                        parent2_idx += 1
                    child[i] = parent2[parent2_idx]

            return child

        except Exception as e:
            print(f"Error in crossover: {str(e)}")
            return parent1.copy()  # Return copy of first parent as fallback

    def mutate(self, schedule: List[int]) -> List[int]:
        """Apply mutation operators"""
        try:
            schedule = schedule.copy()  # Work on a copy

            if random.random() < self.mutation_rate:
                mutation_type = random.choice(['swap', 'insert', 'reverse'])

                if mutation_type == 'swap':
                    # Swap two positions
                    i, j = random.sample(range(self.num_tasks), 2)
                    schedule[i], schedule[j] = schedule[j], schedule[i]

                elif mutation_type == 'insert':
                    # Insert one position into another
                    i, j = random.sample(range(self.num_tasks), 2)
                    value = schedule.pop(i)
                    schedule.insert(j, value)

                else:  # reverse
                    # Reverse a subsequence
                    i, j = sorted(random.sample(range(self.num_tasks), 2))
                    schedule[i:j + 1] = reversed(schedule[i:j + 1])

            return schedule

        except Exception as e:
            print(f"Error in mutation: {str(e)}")
            return schedule  # Return unchanged schedule as fallback

    def _tournament_selection(self, population: List[List[int]], tournament_size: int = 3) -> List[int]:
        """Perform tournament selection"""
        try:
            tournament = random.sample(population, tournament_size)
            return max(tournament, key=self.calculate_fitness)
        except Exception as e:
            print(f"Error in tournament selection: {str(e)}")
            return random.choice(population)  # Random selection as fallback

    def _initialize_tracking(self):
        """Initialize tracking variables with autotuned parameters"""
        # Auto-tune GA parameters based on problem characteristics
        self.population_size = self._tune_population_size()
        self.elite_size = self._tune_elite_size()
        self.mutation_rate = self._tune_mutation_rate()
        self.generations = self._tune_generations()

        # Tracking variables
        self.best_schedule = None
        self.best_fitness = float('-inf')
        self.best_cost = float('inf')
        self.avg_fitness_history = []
        self.best_fitness_history = []
        self.current_violations = {'deadline': 0, 'resource': 0}
        self.start_time = None

    def _tune_population_size(self) -> int:
        """Tune population size based on problem characteristics"""
        # Base size on number of tasks and dependencies
        total_dependencies = sum(len(task.get('dependencies', []))
                                 for task in self.tasks)
        dependency_complexity = total_dependencies / (self.num_tasks + 1)

        # Calculate base population size
        base_size = min(200, self.num_tasks * 2)

        # Adjust based on problem characteristics
        if self.num_tasks < 200:  # Small problem
            pop_size = base_size
        elif self.num_tasks < 1000:  # Medium problem
            pop_size = base_size * 1.5
        else:  # Large problem
            pop_size = base_size * 2

        # Adjust for dependency complexity
        pop_size *= (1 + dependency_complexity / 2)

        # Ensure population size is even for pairing
        pop_size = int(pop_size)
        if pop_size % 2 != 0:
            pop_size += 1

        return min(400, pop_size)  # Cap at 400 to prevent excessive computation

    def _tune_elite_size(self) -> int:
        """Tune elite size based on population size and problem characteristics"""
        # Base elite size on population size
        base_elite = max(2, self.population_size // 20)

        # Adjust based on problem size
        if self.num_tasks < 200:
            return base_elite
        elif self.num_tasks < 1000:
            return int(base_elite * 1.5)
        else:
            return int(base_elite * 2)

    def _tune_mutation_rate(self) -> float:
        """Tune mutation rate based on problem characteristics"""
        # Calculate basic problem complexity
        total_dependencies = sum(len(task.get('dependencies', []))
                                 for task in self.tasks)
        dependency_ratio = total_dependencies / (self.num_tasks + 1)

        # Base mutation rate on problem size
        if self.num_tasks < 200:
            base_rate = 0.2  # Higher for small problems
        elif self.num_tasks < 1000:
            base_rate = 0.15  # Medium problems
        else:
            base_rate = 0.1  # Lower for large problems

        # Adjust for dependency complexity
        adjusted_rate = base_rate * (1 + dependency_ratio / 4)

        return min(0.3, max(0.05, adjusted_rate))  # Keep between 5% and 30%

    def _tune_generations(self) -> int:
        """Tune number of generations based on problem size and complexity"""
        # Base generations on problem size
        base_generations = 100

        # Calculate problem complexity
        resource_complexity = len(self.global_resources)
        dependency_complexity = sum(len(task.get('dependencies', []))
                                    for task in self.tasks) / self.num_tasks

        # Adjust base generations
        adjusted_generations = base_generations * (1 + np.log2(self.num_tasks / 100 + 1))

        # Factor in resource and dependency complexity
        complexity_factor = (1 + resource_complexity / 5) * (1 + dependency_complexity)
        final_generations = int(adjusted_generations * complexity_factor)

        # Cap generations based on problem size
        if self.num_tasks < 200:
            return min(200, final_generations)
        elif self.num_tasks < 1000:
            return min(300, final_generations)
        else:
            return min(400, final_generations)

    def _plot_resource_utilization(self):
        """Plot resource utilization over time"""
        try:
            final_schedule = self._calculate_final_schedule()
            makespan = max(task['end_time'] for task in final_schedule)

            # Calculate resource usage timeline
            timeline = {t: {r: 0 for r in self.global_resources}
                        for t in range(int(makespan) + 1)}

            for task in final_schedule:
                task_id = task['task_id']
                start = int(task['start_time'])
                end = int(task['end_time'])

                for t in range(start, end):
                    for resource, amount in self.tasks[task_id]['resource_requirements'].items():
                        timeline[t][resource] += amount

            plt.figure(figsize=(15, 8))

            for resource in self.global_resources:
                usage = [timeline[t][resource] for t in range(int(makespan) + 1)]
                plt.plot(usage, label=f'{resource} Usage', alpha=0.7)

                # Add capacity line
                plt.axhline(y=self.global_resources[resource],
                            color='red',
                            linestyle='--',
                            alpha=0.3,
                            label=f'{resource} Capacity')

            plt.title('Resource Utilization Over Time')
            plt.xlabel('Time')
            plt.ylabel('Resource Usage')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.savefig(os.path.join(self.viz_dir, 'resource_utilization.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error plotting resource utilization: {str(e)}")

    def _plot_violations(self):
        """Plot violations over time"""
        try:
            final_schedule = self._calculate_final_schedule()
            makespan = max(task['end_time'] for task in final_schedule)

            timeline = {t: {'deadline': 0, 'resource': 0}
                        for t in range(int(makespan) + 1)}

            # Calculate violations at each time point
            for t in range(int(makespan) + 1):
                # Deadline violations
                for task in final_schedule:
                    if task['start_time'] <= t <= task['end_time'] and t > task['deadline']:
                        timeline[t]['deadline'] += 1

                # Resource violations
                resource_usage = {r: 0 for r in self.global_resources}
                for task in final_schedule:
                    if task['start_time'] <= t < task['end_time']:
                        task_id = task['task_id']
                        for resource, amount in self.tasks[task_id]['resource_requirements'].items():
                            resource_usage[resource] += amount
                            if resource_usage[resource] > self.global_resources[resource]:
                                timeline[t]['resource'] += 1

            plt.figure(figsize=(15, 6))

            times = list(timeline.keys())
            deadline_violations = [t['deadline'] for t in timeline.values()]
            resource_violations = [t['resource'] for t in timeline.values()]

            plt.plot(times, deadline_violations, 'r-', label='Deadline Violations', alpha=0.7)
            plt.plot(times, resource_violations, 'b-', label='Resource Violations', alpha=0.7)

            plt.title('Constraint Violations Over Time')
            plt.xlabel('Time')
            plt.ylabel('Number of Violations')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.savefig(os.path.join(self.viz_dir, 'violations.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error plotting violations: {str(e)}")

    def count_deadline_violations(self, schedule: List[int]) -> int:
        """Count number of deadline violations"""
        try:
            violations = 0
            task_end_times = {}
            current_time = 0

            for task_idx in schedule:
                task = self.tasks[task_idx]
                start_time = current_time

                # Consider dependencies
                for dep in task.get('dependencies', []):
                    if dep in task_end_times:
                        start_time = max(start_time, task_end_times[dep])

                # Calculate end time
                end_time = start_time + int(task['processing_time'])
                task_end_times[task_idx] = end_time

                # Check deadline violation
                if end_time > task['deadline']:
                    violations += 1

                current_time = max(current_time, start_time + 1)

            return violations

        except Exception as e:
            print(f"Error counting deadline violations: {str(e)}")
            return 0

    def count_resource_violations(self, schedule: List[int]) -> int:
        """Count number of resource violations"""
        try:
            violations = 0
            resource_timeline = {}

            for task_idx in schedule:
                task = self.tasks[task_idx]
                start_time = self._get_earliest_start_time(task_idx, schedule)
                end_time = start_time + int(task['processing_time'])

                # Check resource violations for each time slot
                for t in range(start_time, end_time):
                    if t not in resource_timeline:
                        resource_timeline[t] = {r: 0 for r in self.global_resources}

                    for resource, amount in task['resource_requirements'].items():
                        resource_timeline[t][resource] += amount
                        if resource_timeline[t][resource] > self.global_resources[resource]:
                            violations += 1

            return violations

        except Exception as e:
            print(f"Error counting resource violations: {str(e)}")
            return 0

    def _get_earliest_start_time(self, task_idx: int, schedule: List[int]) -> int:
        """Calculate earliest possible start time for a task considering dependencies"""
        try:
            task = self.tasks[task_idx]
            dependencies = task.get('dependencies', [])

            if not dependencies:
                return 0

            # Calculate end times of dependencies
            dep_end_times = []
            for dep in dependencies:
                if dep in schedule[:schedule.index(task_idx)]:  # Only consider scheduled dependencies
                    dep_task = self.tasks[dep]
                    dep_start = self._get_earliest_start_time(dep, schedule)
                    dep_end_times.append(dep_start + int(dep_task['processing_time']))

            return max(dep_end_times) if dep_end_times else 0

        except Exception as e:
            print(f"Error calculating earliest start time: {str(e)}")
            return 0

    def _calculate_schedule_metrics(self, schedule: List[int]) -> Dict:
        """Calculate comprehensive schedule metrics"""
        try:
            metrics = {
                'makespan': 0,
                'total_idle_time': 0,
                'resource_utilization': {},
                'violations': {
                    'deadline': self.count_deadline_violations(schedule),
                    'resource': self.count_resource_violations(schedule)
                }
            }

            # Calculate makespan and resource utilization
            resource_timeline = {}
            task_end_times = {}

            for task_idx in schedule:
                task = self.tasks[task_idx]
                start_time = self._get_earliest_start_time(task_idx, schedule)
                end_time = start_time + int(task['processing_time'])
                task_end_times[task_idx] = end_time

                # Update resource timeline
                for t in range(start_time, end_time):
                    if t not in resource_timeline:
                        resource_timeline[t] = {r: 0 for r in self.global_resources}
                    for resource, amount in task['resource_requirements'].items():
                        resource_timeline[t][resource] += amount

            # Calculate makespan
            metrics['makespan'] = max(task_end_times.values()) if task_end_times else 0

            # Calculate resource utilization
            for resource in self.global_resources:
                total_usage = sum(timeline[resource] for timeline in resource_timeline.values())
                capacity = self.global_resources[resource] * metrics['makespan']
                metrics['resource_utilization'][resource] = total_usage / capacity if capacity > 0 else 0

            return metrics

        except Exception as e:
            print(f"Error calculating schedule metrics: {str(e)}")
            return {
                'makespan': 0,
                'total_idle_time': 0,
                'resource_utilization': {},
                'violations': {'deadline': 0, 'resource': 0}
            }

    def _verify_schedule(self, schedule: List[int]) -> bool:
        """Verify schedule validity"""
        try:
            # Check for duplicates
            if len(set(schedule)) != self.num_tasks:
                return False

            # Check all tasks are included
            if set(schedule) != set(range(self.num_tasks)):
                return False

            # Check dependency order
            task_positions = {task_id: pos for pos, task_id in enumerate(schedule)}
            for task_id in schedule:
                task = self.tasks[task_id]
                for dep in task.get('dependencies', []):
                    if task_positions[dep] > task_positions[task_id]:
                        return False

            return True

        except Exception as e:
            print(f"Error verifying schedule: {str(e)}")
            return False

    def _load_dataset(self, dataset_path: str):
        """Load and validate the dataset"""
        try:
            with open(dataset_path, 'r') as f:
                self.dataset = json.load(f)

            self.tasks = self.dataset['tasks']
            self.num_tasks = len(self.tasks)
            self.global_resources = self.dataset['dataset_metadata']['global_resources']
            self.parallel_groups = self.dataset.get('dataset_metadata', {}).get('parallel_groups', {})
        except Exception as e:
            raise ValueError(f"Error loading dataset: {str(e)}")

    def _initialize_tracking(self):
        """Initialize tracking variables with autotuned parameters"""
        # Auto-tune GA parameters based on problem characteristics
        self.population_size = self._tune_population_size()
        self.elite_size = self._tune_elite_size()
        self.mutation_rate = self._tune_mutation_rate()
        self.generations = self._tune_generations()

        # Print tuned parameters
        print("\nTuned GA Parameters:")
        print(f"Population size: {self.population_size}")
        print(f"Elite size: {self.elite_size}")
        print(f"Mutation rate: {self.mutation_rate:.3f}")
        print(f"Generations: {self.generations}")

        # Tracking variables
        self.best_schedule = None
        self.best_fitness = float('-inf')
        self.best_cost = float('inf')
        self.avg_fitness_history = []
        self.best_fitness_history = []
        self.current_violations = {'deadline': 0, 'resource': 0}
        self.start_time = None

    def _create_output_directories(self):
        """Create directories for output files"""
        try:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.abspath(f"genetic_results_{self.timestamp}")
            self.viz_dir = os.path.abspath(os.path.join(self.output_dir, "visualizations"))

            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.viz_dir, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Error creating directories: {str(e)}")

    def optimize(self) -> Dict:
        """Run genetic algorithm optimization"""
        self.start_time = time.time()

        try:
            # Initialize population
            population = self.create_initial_population()

            # Initialize tracking of best solution
            self.best_schedule = None
            self.best_fitness = float('-inf')
            self.best_cost = float('inf')
            self.avg_fitness_history = []
            self.best_fitness_history = []

            # Save parameters for reporting
            tuned_parameters = {
                'population_size': self.population_size,
                'elite_size': self.elite_size,
                'mutation_rate': float(self.mutation_rate),
                'generations': self.generations
            }

            # Main optimization loop
            for generation in range(self.generations):
                # Evaluate fitness
                fitness_scores = []
                for schedule in population:
                    fitness = self.calculate_fitness(schedule)
                    fitness_scores.append((schedule, fitness))

                # Sort by fitness
                fitness_scores.sort(key=lambda x: x[1], reverse=True)

                # Update best solution
                if fitness_scores[0][1] > self.best_fitness:
                    self.best_fitness = fitness_scores[0][1]
                    self.best_schedule = fitness_scores[0][0].copy()
                    self.best_cost = -self.best_fitness  # Convert fitness to cost

                # Store history
                self.best_fitness_history.append(self.best_fitness)
                avg_fitness = sum(score for _, score in fitness_scores) / len(fitness_scores)
                self.avg_fitness_history.append(avg_fitness)

                # Evolution
                new_population = [schedule for schedule, _ in fitness_scores[:self.elite_size]]

                while len(new_population) < self.population_size:
                    # Tournament selection
                    parent1 = self._tournament_selection(population)
                    parent2 = self._tournament_selection(population)

                    # Crossover and mutation
                    child = self.order_crossover(parent1, parent2)
                    child = self.mutate(child)
                    new_population.append(child)

                population = new_population

                if generation % 10 == 0:
                    print(f"Generation {generation}: Best Cost = {self.best_cost:.2f}")

            # Calculate final metrics
            if self.best_schedule is None:
                raise ValueError("No valid schedule found")

            # Calculate violations for best schedule
            self.current_violations = {
                'deadline': self.count_deadline_violations(self.best_schedule),
                'resource': self.count_resource_violations(self.best_schedule)
            }

            # Calculate final schedule and makespan
            final_schedule = self._calculate_final_schedule()
            best_makespan = self.calculate_makespan(self.best_schedule)

            # Prepare result dictionary
            result = {
                'performance_metrics': {
                    'makespan': float(best_makespan),
                    'best_cost': float(self.best_cost),
                    'execution_time': float(time.time() - self.start_time),
                    'iterations': int(self.generations),
                    'violations': self.current_violations
                },
                'schedule': final_schedule,
                'algorithm_parameters': tuned_parameters,
                'problem_characteristics': {
                    'num_tasks': self.num_tasks,
                    'num_resources': len(self.global_resources),
                    'total_dependencies': sum(len(task.get('dependencies', []))
                                              for task in self.tasks)
                }
            }

            # Save results and create visualizations
            self._save_report(result)
            self.create_visualizations()

            return result

        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            error_result = {
                'performance_metrics': {
                    'makespan': float('inf'),
                    'best_cost': float('inf'),
                    'execution_time': float(time.time() - self.start_time),
                    'iterations': 0,
                    'violations': {'deadline': 0, 'resource': 0}
                },
                'schedule': [],
                'algorithm_parameters': {},
                'error': str(e)
            }
            return error_result

    def _plot_parameter_tuning(self):
        """Plot parameter tuning information"""
        try:
            plt.figure(figsize=(15, 10))
            plt.subplot(2, 2, 1)
            plt.bar(['Population Size'], [self.population_size])
            plt.title('Population Size')

            plt.subplot(2, 2, 2)
            plt.bar(['Elite Size'], [self.elite_size])
            plt.title('Elite Size')

            plt.subplot(2, 2, 3)
            plt.bar(['Mutation Rate'], [self.mutation_rate])
            plt.title('Mutation Rate')
            plt.ylim(0, 0.5)

            plt.subplot(2, 2, 4)
            plt.bar(['Generations'], [self.generations])
            plt.title('Number of Generations')

            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'parameter_tuning.png'))
            plt.close()
        except Exception as e:
            print(f"Error plotting parameter tuning: {str(e)}")

    def create_visualizations(self):
        """Generate all visualizations"""
        try:
            if not self.best_schedule:
                print("No schedule to visualize")
                return

            # Create visualization directory
            os.makedirs(self.viz_dir, exist_ok=True)

            # Generate plots
            self._plot_convergence()
            self._plot_schedule()
            self._plot_resource_utilization()
            self._plot_violations()
            self._plot_parameter_tuning()  # Add this line

            print(f"Visualizations saved in: {self.viz_dir}")

        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")

    def _plot_convergence(self):
        """Plot optimization convergence"""
        plt.figure(figsize=(12, 6))
        plt.plot([-x for x in self.best_fitness_history], 'b-', label='Best Cost')
        plt.plot([-x for x in self.avg_fitness_history], 'g-', label='Average Cost')
        plt.title('Optimization Convergence')
        plt.xlabel('Generation')
        plt.ylabel('Cost')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.viz_dir, 'convergence.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _save_report(self, result: Dict):
        """Save the analysis report"""
        try:
            report_path = os.path.join(self.output_dir, 'analysis_report.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Report saved to: {report_path}")
        except Exception as e:
            print(f"Error saving report: {str(e)}")

    def calculate_timings(self, schedule: List[int]) -> Dict:
        """Calculate detailed task timings for a schedule"""
        try:
            timings = {}
            resource_timeline = {}

            for task_idx in schedule:
                task = self.tasks[task_idx]
                start_time = self._find_start_time_for_task(task, timings, resource_timeline)
                end_time = start_time + task['processing_time']

                # Record timing
                timings[task_idx] = {
                    'start_time': float(start_time),
                    'processing_time': float(task['processing_time']),
                    'end_time': float(end_time)
                }

                # Update resource timeline
                self._update_resource_timeline(start_time, end_time, task, resource_timeline)

            return timings

        except Exception as e:
            print(f"Error calculating timings: {str(e)}")
            return {idx: {
                'start_time': 0.0,
                'processing_time': float(self.tasks[idx]['processing_time']),
                'end_time': float(self.tasks[idx]['processing_time'])
            } for idx in schedule}

    def _find_feasible_start_time(self, task: Dict, timings: Dict, resource_timeline: Dict) -> int:
        """Find earliest feasible start time for a task considering resources and dependencies"""
        try:
            # Start after dependencies
            start_time = 0
            for dep in task.get('dependencies', []):
                if dep in timings:
                    start_time = max(start_time, timings[dep]['end_time'])

            # Find time when resources are available
            while not self._is_resource_available(int(start_time), task, resource_timeline):
                start_time += 1

            return int(start_time)

        except Exception as e:
            print(f"Error finding feasible start time: {str(e)}")
            return 0

    def _is_resource_available(self, start_time: int, task: Dict, resource_timeline: Dict) -> bool:
        """Check if resources are available for task at given start time"""
        try:
            end_time = start_time + task['processing_time']

            for t in range(int(start_time), int(end_time)):
                if t not in resource_timeline:
                    resource_timeline[t] = {r: 0 for r in self.global_resources}

                for resource, amount in task['resource_requirements'].items():
                    current_usage = resource_timeline[t].get(resource, 0)
                    if current_usage + amount > self.global_resources[resource]:
                        return False
            return True

        except Exception as e:
            print(f"Error checking resource availability: {str(e)}")
            return False

    def _plot_schedule(self):
        """Create Gantt chart visualization"""
        try:
            timings = self.calculate_timings(self.best_schedule)

            plt.figure(figsize=(15, 8))

            for task_idx, timing in timings.items():
                # Plot task bar
                plt.barh(y=task_idx,
                         width=timing['processing_time'],
                         left=timing['start_time'],
                         color='skyblue',
                         alpha=0.6,
                         edgecolor='navy')

                # Add deadline marker
                plt.vlines(x=self.tasks[task_idx]['deadline'],
                           ymin=task_idx - 0.4,
                           ymax=task_idx + 0.4,
                           color='red',
                           linestyle='--',
                           alpha=0.5)

            plt.title('Schedule (Gantt Chart)')
            plt.xlabel('Time')
            plt.ylabel('Task ID')
            plt.grid(True, alpha=0.3)

            # Add legend
            plt.plot([], [], color='skyblue', label='Task Duration')
            plt.plot([], [], color='red', linestyle='--', label='Deadline')
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'schedule.png'), dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error plotting schedule: {str(e)}")

    def _calculate_final_schedule(self) -> List[Dict]:
        """Convert best schedule to standardized format"""
        try:
            if not self.best_schedule:
                return []

            timings = self.calculate_timings(self.best_schedule)
            schedule = []

            for task_idx in self.best_schedule:
                timing = timings[task_idx]
                schedule.append({
                    'task_id': task_idx,
                    'start_time': float(timing['start_time']),
                    'end_time': float(timing['end_time']),
                    'processing_time': float(timing['processing_time']),
                    'deadline': float(self.tasks[task_idx]['deadline'])
                })

            return sorted(schedule, key=lambda x: x['start_time'])

        except Exception as e:
            print(f"Error calculating final schedule: {str(e)}")
            return []

    def _update_resource_usage(self, resource_usage: Dict,
                               start_time: int, end_time: int,
                               requirements: Dict):
        """Update resource usage for a time period"""
        try:
            for t in range(start_time, end_time):
                if t not in resource_usage:
                    resource_usage[t] = {r: 0 for r in self.global_resources}
                for resource, amount in requirements.items():
                    resource_usage[t][resource] += amount

        except Exception as e:
            print(f"Error updating resource usage: {str(e)}")

    def _update_resource_timeline(self, start_time: int, end_time: int,
                                  task: Dict, timeline: Dict) -> None:
        """Update resource usage in timeline"""
        try:
            for t in range(start_time, end_time):
                if t not in timeline:
                    timeline[t] = {r: 0 for r in self.global_resources}
                for resource, amount in task['resource_requirements'].items():
                    timeline[t][resource] += amount
        except Exception as e:
            print(f"Error updating resource timeline: {str(e)}")

    def calculate_makespan(self, schedule: List[int]) -> float:
        """Calculate makespan considering dependencies and resources"""
        try:
            if not schedule:
                return float('inf')

            task_timings = {}
            resource_timeline = {}

            # Calculate end times for all tasks
            for task_idx in schedule:
                task = self.tasks[task_idx]

                # Find earliest possible start time
                start_time = 0

                # Consider dependencies
                for dep in task.get('dependencies', []):
                    if dep in task_timings:
                        start_time = max(start_time, task_timings[dep]['end_time'])

                # Find time when resources are available
                while True:
                    can_start = True
                    for resource, amount in task['resource_requirements'].items():
                        # Initialize resource timeline if needed
                        if start_time not in resource_timeline:
                            resource_timeline[start_time] = {r: 0 for r in self.global_resources}

                        if resource_timeline[start_time][resource] + amount > self.global_resources[resource]:
                            can_start = False
                            break

                    if can_start:
                        break
                    start_time += 1

                # Calculate end time and store
                end_time = start_time + task['processing_time']
                task_timings[task_idx] = {
                    'start_time': start_time,
                    'end_time': end_time
                }

                # Update resource timeline
                for t in range(start_time, end_time):
                    if t not in resource_timeline:
                        resource_timeline[t] = {r: 0 for r in self.global_resources}
                    for resource, amount in task['resource_requirements'].items():
                        resource_timeline[t][resource] += amount

            # Return makespan (maximum end time)
            return float(max(timing['end_time'] for timing in task_timings.values()))

        except Exception as e:
            print(f"Error calculating makespan: {str(e)}")
            return float('inf')

    def _find_start_time_for_task(self, task: Dict, timings: Dict,
                                  resource_timeline: Dict) -> int:
        """Find earliest feasible start time for a task"""
        try:
            # Start after all dependencies
            start_time = 0
            for dep in task.get('dependencies', []):
                if dep in timings:
                    start_time = max(start_time, int(timings[dep]['end_time']))

            # Find time when resources are available
            while not self._check_resource_availability(start_time, task, resource_timeline):
                start_time += 1

            return start_time

        except Exception as e:
            print(f"Error finding start time for task: {str(e)}")
            return 0

    def _check_resource_availability(self, start_time: int, task: Dict,
                                     resource_timeline: Dict) -> bool:
        """Check if resources are available at given time"""
        try:
            end_time = start_time + task['processing_time']

            for t in range(start_time, end_time):
                if t not in resource_timeline:
                    resource_timeline[t] = {r: 0 for r in self.global_resources}

                for resource, amount in task['resource_requirements'].items():
                    current_usage = resource_timeline[t].get(resource, 0)
                    if current_usage + amount > self.global_resources[resource]:
                        return False
            return True

        except Exception as e:
            print(f"Error checking resource availability: {str(e)}")
            return False

    def _check_resource_constraints(self, schedule: List[int]) -> int:
        """Count total resource constraint violations"""
        try:
            violations = 0
            timeline = {}

            for task_idx in schedule:
                task = self.tasks[task_idx]
                start_time = self._get_earliest_start_time(task_idx, schedule)
                end_time = start_time + task['processing_time']

                for t in range(start_time, end_time):
                    if t not in timeline:
                        timeline[t] = {r: 0 for r in self.global_resources}

                    for resource, amount in task['resource_requirements'].items():
                        timeline[t][resource] += amount
                        if timeline[t][resource] > self.global_resources[resource]:
                            violations += 1

            return violations

        except Exception as e:
            print(f"Error checking resource constraints: {str(e)}")
            return 0


    def _check_resource_feasibility(self, task: Dict, start_time: int,
                                    resource_usage: Dict) -> bool:
        """Check if task can be scheduled at the given start time"""
        try:
            end_time = start_time + task['processing_time']

            # Check each time slot in the task's duration
            for t in range(start_time, end_time):
                # Initialize resource usage for this time slot if needed
                if t not in resource_usage:
                    resource_usage[t] = {r: 0 for r in self.global_resources}

                # Check each resource requirement
                for resource, amount in task['resource_requirements'].items():
                    current_usage = resource_usage[t].get(resource, 0)
                    if current_usage + amount > self.global_resources[resource]:
                        return False

            return True

        except Exception as e:
            print(f"Error checking resource feasibility: {str(e)}")
            return False

    def _calculate_task_end_times(self, schedule: List[int]) -> Dict[int, float]:
        """Calculate end times for all tasks in schedule"""
        try:
            end_times = {}
            current_time = 0.0
            resource_usage = {r: {} for r in self.global_resources.keys()}

            for task_idx in schedule:
                task = self.tasks[task_idx]
                start_time = self._find_start_time(task_idx, end_times, resource_usage)
                end_time = start_time + float(task['processing_time'])
                end_times[task_idx] = end_time

                # Update resource usage
                for resource, amount in task['resource_requirements'].items():
                    if end_time not in resource_usage[resource]:
                        resource_usage[resource][end_time] = 0
                    resource_usage[resource][end_time] += amount

            return end_times

        except Exception as e:
            print(f"Error calculating task end times: {str(e)}")
            return {}

    def _validate_makespan(self, makespan: float, schedule: List[int]) -> bool:
        """Validate calculated makespan"""
        try:
            # Check if makespan is reasonable
            if makespan < 0:
                return False

            # Check if makespan is sufficient for all tasks
            total_processing_time = sum(float(self.tasks[i]['processing_time'])
                                        for i in schedule)
            if makespan < total_processing_time:
                return False

            # Check if makespan includes dependencies
            end_times = self._calculate_task_end_times(schedule)
            if max(end_times.values()) > makespan:
                return False

            return True

        except Exception as e:
            print(f"Error validating makespan: {str(e)}")
            return False

    def get_schedule_statistics(self) -> Dict:
        """Calculate comprehensive schedule statistics"""
        try:
            if not self.best_schedule:
                return {}

            makespan = self.calculate_makespan(self.best_schedule)
            end_times = self._calculate_task_end_times(self.best_schedule)

            stats = {
                'makespan': float(makespan),
                'total_processing_time': sum(float(self.tasks[i]['processing_time'])
                                             for i in self.best_schedule),
                'avg_completion_time': float(sum(end_times.values()) / len(end_times)),
                'max_end_time': float(max(end_times.values())),
                'min_start_time': float(min(end_times.values()) -
                                        float(self.tasks[i]['processing_time'])),
                'violations': {
                    'deadline': self.count_deadline_violations(self.best_schedule),
                    'resource': self.count_resource_violations(self.best_schedule)
                }
            }

            return stats

        except Exception as e:
            print(f"Error calculating schedule statistics: {str(e)}")
            return {}

    def _get_task_end_times(self, schedule: List[int]) -> Dict[int, int]:
        """Calculate end times for all tasks in the schedule"""
        try:
            end_times = {}
            current_time = 0
            resource_usage = {}

            for task_idx in schedule:
                task = self.tasks[task_idx]
                start_time = current_time

                # Consider dependencies
                for dep in task.get('dependencies', []):
                    if dep in end_times:
                        start_time = max(start_time, end_times[dep])

                # Find feasible start time
                while not self._check_resource_feasibility(task, start_time, resource_usage):
                    start_time += 1

                end_time = start_time + task['processing_time']
                end_times[task_idx] = end_time

                # Update resource usage
                for t in range(start_time, end_time):
                    if t not in resource_usage:
                        resource_usage[t] = {r: 0 for r in self.global_resources}
                    for resource, amount in task['resource_requirements'].items():
                        resource_usage[t][resource] += amount

            return end_times

        except Exception as e:
            print(f"Error calculating task end times: {str(e)}")
            return {}

    def _calculate_resource_timeline(self, schedule: List[int],
                                     end_times: Dict[int, int]) -> Dict:
        """Calculate resource usage timeline"""
        try:
            makespan = max(end_times.values()) if end_times else 0
            timeline = {t: {r: 0 for r in self.global_resources}
                        for t in range(makespan + 1)}

            for task_idx in schedule:
                task = self.tasks[task_idx]
                start_time = 0

                # Find start time from end time and processing time
                for dep in task.get('dependencies', []):
                    if dep in end_times:
                        start_time = max(start_time, end_times[dep])

                end_time = end_times[task_idx]

                # Update resource usage
                for t in range(start_time, end_time):
                    for resource, amount in task['resource_requirements'].items():
                        timeline[t][resource] += amount

            return timeline

        except Exception as e:
            print(f"Error calculating resource timeline: {str(e)}")
            return {}

    def _validate_schedule(self, schedule: List[int]) -> bool:
        """Validate schedule feasibility"""
        try:
            # Check for duplicates
            if len(set(schedule)) != self.num_tasks:
                return False

            # Check dependency constraints
            task_positions = {task_id: pos for pos, task_id in enumerate(schedule)}
            for task_id in schedule:
                task = self.tasks[task_id]
                for dep in task.get('dependencies', []):
                    if dep not in task_positions:
                        return False
                    if task_positions[dep] > task_positions[task_id]:
                        return False

            # Get end times
            end_times = self._get_task_end_times(schedule)
            if not end_times:
                return False

            # Check resource constraints
            timeline = self._calculate_resource_timeline(schedule, end_times)
            for t in timeline:
                for resource, amount in timeline[t].items():
                    if amount > self.global_resources[resource]:
                        return False

            return True

        except Exception as e:
            print(f"Error validating schedule: {str(e)}")
            return False


def main():
    try:
        dataset_path = os.path.join('advanced_task_scheduling_datasets',
                                    'advanced_task_scheduling_small_dataset.json')

        scheduler = GeneticScheduler(dataset_path)

        print("\nGenetic Algorithm parameters:")
        print(f"Population size: {scheduler.population_size}")
        print(f"Elite size: {scheduler.elite_size}")
        print(f"Mutation rate: {scheduler.mutation_rate}")
        print(f"Generations: {scheduler.generations}")

        print("\nStarting optimization...")
        result = scheduler.optimize()

        print("\nResults:")
        print(f"Makespan: {result['performance_metrics']['makespan']:.2f}")
        print(f"Best Cost: {result['performance_metrics']['best_cost']:.2f}")
        print(f"Execution Time: {result['performance_metrics']['execution_time']:.2f} seconds")
        print(f"Deadline Violations: {result['performance_metrics']['violations']['deadline']}")
        print(f"Resource Violations: {result['performance_metrics']['violations']['resource']}")
        print(f"\nResults and visualizations saved in: {scheduler.output_dir}")

        return result

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        return None


if __name__ == "__main__":
    main()
