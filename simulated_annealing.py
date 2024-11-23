import numpy as np
import matplotlib.pyplot as plt
import json
import random
import time
from typing import List, Dict, Tuple
import os
from datetime import datetime


class SimulatedAnnealingScheduler:
    def __init__(self, dataset_path: str):
        """Initialize the Simulated Annealing scheduler with dataset"""
        self._load_dataset(dataset_path)
        self._initialize_tracking()
        self._create_output_directories()

    def _load_dataset(self, dataset_path: str):
        """Load and validate the dataset"""
        try:
            with open(dataset_path, 'r') as f:
                self.dataset = json.load(f)

            self.tasks = self.dataset['tasks']
            self.num_tasks = len(self.tasks)
            self.global_resources = self.dataset['dataset_metadata']['global_resources']
            self.parallel_groups = self.dataset.get('dataset_metadata', {}).get('parallel_groups', {})

        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in dataset file")
        except KeyError as e:
            raise ValueError(f"Missing required field in dataset: {str(e)}")

    def _initialize_tracking(self):
        """Initialize tracking variables and tune parameters"""
        # Auto-tune SA parameters
        tuned_params = self._tune_parameters()
        self.initial_temp = tuned_params['initial_temperature']
        self.min_temp = tuned_params['min_temperature']
        self.alpha = tuned_params['cooling_rate']
        self.max_iterations = tuned_params['max_iterations']

        # Print tuned parameters
        print("\nTuned SA Parameters:")
        print(f"Initial Temperature: {self.initial_temp:.2f}")
        print(f"Minimum Temperature: {self.min_temp:.2f}")
        print(f"Cooling Rate: {self.alpha:.3f}")
        print(f"Max Iterations: {self.max_iterations}")

        # Initialize tracking variables
        self.best_schedule = None
        self.best_cost = float('inf')
        self.cost_history = []
        self.temperature_history = []
        self.start_time = None
        self.current_violations = {'deadline': 0, 'resource': 0}

    def _tune_parameters(self) -> Dict:
        """Tune SA parameters based on problem characteristics with focus on resource constraints"""
        try:
            # Get initial cost estimate
            initial_solution = self._create_initial_solution()
            initial_cost, initial_violations = self._calculate_cost(initial_solution)

            # Calculate problem characteristics
            total_processing_time = sum(task['processing_time'] for task in self.tasks)
            max_deadline = max(task['deadline'] for task in self.tasks)
            total_dependencies = sum(len(task.get('dependencies', []))
                                     for task in self.tasks)

            # Calculate resource complexity
            resource_complexity = 0
            for resource in self.global_resources:
                total_resource_demand = sum(task['resource_requirements'][resource]
                                            for task in self.tasks)
                capacity = self.global_resources[resource]
                # Calculate average utilization
                avg_utilization = total_resource_demand / (capacity * self.num_tasks)
                resource_complexity += avg_utilization

            resource_complexity /= len(self.global_resources)  # Normalize

            # Calculate dependency density
            max_possible_deps = self.num_tasks * (self.num_tasks - 1) / 2
            dependency_density = total_dependencies / max_possible_deps if max_possible_deps > 0 else 0

            # Calculate time complexity
            time_complexity = total_processing_time / max_deadline if max_deadline > 0 else 1

            # Base parameters by problem size
            if self.num_tasks < 200:
                base_iterations = 2000
                base_alpha = 0.99
                base_temp_factor = 0.1
            elif self.num_tasks < 1000:
                base_iterations = 5000
                base_alpha = 0.98
                base_temp_factor = 0.15
            else:
                base_iterations = 10000
                base_alpha = 0.97
                base_temp_factor = 0.2

            # Calculate overall complexity factor
            complexity_factor = (1 + dependency_density) * \
                                (1 + time_complexity) * \
                                (1 + resource_complexity * 2)  # Extra weight on resource complexity

            # Adjust parameters based on complexity
            max_iterations = int(base_iterations * np.sqrt(complexity_factor))

            # Slower cooling for more complex problems
            alpha = base_alpha ** (1 / (np.sqrt(complexity_factor) * 1.5))

            # Calculate initial temperature
            # Want to accept worse solutions with high probability early on
            avg_resource_violation_cost = initial_violations['resource'] * 1000 / self.num_tasks
            avg_deadline_violation_cost = initial_violations['deadline'] * 1000 / self.num_tasks

            # Initial temperature calculation based on violation costs
            initial_temp = max(
                1000.0,  # Minimum temperature
                max(
                    avg_resource_violation_cost,
                    avg_deadline_violation_cost,
                    initial_cost * base_temp_factor
                ) * (1 + complexity_factor)
            )

            # Adjust for resource complexity
            if resource_complexity > 0.3:  # High resource utilization
                initial_temp *= 1.5
                max_iterations = int(max_iterations * 1.5)
                alpha = alpha ** 0.95  # Slower cooling

            # Calculate minimum temperature as fraction of initial
            min_temp = initial_temp * 0.001

            # Ensure parameters are within reasonable bounds
            max_iterations = min(50000, max(1000, max_iterations))
            alpha = min(0.9999, max(0.9, alpha))

            # Print debugging information
            print("\nParameter Tuning Details:")
            print(f"Problem Characteristics:")
            print(f"- Tasks: {self.num_tasks}")
            print(f"- Resources: {len(self.global_resources)}")
            print(f"- Dependency Density: {dependency_density:.3f}")
            print(f"- Resource Complexity: {resource_complexity:.3f}")
            print(f"- Time Complexity: {time_complexity:.3f}")
            print(f"- Overall Complexity Factor: {complexity_factor:.3f}")
            print("\nTuned Parameters:")
            print(f"- Initial Temperature: {initial_temp:.2f}")
            print(f"- Minimum Temperature: {min_temp:.2f}")
            print(f"- Cooling Rate: {alpha:.4f}")
            print(f"- Maximum Iterations: {max_iterations}")

            # Store complexity metrics for reporting
            self.complexity_metrics = {
                'dependency_density': float(dependency_density),
                'resource_complexity': float(resource_complexity),
                'time_complexity': float(time_complexity),
                'overall_complexity': float(complexity_factor)
            }

            return {
                'initial_temperature': float(initial_temp),
                'min_temperature': float(min_temp),
                'cooling_rate': float(alpha),
                'max_iterations': int(max_iterations)
            }

        except Exception as e:
            print(f"Error tuning parameters: {str(e)}")
            # Return safe default parameters
            return {
                'initial_temperature': 10000.0,
                'min_temperature': 1.0,
                'cooling_rate': 0.98,
                'max_iterations': 3000
            }

    def _get_task_times(self, schedule: List[int]) -> Dict[int, Tuple[int, int]]:
        """Calculate start and end times for all tasks in schedule"""
        task_times = {}  # {task_id: (start_time, end_time)}
        current_time = 0
        resource_usage = {}

        try:
            for task_id in schedule:
                task = self.tasks[task_id]
                start_time = current_time

                # Consider dependencies
                for dep in task.get('dependencies', []):
                    if dep in task_times:
                        start_time = max(start_time, task_times[dep][1])

                # Find valid start time considering resources
                while not self._is_resource_available(start_time, task, resource_usage):
                    start_time += 1

                end_time = start_time + int(task['processing_time'])
                task_times[task_id] = (start_time, end_time)

                # Update resource usage
                for t in range(start_time, end_time):
                    if t not in resource_usage:
                        resource_usage[t] = {r: 0 for r in self.global_resources}
                    for resource, amount in task['resource_requirements'].items():
                        resource_usage[t][resource] += amount

                current_time = max(current_time, start_time + 1)

            return task_times

        except Exception as e:
            print(f"Error in _get_task_times: {str(e)}")
            return {task_id: (0, self.tasks[task_id]['processing_time'])
                    for task_id in schedule}

    def _is_resource_available(self, start_time: int, task: Dict,
                               resource_usage: Dict) -> bool:
        """Check if resources are available for task at given time"""
        try:
            end_time = start_time + task['processing_time']

            # Check resource availability for each time slot
            for t in range(start_time, end_time):
                if t not in resource_usage:
                    resource_usage[t] = {r: 0 for r in self.global_resources}

                # Check each resource requirement
                for resource, amount in task['resource_requirements'].items():
                    current_usage = resource_usage[t].get(resource, 0)
                    if current_usage + amount > self.global_resources[resource]:
                        return False

            return True

        except Exception as e:
            print(f"Error checking resource availability: {str(e)}")
            return True  # Default to available in case of error

    def _update_resource_usage(self, task: Dict, start_time: int, end_time: int,
                               resource_usage: Dict) -> None:
        """Update resource usage timeline for a task"""
        try:
            for t in range(start_time, end_time):
                if t not in resource_usage:
                    resource_usage[t] = {r: 0 for r in self.global_resources}

                for resource, amount in task['resource_requirements'].items():
                    resource_usage[t][resource] += amount

        except Exception as e:
            print(f"Error updating resource usage: {str(e)}")


    def _plot_parameter_tuning(self):
        """Plot parameter tuning information"""
        try:
            fig = plt.figure(figsize=(15, 10))

            # Temperature cooling curve
            ax1 = plt.subplot(2, 2, 1)
            iterations = np.linspace(0, self.max_iterations, 1000)
            temperatures = [self.initial_temp * (self.alpha ** i) for i in iterations]
            ax1.plot(iterations, temperatures)
            ax1.set_title('Temperature Cooling Schedule')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Temperature')
            if min(temperatures) > 0:  # Only use log scale if all values are positive
                ax1.set_yscale('log')
            ax1.grid(True)

            # Acceptance probability curve
            ax2 = plt.subplot(2, 2, 2)
            cost_ranges = [self.initial_temp * x for x in [0.01, 0.1, 0.5]]  # Sample cost differences
            for delta in cost_ranges:
                probs = [min(1.0, np.exp(-delta / max(t, 1e-10))) for t in temperatures]
                ax2.plot(iterations, probs, label=f'Δcost={delta:.0f}')
            ax2.set_title('Acceptance Probabilities')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Probability')
            ax2.legend()
            ax2.grid(True)

            # Parameters table
            ax3 = plt.subplot(2, 2, 3)
            params_data = [
                ['Parameter', 'Value'],
                ['Initial Temperature', f'{self.initial_temp:.1f}'],
                ['Minimum Temperature', f'{self.min_temp:.1f}'],
                ['Cooling Rate', f'{self.alpha:.3f}'],
                ['Max Iterations', str(self.max_iterations)]
            ]
            ax3.axis('tight')
            ax3.axis('off')
            table = ax3.table(cellText=params_data[1:],
                              colLabels=params_data[0],
                              cellLoc='center',
                              loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            ax3.set_title('Algorithm Parameters')

            # Problem characteristics
            ax4 = plt.subplot(2, 2, 4)
            total_deps = sum(len(t.get('dependencies', [])) for t in self.tasks)
            chars_data = [
                ['Characteristic', 'Value'],
                ['Tasks', str(self.num_tasks)],
                ['Resources', str(len(self.global_resources))],
                ['Dependencies', str(total_deps)],
                ['Dep. Density', f'{total_deps / (self.num_tasks ** 2):.3f}']
            ]
            ax4.axis('tight')
            ax4.axis('off')
            table = ax4.table(cellText=chars_data[1:],
                              colLabels=chars_data[0],
                              cellLoc='center',
                              loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            ax4.set_title('Problem Characteristics')

            plt.tight_layout()
            fig.savefig(os.path.join(self.viz_dir, 'parameter_tuning.png'))
            plt.close(fig)

            return True

        except Exception as e:
            print(f"Error plotting parameter tuning: {str(e)}")
            return False

    def _estimate_initial_temperature(self) -> float:
        """Estimate good initial temperature based on cost sampling"""
        try:
            # Generate sample solutions and calculate costs
            costs = []
            for _ in range(10):
                schedule = self._create_initial_solution()
                cost, _ = self._calculate_cost(schedule)
                costs.append(cost)

            # Calculate average cost difference
            cost_diffs = [abs(costs[i] - costs[i - 1]) for i in range(1, len(costs))]
            avg_diff = sum(cost_diffs) / len(cost_diffs)

            # Initial temperature should give ~80% acceptance for average cost difference
            initial_temp = -avg_diff / np.log(0.8)

            # Adjust based on problem size
            size_factor = np.log2(self.num_tasks / 100 + 1)
            initial_temp *= size_factor

            return float(initial_temp)

        except Exception as e:
            print(f"Error estimating initial temperature: {str(e)}")
            return 1000.0

    def _create_output_directories(self):
        """Create directories for output files with Streamlit compatibility"""
        try:
            # Create timestamp
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Get project root directory (where the Streamlit app is running)
            root_dir = os.path.abspath(os.getcwd())

            # Create absolute paths that work in both direct and Streamlit execution
            self.output_dir = os.path.join(root_dir, f"sa_results_{self.timestamp}")
            self.viz_dir = os.path.join(self.output_dir, "visualizations")

            # Create directories with proper permissions
            os.makedirs(self.output_dir, mode=0o777, exist_ok=True)
            os.makedirs(self.viz_dir, mode=0o777, exist_ok=True)

            print(f"\nOutput directories created:")
            print(f"Results: {self.output_dir}")
            print(f"Visualizations: {self.viz_dir}")

            # Test write permissions by creating a test file
            test_file = os.path.join(self.output_dir, 'test.txt')
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
            except Exception as e:
                print(f"Warning: Write permission test failed: {str(e)}")

            return True

        except Exception as e:
            print(f"Error creating directories: {str(e)}")
            return False

    def _plot_convergence(self):
        """Plot optimization convergence history"""
        try:
            fig = plt.figure(figsize=(15, 10))

            # Cost history
            plt.subplot(2, 1, 1)
            plt.plot(self.cost_history, 'b-', label='Current Cost', alpha=0.6)
            plt.axhline(y=self.best_cost, color='g', linestyle='--', label='Best Cost')
            plt.title('Optimization Convergence')
            plt.xlabel('Iteration')
            plt.ylabel('Cost')
            plt.yscale('log')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Temperature cooling
            plt.subplot(2, 1, 2)
            plt.plot(self.temperature_history, 'r-', label='Temperature', alpha=0.7)
            plt.title('Temperature Cooling')
            plt.xlabel('Iteration')
            plt.ylabel('Temperature')
            plt.yscale('log')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            return self._save_plot(fig, 'convergence.png')

        except Exception as e:
            print(f"Error creating convergence plot: {str(e)}")
            return False

    def _plot_gantt_chart(self):
        """Create Gantt chart of the schedule"""
        try:
            final_schedule = sorted(self._calculate_final_schedule(), key=lambda x: x['start_time'])

            fig = plt.figure(figsize=(15, 8))

            # Plot tasks
            for task in final_schedule:
                task_id = task['task_id']

                # Task bar
                plt.barh(y=task_id,
                         width=task['processing_time'],
                         left=task['start_time'],
                         color='skyblue',
                         alpha=0.6,
                         edgecolor='navy')

                # Deadline marker
                plt.vlines(x=task['deadline'],
                           ymin=task_id - 0.4,
                           ymax=task_id + 0.4,
                           color='red',
                           linestyle='--',
                           alpha=0.5)

            plt.title('Schedule Gantt Chart')
            plt.xlabel('Time')
            plt.ylabel('Task ID')
            plt.grid(True, alpha=0.3)

            # Add legend
            plt.plot([], [], color='skyblue', label='Task Duration')
            plt.plot([], [], color='red', linestyle='--', label='Deadline')
            plt.legend()

            plt.tight_layout()
            return self._save_plot(fig, 'gantt_chart.png')

        except Exception as e:
            print(f"Error creating Gantt chart: {str(e)}")
            return False

    def _plot_resource_utilization(self):
        """Plot resource utilization over time"""
        try:
            # Get schedule and makespan
            final_schedule = self._calculate_final_schedule()
            makespan = int(self._calculate_makespan(final_schedule))

            # Calculate resource usage timeline
            resource_usage = {
                t: {r: 0 for r in self.global_resources}
                for t in range(makespan + 1)
            }

            # Calculate resource usage for each task
            for task in final_schedule:
                task_id = task['task_id']
                start = int(task['start_time'])
                end = int(task['end_time'])

                for t in range(start, end):
                    for resource, amount in self.tasks[task_id]['resource_requirements'].items():
                        resource_usage[t][resource] += amount

            # Create plot
            plt.figure(figsize=(15, 8))

            # Plot each resource usage and capacity
            for resource in self.global_resources:
                # Plot usage line
                usage = [resource_usage[t][resource] for t in range(makespan + 1)]
                plt.plot(usage,
                         label=f'{resource} Usage',
                         alpha=0.7)

                # Plot capacity line
                plt.axhline(y=self.global_resources[resource],
                            color='red',
                            linestyle='--',
                            alpha=0.3,
                            label=f'{resource} Capacity')

            # Add labels and styling
            plt.title('Resource Utilization Over Time')
            plt.xlabel('Time')
            plt.ylabel('Resource Usage')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save and close
            plt.savefig(os.path.join(self.viz_dir, 'resource_utilization.png'),
                        dpi=300,
                        bbox_inches='tight')
            plt.close()

            return True  # Return True on success

        except Exception as e:
            print(f"Error plotting resource utilization: {str(e)}")
            return False  # Return False on failure

    def _plot_violations_overview(self):
        """Plot violations summary"""
        try:
            final_schedule = self._calculate_final_schedule()
            makespan = int(self._calculate_makespan(final_schedule))

            # Track violations over time
            violations = {t: {'deadline': 0, 'resource': 0} for t in range(makespan + 1)}

            # Calculate violations at each time point
            for t in range(makespan + 1):
                # Deadline violations
                for task in final_schedule:
                    if task['start_time'] <= t <= task['end_time'] and t > task['deadline']:
                        violations[t]['deadline'] += 1

                # Resource violations
                resource_usage = {r: 0 for r in self.global_resources}
                for task in final_schedule:
                    if task['start_time'] <= t < task['end_time']:
                        task_id = task['task_id']
                        for resource, amount in self.tasks[task_id]['resource_requirements'].items():
                            resource_usage[resource] += amount
                            if resource_usage[resource] > self.global_resources[resource]:
                                violations[t]['resource'] += 1

            plt.figure(figsize=(15, 6))

            times = list(violations.keys())
            deadline_violations = [v['deadline'] for v in violations.values()]
            resource_violations = [v['resource'] for v in violations.values()]

            plt.plot(times, deadline_violations, 'r-', label='Deadline Violations', alpha=0.7)
            plt.plot(times, resource_violations, 'b-', label='Resource Violations', alpha=0.7)

            plt.title('Constraint Violations Over Time')
            plt.xlabel('Time')
            plt.ylabel('Number of Violations')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'violations.png'), dpi=300, bbox_inches='tight')
            plt.close()

            return True  # Add explicit return True for success

        except Exception as e:
            print(f"Error plotting violations: {str(e)}")
            return False  # Return False on failure

    def _save_report(self, result: Dict):
        """Save optimization results with Streamlit compatibility"""
        try:
            # Ensure directory exists and is writable
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir, mode=0o777, exist_ok=True)

            # Convert all numpy and special types to Python native types
            def convert_to_native(obj):
                if isinstance(obj, (np.int_, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native(i) for i in obj]
                return obj

            # Convert result to native Python types
            result = convert_to_native(result)

            # Save report with absolute path
            report_path = os.path.join(self.output_dir, 'analysis_report.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            # Verify file was created
            if not os.path.exists(report_path):
                print(f"Warning: Report file was not created at {report_path}")
                return False

            print(f"Report saved to: {report_path}")
            return True

        except Exception as e:
            print(f"Error saving report: {str(e)}")
            return False

    def _save_plot(self, fig, filename: str) -> bool:
        """Save plot with Streamlit compatibility"""
        try:
            # Ensure visualization directory exists
            if not os.path.exists(self.viz_dir):
                os.makedirs(self.viz_dir, mode=0o777, exist_ok=True)

            # Create full file path
            filepath = os.path.join(self.viz_dir, filename)

            # Save with high quality
            fig.savefig(filepath,
                        dpi=300,
                        bbox_inches='tight',
                        pad_inches=0.1,
                        format='png')
            plt.close(fig)

            # Verify file was created
            if not os.path.exists(filepath):
                print(f"Warning: Plot file was not created at {filepath}")
                return False

            print(f"Saved plot: {filepath}")
            return True

        except Exception as e:
            print(f"Error saving plot {filename}: {str(e)}")
            return False

    def create_visualizations(self):
        """Generate all visualizations"""
        if not self.best_schedule:
            print("No optimization results to visualize")
            return

        try:
            # Ensure visualization directory exists
            os.makedirs(self.viz_dir, exist_ok=True)

            # Track each visualization separately
            visualization_results = {
                'convergence': self._plot_convergence(),
                'gantt_chart': self._plot_gantt_chart(),
                'resource_utilization': self._plot_resource_utilization(),
                'violations': self._plot_violations_overview(),
                'parameter_tuning': self._plot_parameter_tuning()
            }

            # Count successful and failed visualizations
            successful = sum(1 for result in visualization_results.values() if result)
            failed = len(visualization_results) - successful

            # Print detailed status
            print("\nVisualization Status:")
            for name, success in visualization_results.items():
                status = "✓ Success" if success else "✗ Failed"
                print(f"{name}: {status}")

            if failed == 0:
                print(f"\nAll {successful} visualizations saved successfully in: {self.viz_dir}")
            else:
                print(f"\n{successful} visualizations saved, {failed} failed")

        except Exception as e:
            print(f"Error during visualization generation: {str(e)}")

    def optimize(self) -> Dict:
        """Run optimization with comprehensive result tracking and proper file handling"""
        print("Starting optimization...")

        try:
            self.start_time = time.time()

            # Initialize optimization
            current_schedule = self._create_initial_solution()
            current_cost, current_violations = self._calculate_cost(current_schedule)

            self.best_schedule = current_schedule.copy()
            self.best_cost = current_cost
            self.current_violations = current_violations.copy()

            # Initialize tracking
            temperature = self.initial_temp
            iteration = 0
            no_improvement_counter = 0
            last_improvement_cost = self.best_cost

            # Clear history lists
            self.cost_history = []
            self.temperature_history = []

            # Statistics tracking
            accepted_moves = 0
            rejected_moves = 0
            improvement_moves = 0

            while temperature > self.min_temp and iteration < self.max_iterations:
                # Generate and evaluate neighbor
                neighbor_schedule = self._generate_neighbor(current_schedule)
                neighbor_cost, neighbor_violations = self._calculate_cost(neighbor_schedule)

                # Calculate acceptance probability
                cost_diff = neighbor_cost - current_cost
                if cost_diff < 0:
                    acceptance_probability = 1.0
                    improvement_moves += 1
                else:
                    # Scale cost_diff to prevent overflow
                    scaled_diff = min(500, cost_diff / (current_cost + 1))
                    acceptance_probability = np.exp(-scaled_diff / temperature)

                # Accept or reject neighbor
                if random.random() < acceptance_probability:
                    current_schedule = neighbor_schedule
                    current_cost = neighbor_cost
                    current_violations = neighbor_violations
                    accepted_moves += 1

                    # Update best solution if improved
                    if current_cost < self.best_cost:
                        self.best_schedule = current_schedule.copy()
                        self.best_cost = current_cost
                        self.current_violations = current_violations.copy()
                        last_improvement_cost = current_cost
                        no_improvement_counter = 0
                else:
                    rejected_moves += 1

                # Track progress
                self.cost_history.append(float(current_cost))
                self.temperature_history.append(float(temperature))

                # Check for stagnation
                if current_cost >= last_improvement_cost:
                    no_improvement_counter += 1
                else:
                    no_improvement_counter = 0
                    last_improvement_cost = current_cost

                # Apply perturbation if stuck
                if no_improvement_counter >= 1000:
                    current_schedule = self._perturb_solution(current_schedule)
                    current_cost, current_violations = self._calculate_cost(current_schedule)
                    no_improvement_counter = 0

                # Update temperature
                temperature *= self.alpha
                iteration += 1

                # Progress reporting
                if iteration % 100 == 0:
                    print(f"Iteration {iteration}, Cost: {current_cost:.2f}, Temp: {temperature:.2f}")

            # Calculate final schedule and metrics
            final_schedule = self._calculate_final_schedule()
            makespan = self._calculate_makespan(final_schedule)
            execution_time = time.time() - self.start_time

            # Prepare comprehensive results
            result = {
                'performance_metrics': {
                    'makespan': float(makespan),
                    'best_cost': float(self.best_cost),
                    'execution_time': float(execution_time),
                    'iterations': int(iteration),
                    'final_temperature': float(temperature),
                    'acceptance_rate': float(accepted_moves / max(1, accepted_moves + rejected_moves)),
                    'improvement_rate': float(improvement_moves / max(1, iteration)),
                    'violations': {
                        'deadline': int(self.current_violations['deadline']),
                        'resource': int(self.current_violations['resource'])
                    }
                },
                'optimization_params': {
                    'initial_temperature': float(self.initial_temp),
                    'final_temperature': float(temperature),
                    'cooling_rate': float(self.alpha),
                    'max_iterations': int(self.max_iterations)
                },
                'schedule': final_schedule,
                'convergence_history': {
                    'costs': [float(c) for c in self.cost_history],
                    'temperatures': [float(t) for t in self.temperature_history]
                },
                'algorithm_statistics': {
                    'accepted_moves': int(accepted_moves),
                    'rejected_moves': int(rejected_moves),
                    'improvement_moves': int(improvement_moves)
                }
            }

            # Save results and create visualizations
            try:
                # Ensure output directories exist
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir, mode=0o777, exist_ok=True)
                if not os.path.exists(self.viz_dir):
                    os.makedirs(self.viz_dir, mode=0o777, exist_ok=True)

                # Save report
                report_path = os.path.join(self.output_dir, 'analysis_report.json')
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"Report saved to: {report_path}")

                # Create visualizations
                self.create_visualizations()

                # Force sync to ensure files are written
                if hasattr(os, 'sync'):
                    os.sync()

            except Exception as e:
                print(f"Error saving results: {str(e)}")

            return result

        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            execution_time = time.time() - self.start_time

            # Return minimal valid result structure
            return {
                'performance_metrics': {
                    'makespan': float('inf'),
                    'best_cost': float('inf'),
                    'execution_time': float(execution_time),
                    'iterations': 0,
                    'violations': {'deadline': 0, 'resource': 0}
                },
                'schedule': [],
                'error': str(e)
            }

    def _perturb_solution(self, schedule: List[int]) -> List[int]:
        """Apply strong perturbation to escape local optima"""
        try:
            perturbed = schedule.copy()

            # Choose perturbation type
            perturbation = random.choice([
                'multiple_swaps',
                'block_reverse',
                'shuffle_segment'
            ])

            if perturbation == 'multiple_swaps':
                # Perform multiple random swaps
                swaps = random.randint(3, 10)
                for _ in range(swaps):
                    i, j = random.sample(range(len(perturbed)), 2)
                    perturbed[i], perturbed[j] = perturbed[j], perturbed[i]

            elif perturbation == 'block_reverse':
                # Reverse a larger block
                block_size = len(perturbed) // 4
                start = random.randint(0, len(perturbed) - block_size)
                perturbed[start:start + block_size] = \
                    reversed(perturbed[start:start + block_size])

            else:  # shuffle_segment
                # Shuffle a random segment
                segment_size = len(perturbed) // 3
                start = random.randint(0, len(perturbed) - segment_size)
                segment = perturbed[start:start + segment_size]
                random.shuffle(segment)
                perturbed[start:start + segment_size] = segment

            return perturbed

        except Exception as e:
            print(f"Error in perturbation: {str(e)}")
            return schedule.copy()

    def _print_progress(self, iteration: int, current_cost: float, temperature: float,
                        current_violations: Dict):
        """Print detailed progress information"""
        progress = (iteration / self.max_iterations) * 100
        print(f"\nIteration {iteration}/{self.max_iterations} ({progress:.1f}%)")
        print(f"Temperature: {temperature:.2f}")
        print(f"Current Cost: {current_cost:.2f}")
        print(f"Best Cost: {self.best_cost:.2f}")
        print(f"Violations:")
        print(f"  Deadline: {current_violations['deadline']}")
        print(f"  Resource: {current_violations['resource']}")

    def _create_initial_solution(self) -> List[int]:
        """Create initial solution considering task priorities and deadlines"""
        task_weights = []
        for i in range(self.num_tasks):
            task = self.tasks[i]
            weight = (
                -len(task.get('dependencies', [])),  # Fewer dependencies first
                task['deadline'],  # Earlier deadlines first
                -task['priority']  # Higher priority first
            )
            task_weights.append((i, weight))

        sorted_tasks = sorted(task_weights, key=lambda x: x[1])
        return [task_id for task_id, _ in sorted_tasks]

    def _generate_neighbor(self, schedule: List[int]) -> List[int]:
        """Generate neighbor with efficient resource balancing"""
        try:
            neighbor = schedule.copy()

            # More varied move types with weights
            move_type = random.choices(
                ['swap', 'insert', 'block_move', 'reverse'],
                weights=[0.4, 0.3, 0.2, 0.1]
            )[0]

            if move_type == 'swap':
                # Simple swap of two positions
                i, j = random.sample(range(len(neighbor)), 2)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

            elif move_type == 'insert':
                # Remove from one position and insert at another
                from_idx = random.randrange(len(neighbor))
                to_idx = random.randrange(len(neighbor))
                task = neighbor.pop(from_idx)
                neighbor.insert(to_idx, task)

            elif move_type == 'block_move':
                # Move a block of tasks
                block_size = random.randint(2, min(5, len(neighbor) // 4))
                start = random.randint(0, len(neighbor) - block_size)
                block = neighbor[start:start + block_size]
                del neighbor[start:start + block_size]
                new_start = random.randint(0, len(neighbor))
                neighbor[new_start:new_start] = block

            else:  # reverse
                # Reverse a subsection
                i, j = sorted(random.sample(range(len(neighbor)), 2))
                size = j - i + 1
                if size > 1:  # Only reverse if we have at least 2 elements
                    neighbor[i:j + 1] = reversed(neighbor[i:j + 1])

            return neighbor

        except Exception as e:
            print(f"Error generating neighbor: {str(e)}")
            return schedule.copy()

    def _quick_resource_check(self, schedule: List[int]) -> int:
        """Quick check for resource violations"""
        violations = 0
        current_usage = {r: 0 for r in self.global_resources}

        for task_id in schedule:
            task = self.tasks[task_id]
            for resource, amount in task['resource_requirements'].items():
                if current_usage[resource] + amount > self.global_resources[resource]:
                    violations += 1
                current_usage[resource] += amount

        return violations

    def _find_violation_task(self, schedule: List[int]) -> int:
        """Quickly find a task involved in resource violation"""
        current_usage = {r: 0 for r in self.global_resources}

        for task_id in schedule:
            task = self.tasks[task_id]
            for resource, amount in task['resource_requirements'].items():
                if current_usage[resource] + amount > self.global_resources[resource]:
                    return task_id
                current_usage[resource] += amount

        return None

    def _find_low_usage_time(self, resource_usage: Dict) -> int:
        """Find time period with low resource utilization"""
        try:
            min_usage_time = 0
            min_usage = float('inf')

            for time, usage in resource_usage.items():
                total_usage = sum(amount for resource, amount in usage.items())
                total_capacity = sum(self.global_resources.values())
                usage_ratio = total_usage / total_capacity if total_capacity > 0 else float('inf')

                if usage_ratio < min_usage:
                    min_usage = usage_ratio
                    min_usage_time = time

            return min_usage_time
        except Exception as e:
            print(f"Error finding low usage time: {str(e)}")
            return 0

    def _find_position_for_time(self, schedule: List[int], target_time: int) -> int:
        """Find appropriate position in schedule for target time"""
        try:
            task_times = self._get_task_times(schedule)

            # Find first position where task would start after target_time
            for i, task_id in enumerate(schedule):
                if task_times[task_id][0] > target_time:
                    return i

            return len(schedule) - 1  # Append to end if no better position found
        except Exception as e:
            print(f"Error finding position for time: {str(e)}")
            return len(schedule) - 1

    def _find_tasks_at_time(self, schedule: List[int], time: int) -> List[int]:
        """Find all tasks active at given time"""
        try:
            active_tasks = []
            task_times = self._get_task_times(schedule)

            for task_id in schedule:
                start_time, end_time = task_times[task_id]
                if start_time <= time < end_time:
                    active_tasks.append(task_id)

            return active_tasks
        except Exception as e:
            print(f"Error finding tasks at time: {str(e)}")
            return []

    def _calculate_resource_profile(self, schedule: List[int]) -> Dict:
        """Calculate complete resource usage profile"""
        try:
            # Get task timings
            task_times = self._get_task_times(schedule)
            if not task_times:
                return {}

            # Find makespan
            makespan = max(end_time for _, end_time in task_times.values())

            # Initialize profile
            profile = {t: {r: 0 for r in self.global_resources}
                       for t in range(int(makespan) + 1)}

            # Calculate resource usage
            for task_id in schedule:
                start_time, end_time = task_times[task_id]
                task = self.tasks[task_id]

                for t in range(int(start_time), int(end_time)):
                    for resource, amount in task['resource_requirements'].items():
                        profile[t][resource] += amount

            return profile
        except Exception as e:
            print(f"Error calculating resource profile: {str(e)}")
            return {}

    def _find_peak_periods(self, profile: Dict) -> List[int]:
        """Find time periods with peak resource usage"""
        try:
            peaks = []
            for time, usage in profile.items():
                for resource, amount in usage.items():
                    if amount > self.global_resources[resource] * 0.9:  # 90% threshold
                        peaks.append(time)
                        break
            return sorted(list(set(peaks)))  # Remove duplicates and sort
        except Exception as e:
            print(f"Error finding peak periods: {str(e)}")
            return []

    def _move_task_to_time(self, schedule: List[int], task_id: int, target_time: int) -> List[int]:
        """Move a task to start at target time if possible"""
        try:
            # Create new schedule
            new_schedule = schedule.copy()
            current_pos = new_schedule.index(task_id)

            # Find appropriate position
            new_pos = self._find_position_for_time(schedule, target_time)

            # Move task
            task = new_schedule.pop(current_pos)
            new_schedule.insert(new_pos, task)

            return new_schedule
        except Exception as e:
            print(f"Error moving task: {str(e)}")
            return schedule.copy()

    def _smooth_resource_usage(self, schedule: List[int]) -> List[int]:
        """Try to smooth out resource usage peaks"""
        try:
            improved_schedule = schedule.copy()
            profile = self._calculate_resource_profile(schedule)
            peaks = self._find_peak_periods(profile)

            for peak in peaks:
                tasks = self._find_tasks_at_time(improved_schedule, peak)
                for task_id in tasks:
                    # Find better time slot
                    better_time = self._find_low_usage_time(profile)
                    if better_time is not None:
                        improved_schedule = self._move_task_to_time(
                            improved_schedule, task_id, better_time)
                        # Update profile after move
                        profile = self._calculate_resource_profile(improved_schedule)

            return improved_schedule
        except Exception as e:
            print(f"Error smoothing resource usage: {str(e)}")
            return schedule.copy()

    def _calculate_cost(self, schedule: List[int]) -> Tuple[float, Dict]:
        """Calculate cost with both resource and deadline violation tracking"""
        try:
            if len(set(schedule)) != self.num_tasks:
                return float('inf'), {'deadline': self.num_tasks, 'resource': self.num_tasks}

            makespan = 0
            resource_violations = 0
            deadline_violations = 0
            current_time = 0
            task_end_times = {}
            resource_usage = {}  # {time: {resource: usage}}

            # Calculate schedule timings and violations
            for task_id in schedule:
                task = self.tasks[task_id]
                start_time = current_time

                # Consider dependencies
                for dep in task.get('dependencies', []):
                    if dep in task_end_times:
                        start_time = max(start_time, task_end_times[dep])

                end_time = start_time + task['processing_time']
                task_end_times[task_id] = end_time

                # Check deadline violation
                if end_time > task['deadline']:
                    deadline_violations += 1

                # Check resource constraints
                for t in range(start_time, end_time):
                    if t not in resource_usage:
                        resource_usage[t] = {r: 0 for r in self.global_resources}

                    for resource, amount in task['resource_requirements'].items():
                        resource_usage[t][resource] += amount
                        if resource_usage[t][resource] > self.global_resources[resource]:
                            resource_violations += 1

                makespan = max(makespan, end_time)
                current_time = min(current_time + 1, end_time)

            # Calculate cost components
            base_cost = makespan * 10
            deadline_cost = deadline_violations * 1500  # Higher penalty for deadline violations
            resource_cost = resource_violations * 1000

            total_cost = base_cost + deadline_cost + resource_cost

            violations = {
                'deadline': deadline_violations,
                'resource': resource_violations
            }

            return float(total_cost), violations

        except Exception as e:
            print(f"Error calculating cost: {str(e)}")
            return float('inf'), {'deadline': 0, 'resource': 0}

    def _calculate_resource_usage(self, schedule: List[int]) -> Dict:
        """Calculate resource usage timeline"""
        resource_usage = {}
        task_times = self._get_task_times(schedule)

        for task_id, (start, end) in task_times.items():
            for t in range(start, end):
                if t not in resource_usage:
                    resource_usage[t] = {r: 0 for r in self.global_resources}
                for resource, amount in self.tasks[task_id]['resource_requirements'].items():
                    resource_usage[t][resource] += amount

        return resource_usage

    def _find_peak_resource_times(self, resource_usage: Dict) -> List[int]:
        """Find times with high resource utilization"""
        peaks = []
        for time, usage in resource_usage.items():
            for resource, amount in usage.items():
                if amount > self.global_resources[resource] * 0.8:  # 80% threshold
                    peaks.append(time)
                    break
        return peaks



    def _calculate_final_schedule(self) -> List[Dict]:
        """Convert best schedule to detailed timing information"""
        schedule = []
        current_time = 0
        resource_usage = {}

        for task_id in self.best_schedule:
            task = self.tasks[task_id]
            start_time = int(current_time)  # Convert to int for indexing

            # Consider dependencies
            for dep in task.get('dependencies', []):
                for scheduled_task in schedule:
                    if scheduled_task['task_id'] == dep:
                        start_time = max(start_time, int(scheduled_task['end_time']))

            # Find valid start time
            while True:
                can_start = True
                if start_time not in resource_usage:
                    resource_usage[start_time] = {r: 0 for r in self.global_resources}

                for resource, amount in task['resource_requirements'].items():
                    if resource_usage[start_time][resource] + amount > self.global_resources[resource]:
                        can_start = False
                        break

                if can_start:
                    break

                start_time += 1

            end_time = start_time + int(task['processing_time'])

            # Update resource usage
            for t in range(start_time, end_time):
                if t not in resource_usage:
                    resource_usage[t] = {r: 0 for r in self.global_resources}
                for resource, amount in task['resource_requirements'].items():
                    resource_usage[t][resource] += amount

            schedule.append({
                'task_id': task_id,
                'start_time': float(start_time),
                'end_time': float(end_time),
                'processing_time': float(task['processing_time']),
                'deadline': float(task['deadline'])
            })

            current_time = min(current_time + 1, end_time)

        return schedule

    def _calculate_makespan(self, schedule: List[Dict]) -> float:
        """Calculate makespan of the final schedule"""
        if not schedule:
            return 0.0
        return max(task['end_time'] for task in schedule)

    def _save_report(self, result: Dict):
        """Save optimization results to file"""
        report_path = os.path.join(self.output_dir, 'analysis_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


def main():
    try:
        dataset_path = os.path.join('advanced_task_scheduling_datasets',
                                    'advanced_task_scheduling_small_dataset.json')

        scheduler = SimulatedAnnealingScheduler(dataset_path)
        result = scheduler.optimize()

        if 'performance_metrics' in result:
            metrics = result['performance_metrics']
            print("\nResults:")
            print(f"Makespan: {metrics['makespan']:.2f}")
            print(f"Execution time: {metrics['execution_time']:.2f} seconds")
            print(f"Deadline violations: {metrics['violations']['deadline']}")
            print(f"Resource violations: {metrics['violations']['resource']}")

            # Calculate and show violation percentages
            total_tasks = len(scheduler.tasks)
            print(f"\nViolation Percentages:")
            print(f"Deadline violations: {(metrics['violations']['deadline'] / total_tasks) * 100:.2f}%")
            print(
                f"Resource violations: {(metrics['violations']['resource'] / (total_tasks * len(scheduler.global_resources))) * 100:.2f}%")
        else:
            print("Error: Invalid result format")

        return result

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        return None

if __name__ == "__main__":
    main()