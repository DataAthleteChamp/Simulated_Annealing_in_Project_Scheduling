# import numpy as np
# import json
# import time
# from typing import List, Dict
# import os
# from datetime import datetime
# import matplotlib.pyplot as plt
# from scipy.optimize import minimize, LinearConstraint, Bounds
#
#
# class OptimalTaskScheduler:
#     def __init__(self, dataset_path: str):
#         """Initialize the optimal scheduler"""
#         with open(dataset_path, 'r') as f:
#             self.dataset = json.load(f)
#
#         self.tasks = self.dataset['tasks']
#         self.num_tasks = len(self.tasks)
#         self.global_resources = self.dataset['dataset_metadata']['global_resources']
#
#         # Create output directory
#         self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         self.output_dir = f"optimal_results_{self.timestamp}"
#         self.viz_dir = os.path.join(self.output_dir, "visualizations")
#         os.makedirs(self.viz_dir, exist_ok=True)
#
#         # Track optimization progress
#         self.best_solution = None
#         self.best_cost = float('inf')
#         self.convergence_history = []
#         self.start_time = None
#
#         # Pre-process tasks
#         self._preprocess_tasks()
#
#     def _preprocess_tasks(self):
#         """Pre-process tasks to simplify optimization"""
#         # Calculate earliest possible start times based on dependencies
#         self.earliest_starts = np.zeros(self.num_tasks)
#         for i in range(self.num_tasks):
#             deps = self.tasks[i].get('dependencies', [])
#             if deps:
#                 max_dep_time = max(self.tasks[dep]['processing_time'] for dep in deps)
#                 self.earliest_starts[i] = max_dep_time
#
#     def objective_function(self, x: np.ndarray) -> float:
#         """
#         Objective function for optimization.
#         x: array of task start times and sequence values
#         """
#         # Split x into start times and sequence values
#         start_times = x[:self.num_tasks]
#         sequence_values = x[self.num_tasks:]
#
#         # Calculate task end times
#         end_times = start_times + np.array([task['processing_time'] for task in self.tasks])
#         makespan = np.max(end_times)
#
#         # Penalties
#         penalties = 0
#
#         # Deadline violations
#         deadline_violations = np.maximum(0, end_times - np.array([task['deadline'] for task in self.tasks]))
#         penalties += np.sum(deadline_violations) * 1000
#
#         # Resource violations
#         resource_usage = np.zeros(int(makespan) + 1)
#         for i in range(self.num_tasks):
#             task = self.tasks[i]
#             start = int(start_times[i])
#             end = int(end_times[i])
#             for t in range(start, end):
#                 if t < len(resource_usage):
#                     for resource, amount in task['resource_requirements'].items():
#                         current_usage = resource_usage[t]
#                         if current_usage + amount > self.global_resources[resource]:
#                             penalties += 1000
#
#         # Dependency violations
#         for i in range(self.num_tasks):
#             task = self.tasks[i]
#             for dep in task.get('dependencies', []):
#                 if start_times[i] < end_times[dep]:
#                     penalties += 5000
#
#         # Sequence violations (ensure tasks don't overlap)
#         for i in range(self.num_tasks):
#             for j in range(i + 1, self.num_tasks):
#                 if (start_times[i] < end_times[j] and end_times[i] > start_times[j]):
#                     penalties += 1000
#
#         # Priority reward
#         priority_reward = np.sum([task['priority'] * (1000 / (end_times[i] + 1))
#                                   for i, task in enumerate(self.tasks)])
#
#         total_cost = makespan + penalties - priority_reward
#
#         # Track progress
#         if total_cost < self.best_cost:
#             self.best_cost = total_cost
#             self.best_solution = start_times
#             print(f"New best cost: {total_cost:.2f}, Makespan: {makespan:.2f}")
#
#         self.convergence_history.append(total_cost)
#         return total_cost
#
#
#
#     def visualize_results(self, results: Dict):
#         """Generate visualizations"""
#         # Convergence plot
#         plt.figure(figsize=(12, 6))
#         plt.plot(self.convergence_history, 'b-')
#         plt.yscale('log')
#         plt.title('Optimization Convergence')
#         plt.xlabel('Iteration')
#         plt.ylabel('Cost (log scale)')
#         plt.grid(True)
#         plt.savefig(os.path.join(self.viz_dir, 'convergence.png'))
#         plt.close()
#
#         # Schedule visualization
#         self.visualize_schedule(results['start_times'])
#
#     def visualize_schedule(self, start_times: np.ndarray):
#         """Visualize the optimal schedule"""
#         plt.figure(figsize=(15, 8))
#
#         for i, start_time in enumerate(start_times):
#             task = self.tasks[i]
#             plt.barh(y=i,
#                      width=task['processing_time'],
#                      left=start_time,
#                      alpha=0.6)
#
#             plt.axvline(x=task['deadline'],
#                         ymin=(i - 0.4) / self.num_tasks,
#                         ymax=(i + 0.4) / self.num_tasks,
#                         color='red',
#                         linestyle='--',
#                         alpha=0.3)
#
#         plt.title('Optimal Schedule')
#         plt.xlabel('Time')
#         plt.ylabel('Task ID')
#         plt.grid(True, alpha=0.3)
#         plt.savefig(os.path.join(self.viz_dir, 'schedule.png'))
#         plt.close()
#
#     def optimize(self) -> Dict:
#         """Run optimization with fixed return types and proper makespan calculation"""
#         self.start_time = time.time()
#
#         try:
#             # Initial solution
#             x0 = np.zeros(2 * self.num_tasks)
#             for i in range(self.num_tasks):
#                 x0[i] = self.earliest_starts[i]
#                 x0[self.num_tasks + i] = i
#
#             # Bounds
#             max_time = sum(task['processing_time'] for task in self.tasks)
#             bounds = Bounds(
#                 lb=np.concatenate([self.earliest_starts, np.zeros(self.num_tasks)]),
#                 ub=np.concatenate([np.ones(self.num_tasks) * max_time,
#                                    np.ones(self.num_tasks) * self.num_tasks])
#             )
#
#             # Run optimization
#             result = minimize(
#                 self.objective_function,
#                 x0,
#                 method='SLSQP',
#                 bounds=bounds,
#                 options={
#                     'maxiter': 1000,
#                     'ftol': 1e-8,
#                     'disp': True
#                 }
#             )
#
#             # Calculate makespan and violations
#             start_times = result.x[:self.num_tasks]
#             end_times = start_times + np.array([task['processing_time'] for task in self.tasks])
#             makespan = float(np.max(end_times))
#
#             # Count violations
#             deadline_violations = sum(1 for i, task in enumerate(self.tasks)
#                                       if end_times[i] > task['deadline'])
#
#             resource_violations = 0
#             resource_usage = {r: [] for r in self.global_resources.keys()}
#             for i, task in enumerate(self.tasks):
#                 start = int(start_times[i])
#                 end = int(end_times[i])
#                 for resource, amount in task['resource_requirements'].items():
#                     for t in range(start, end):
#                         if t >= len(resource_usage[resource]):
#                             resource_usage[resource].extend([0] * (t - len(resource_usage[resource]) + 1))
#                         resource_usage[resource][t] += amount
#                         if resource_usage[resource][t] > self.global_resources[resource]:
#                             resource_violations += 1
#
#             # Prepare results in standard format
#             results = {
#                 'makespan': makespan,
#                 'execution_time': float(time.time() - self.start_time),
#                 'schedule': [
#                     {
#                         'task_id': i,
#                         'start_time': float(start_times[i]),
#                         'end_time': float(end_times[i]),
#                         'processing_time': float(self.tasks[i]['processing_time']),
#                         'deadline': float(self.tasks[i]['deadline'])
#                     }
#                     for i in range(self.num_tasks)
#                 ],
#                 'violations': {
#                     'deadline': deadline_violations,
#                     'resource': resource_violations
#                 },
#                 'optimization_success': bool(result.success),
#                 'iterations': int(result.nit),
#                 'best_cost': float(result.fun)
#             }
#
#             # Generate report and visualizations
#             self.generate_report(results)
#             self.visualize_results(results)
#
#             return results
#
#         except Exception as e:
#             print(f"Optimization error: {str(e)}")
#             # Return minimal valid results structure
#             return {
#                 'makespan': float('inf'),
#                 'execution_time': float(time.time() - self.start_time),
#                 'schedule': [],
#                 'violations': {'deadline': 0, 'resource': 0},
#                 'optimization_success': False,
#                 'iterations': 0,
#                 'best_cost': float('inf')
#             }
#
#     def generate_report(self, results: Dict):
#         """Generate analysis report with proper type conversion"""
#         report = {
#             'optimization_results': {
#                 'makespan': float(results['makespan']),
#                 'execution_time': float(results['execution_time']),
#                 'best_cost': float(results['best_cost']),
#                 'iterations': int(results['iterations']),
#                 'optimization_success': bool(results['optimization_success'])
#             },
#             'violations': {
#                 'deadline': int(results['violations']['deadline']),
#                 'resource': int(results['violations']['resource'])
#             },
#             'schedule': [
#                 {
#                     'task_id': int(task['task_id']),
#                     'start_time': float(task['start_time']),
#                     'end_time': float(task['end_time']),
#                     'deadline': float(task['deadline']),
#                     'processing_time': float(task['processing_time'])
#                 }
#                 for task in results['schedule']
#             ]
#         }
#
#         # Save report
#         report_path = os.path.join(self.output_dir, 'optimal_report.json')
#         with open(report_path, 'w') as f:
#             json.dump(report, f, indent=2)
#
# def main():
#     dataset_path = os.path.join('advanced_task_scheduling_datasets',
#                                 'advanced_task_scheduling_medium_dataset.json')
#
#     scheduler = OptimalTaskScheduler(dataset_path)
#
#     print("Starting optimal scheduling optimization...")
#     try:
#         results = scheduler.optimize()
#
#         print("\nResults:")
#         print(f"Optimal Cost: {results['cost']:.2f}")
#         print(f"Execution Time: {results['execution_time']:.2f} seconds")
#         print(f"Iterations: {results['iterations']}")
#         print(f"Optimization Success: {results['success']}")
#         print("\nCheck output directory for visualizations and report.")
#
#     except Exception as e:
#         print(f"Error during optimization: {str(e)}")
#         raise
#
#
# if __name__ == "__main__":
#     main()


import numpy as np
import json
import time
from typing import List, Dict
import os
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds


class OptimalTaskScheduler:
    def __init__(self, dataset_path: str):
        """Initialize the optimal scheduler"""
        self._load_dataset(dataset_path)
        self._initialize_tracking()
        self._create_output_directories()
        self._preprocess_tasks()

    def _load_dataset(self, dataset_path: str):
        """Load and validate the dataset"""
        try:
            with open(dataset_path, 'r') as f:
                self.dataset = json.load(f)

            self.tasks = self.dataset['tasks']
            self.num_tasks = len(self.tasks)
            self.global_resources = self.dataset['dataset_metadata']['global_resources']
        except Exception as e:
            raise ValueError(f"Error loading dataset: {str(e)}")

    def _initialize_tracking(self):
        """Initialize tracking variables"""
        self.best_solution = None
        self.best_cost = float('inf')
        self.convergence_history = []
        self.current_violations = {'deadline': 0, 'resource': 0}
        self.start_time = None

    def _create_output_directories(self):
        """Create directories for output files"""
        try:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.abspath(f"optimal_results_{self.timestamp}")
            self.viz_dir = os.path.abspath(os.path.join(self.output_dir, "visualizations"))

            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.viz_dir, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Error creating directories: {str(e)}")

    def _preprocess_tasks(self):
        """Pre-process tasks for optimization"""
        self.earliest_starts = np.zeros(self.num_tasks)
        for i in range(self.num_tasks):
            deps = self.tasks[i].get('dependencies', [])
            if deps:
                max_dep_time = max(self.tasks[dep]['processing_time'] for dep in deps)
                self.earliest_starts[i] = max_dep_time

    def optimize(self) -> Dict:
        """Run optimization to find optimal schedule"""
        self.start_time = time.time()

        try:
            # Initial solution
            x0 = np.zeros(2 * self.num_tasks)
            for i in range(self.num_tasks):
                x0[i] = self.earliest_starts[i]
                x0[self.num_tasks + i] = i

            # Bounds
            max_time = sum(task['processing_time'] for task in self.tasks)
            bounds = Bounds(
                lb=np.concatenate([self.earliest_starts, np.zeros(self.num_tasks)]),
                ub=np.concatenate([np.ones(self.num_tasks) * max_time,
                                   np.ones(self.num_tasks) * self.num_tasks])
            )

            # Run optimization
            result = minimize(
                self.objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={
                    'maxiter': 1000,
                    'ftol': 1e-8,
                    'disp': True
                }
            )

            # Extract results
            start_times = result.x[:self.num_tasks]
            end_times = start_times + np.array([task['processing_time'] for task in self.tasks])
            makespan = float(np.max(end_times))

            # Calculate violations
            self.current_violations = self._count_violations(start_times, end_times)

            # Create standardized results
            results = {
                'performance_metrics': {
                    'makespan': makespan,
                    'best_cost': float(result.fun),
                    'execution_time': float(time.time() - self.start_time),
                    'iterations': int(result.nit),
                    'violations': self.current_violations
                },
                'schedule': self._create_final_schedule(start_times, end_times),
                'algorithm_parameters': {
                    'method': 'SLSQP',
                    'max_iterations': 1000,
                    'tolerance': 1e-8,
                    'optimization_success': bool(result.success)
                }
            }

            # Save results and create visualizations
            self._save_report(results)
            self.create_visualizations(results)

            return results

        except Exception as e:
            print(f"Optimization error: {str(e)}")
            return {
                'performance_metrics': {
                    'makespan': float('inf'),
                    'best_cost': float('inf'),
                    'execution_time': float(time.time() - self.start_time),
                    'iterations': 0,
                    'violations': {'deadline': 0, 'resource': 0}
                },
                'schedule': [],
                'algorithm_parameters': {
                    'optimization_success': False
                }
            }

    def objective_function(self, x: np.ndarray) -> float:
        """Objective function for optimization"""
        # Split x into start times and sequence values
        start_times = x[:self.num_tasks]
        sequence_values = x[self.num_tasks:]

        # Calculate task end times
        end_times = start_times + np.array([task['processing_time'] for task in self.tasks])
        makespan = np.max(end_times)

        # Calculate penalties
        penalties = self._calculate_penalties(start_times, end_times)

        # Calculate priority reward
        priority_reward = np.sum([task['priority'] * (1000 / (end_times[i] + 1))
                                  for i, task in enumerate(self.tasks)])

        total_cost = makespan + penalties - priority_reward

        # Track progress
        if total_cost < self.best_cost:
            self.best_cost = total_cost
            self.best_solution = start_times.copy()
            print(f"New best cost: {total_cost:.2f}, Makespan: {makespan:.2f}")

        self.convergence_history.append(total_cost)
        return total_cost

    def _calculate_penalties(self, start_times: np.ndarray, end_times: np.ndarray) -> float:
        """Calculate constraint violation penalties"""
        penalties = 0.0

        # Deadline violations
        deadline_violations = np.maximum(0, end_times - np.array([task['deadline'] for task in self.tasks]))
        penalties += np.sum(deadline_violations) * 1000

        # Resource violations
        makespan = int(np.max(end_times))
        resource_usage = {r: np.zeros(makespan + 1) for r in self.global_resources}

        for i in range(self.num_tasks):
            task = self.tasks[i]
            start = int(start_times[i])
            end = int(end_times[i])

            for t in range(start, end):
                if t < makespan + 1:
                    for resource, amount in task['resource_requirements'].items():
                        resource_usage[resource][t] += amount
                        if resource_usage[resource][t] > self.global_resources[resource]:
                            penalties += 1000

        # Dependency violations
        for i in range(self.num_tasks):
            task = self.tasks[i]
            for dep in task.get('dependencies', []):
                if start_times[i] < end_times[dep]:
                    penalties += 5000

        return penalties

    def _count_violations(self, start_times: np.ndarray, end_times: np.ndarray) -> Dict:
        """Count deadline and resource violations"""
        violations = {'deadline': 0, 'resource': 0}

        # Deadline violations
        for i, task in enumerate(self.tasks):
            if end_times[i] > task['deadline']:
                violations['deadline'] += 1

        # Resource violations
        makespan = int(np.max(end_times))
        resource_usage = {r: np.zeros(makespan + 1) for r in self.global_resources}

        for i in range(self.num_tasks):
            task = self.tasks[i]
            start = int(start_times[i])
            end = int(end_times[i])

            for t in range(start, end):
                if t < makespan + 1:
                    for resource, amount in task['resource_requirements'].items():
                        resource_usage[resource][t] += amount
                        if resource_usage[resource][t] > self.global_resources[resource]:
                            violations['resource'] += 1

        return violations

    def _create_final_schedule(self, start_times: np.ndarray, end_times: np.ndarray) -> List[Dict]:
        """Convert solution to standardized schedule format"""
        return [
            {
                'task_id': i,
                'start_time': float(start_times[i]),
                'end_time': float(end_times[i]),
                'processing_time': float(self.tasks[i]['processing_time']),
                'deadline': float(self.tasks[i]['deadline'])
            }
            for i in range(self.num_tasks)
        ]

    def _save_report(self, results: Dict):
        """Save the analysis report"""
        try:
            report_path = os.path.join(self.output_dir, 'optimal_report.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Report saved to: {report_path}")
        except Exception as e:
            print(f"Error saving report: {str(e)}")

    def create_visualizations(self, results: Dict):
        """Generate all visualizations"""
        try:
            self._plot_convergence()
            self._plot_schedule(results['schedule'])
            self._plot_resource_utilization(results['schedule'])
            print(f"Visualizations saved in: {self.viz_dir}")
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")

    def _plot_convergence(self):
        """Plot optimization convergence"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.convergence_history, 'b-')
        plt.yscale('log')
        plt.title('Optimization Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Cost (log scale)')
        plt.grid(True)
        plt.savefig(os.path.join(self.viz_dir, 'convergence.png'))
        plt.close()

    def _plot_schedule(self, schedule: List[Dict]):
        """Create Gantt chart visualization"""
        plt.figure(figsize=(15, 8))

        for task in schedule:
            plt.barh(y=task['task_id'],
                     width=task['processing_time'],
                     left=task['start_time'],
                     alpha=0.6)

            plt.axvline(x=task['deadline'],
                        ymin=(task['task_id'] - 0.4) / self.num_tasks,
                        ymax=(task['task_id'] + 0.4) / self.num_tasks,
                        color='red',
                        linestyle='--',
                        alpha=0.3)

        plt.title('Optimal Schedule')
        plt.xlabel('Time')
        plt.ylabel('Task ID')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.viz_dir, 'schedule.png'))
        plt.close()

    def _plot_resource_utilization(self, schedule: List[Dict]):
        """Plot resource utilization over time"""
        makespan = max(task['end_time'] for task in schedule)
        timeline = {t: {r: 0 for r in self.global_resources}
                    for t in range(int(makespan) + 1)}

        # Calculate resource usage
        for task in schedule:
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
        plt.savefig(os.path.join(self.viz_dir, 'resource_utilization.png'))
        plt.close()


def main():
    dataset_path = os.path.join('advanced_task_scheduling_datasets',
                                'advanced_task_scheduling_small_dataset.json')

    scheduler = OptimalTaskScheduler(dataset_path)
    print("Starting optimal scheduling optimization...")

    results = scheduler.optimize()

    print("\nResults:")
    print(f"Makespan: {results['performance_metrics']['makespan']:.2f}")
    print(f"Best Cost: {results['performance_metrics']['best_cost']:.2f}")
    print(f"Execution Time: {results['performance_metrics']['execution_time']:.2f} seconds")
    print(f"Iterations: {results['performance_metrics']['iterations']}")
    print(f"Deadline Violations: {results['performance_metrics']['violations']['deadline']}")
    print(f"Resource Violations: {results['performance_metrics']['violations']['resource']}")
    print(f"\nResults and visualizations saved in: {scheduler.output_dir}")


if __name__ == "__main__":
    main()