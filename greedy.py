import numpy as np
import json
import time
from typing import List, Dict
import os
from datetime import datetime
import matplotlib.pyplot as plt


class FastGreedyScheduler:
    def __init__(self, dataset_path: str):
        """Initialize the greedy scheduler"""
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
        except Exception as e:
            raise ValueError(f"Error loading dataset: {str(e)}")

    def _initialize_tracking(self):
        """Initialize tracking variables"""
        self.best_schedule = None
        self.best_cost = float('inf')
        self.current_violations = {'deadline': 0, 'resource': 0}
        self.start_time = None

    def _create_output_directories(self):
        """Create directories for output files"""
        try:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.abspath(f"greedy_results_{self.timestamp}")
            self.viz_dir = os.path.abspath(os.path.join(self.output_dir, "visualizations"))

            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.viz_dir, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Error creating directories: {str(e)}")

    def optimize(self) -> Dict:
        """Fast greedy optimization"""
        self.start_time = time.time()

        try:
            # Initialize tracking
            schedule = []
            task_timings = {}
            current_time = 0

            # Create task priority list
            task_priorities = []
            for i in range(self.num_tasks):
                task = self.tasks[i]
                score = (
                    len(task.get('dependencies', [])),  # Fewer dependencies better
                    task['deadline'],  # Earlier deadline better
                    -task['priority']  # Higher priority better
                )
                task_priorities.append((i, score))

            # Sort tasks by priority
            sorted_tasks = sorted(task_priorities, key=lambda x: x[1])

            # Schedule tasks
            resource_usage = {r: [] for r in self.global_resources.keys()}
            for task_id, _ in sorted_tasks:
                task = self.tasks[task_id]

                # Find earliest feasible start time
                start_time = self._find_start_time(task_id, task_timings, resource_usage)
                end_time = start_time + task['processing_time']

                # Update timings
                task_timings[task_id] = {
                    'start': start_time,
                    'end': end_time
                }

                # Update resource usage
                self._update_resource_usage(resource_usage, start_time, end_time, task)

                schedule.append(task_id)
                current_time = start_time

            # Calculate final metrics
            makespan = max(timing['end'] for timing in task_timings.values())
            self.current_violations = self._count_violations(schedule, task_timings)

            # Create standardized results
            results = {
                'performance_metrics': {
                    'makespan': float(makespan),
                    'best_cost': float(makespan + sum(self.current_violations.values()) * 1000),
                    'execution_time': float(time.time() - self.start_time),
                    'iterations': 1,
                    'violations': self.current_violations
                },
                'schedule': self._create_final_schedule(schedule, task_timings),
                'algorithm_parameters': {
                    'method': 'greedy',
                    'priority_weights': {
                        'dependencies': 1,
                        'deadline': 1,
                        'priority': 1
                    }
                }
            }

            # Save results and generate visualizations
            self._save_report(results)
            self.create_visualizations(results)

            return results

        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            return {
                'performance_metrics': {
                    'makespan': float('inf'),
                    'best_cost': float('inf'),
                    'execution_time': float(time.time() - self.start_time),
                    'iterations': 0,
                    'violations': {'deadline': 0, 'resource': 0}
                },
                'schedule': [],
                'error': str(e)
            }

    def _find_start_time(self, task_id: int, task_timings: Dict, resource_usage: Dict) -> int:
        """Find earliest feasible start time for a task"""
        task = self.tasks[task_id]
        start_time = 0

        # Consider dependencies
        for dep in task.get('dependencies', []):
            if dep in task_timings:
                start_time = max(start_time, task_timings[dep]['end'])

        # Find time when resources are available
        while not self._check_resource_availability(start_time, task, resource_usage):
            start_time += 1

        return start_time

    def _check_resource_availability(self, start_time: int, task: Dict, resource_usage: Dict) -> bool:
        """Check if resources are available at given time"""
        end_time = start_time + task['processing_time']

        for resource, usage_list in resource_usage.items():
            # Remove completed tasks
            usage_list = [u for u in usage_list if u[1] > start_time]

            # Check resource availability
            for t in range(start_time, end_time):
                current_usage = sum(amount for s, e, amount in usage_list if s <= t < e)
                if current_usage + task['resource_requirements'].get(resource, 0) > self.global_resources[resource]:
                    return False

        return True

    def _update_resource_usage(self, resource_usage: Dict, start_time: int, end_time: int, task: Dict):
        """Update resource usage tracking"""
        for resource, amount in task['resource_requirements'].items():
            if resource not in resource_usage:
                resource_usage[resource] = []
            resource_usage[resource].append((start_time, end_time, amount))
            resource_usage[resource].sort()  # Keep sorted by start time

    def _count_violations(self, schedule: List[int], timings: Dict) -> Dict:
        """Count deadline and resource violations"""
        violations = {'deadline': 0, 'resource': 0}

        # Create resource usage timeline
        makespan = max(timing['end'] for timing in timings.values())
        resource_timeline = {t: {r: 0 for r in self.global_resources}
                             for t in range(int(makespan) + 1)}

        # Fill resource timeline
        for task_id in schedule:
            task = self.tasks[task_id]
            timing = timings[task_id]

            # Deadline violations
            if timing['end'] > task['deadline']:
                violations['deadline'] += 1

            # Calculate resource usage
            for t in range(int(timing['start']), int(timing['end'])):
                for resource, amount in task['resource_requirements'].items():
                    resource_timeline[t][resource] += amount
                    # Check if usage exceeds capacity
                    if resource_timeline[t][resource] > self.global_resources[resource]:
                        violations['resource'] += 1

        return violations

    def _create_final_schedule(self, schedule: List[int], timings: Dict) -> List[Dict]:
        """Convert schedule to standardized format"""
        return [
            {
                'task_id': task_id,
                'start_time': float(timings[task_id]['start']),
                'end_time': float(timings[task_id]['end']),
                'processing_time': float(self.tasks[task_id]['processing_time']),
                'deadline': float(self.tasks[task_id]['deadline'])
            }
            for task_id in schedule
        ]

    def _save_report(self, results: Dict):
        """Save the analysis report"""
        try:
            report_path = os.path.join(self.output_dir, 'greedy_report.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Report saved to: {report_path}")
        except Exception as e:
            print(f"Error saving report: {str(e)}")

    def create_visualizations(self, results: Dict):
        """Generate all visualizations"""
        try:
            schedule = results['schedule']
            if not schedule:
                print("No schedule to visualize")
                return

            self._plot_schedule(schedule)
            self._plot_resource_utilization(schedule)
            print(f"Visualizations saved in: {self.viz_dir}")

        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")

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

        plt.title('Greedy Schedule')
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

    scheduler = FastGreedyScheduler(dataset_path)
    print("Starting Greedy optimization...")

    results = scheduler.optimize()

    print("\nResults:")
    print(f"Makespan: {results['performance_metrics']['makespan']:.2f}")
    print(f"Best Cost: {results['performance_metrics']['best_cost']:.2f}")
    print(f"Execution Time: {results['performance_metrics']['execution_time']:.2f} seconds")
    print(f"Deadline Violations: {results['performance_metrics']['violations']['deadline']}")
    print(f"Resource Violations: {results['performance_metrics']['violations']['resource']}")
    print(f"\nResults and visualizations saved in: {scheduler.output_dir}")


if __name__ == "__main__":
    main()