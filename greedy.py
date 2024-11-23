import numpy as np
import json
import time
from typing import List, Dict, Tuple, Optional
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
        self.start_time = None

    def _create_output_directories(self):
        """Create directories for output files"""
        try:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.abspath(f"greedy_results_{self.timestamp}")
            self.viz_dir = os.path.join(self.output_dir, "visualizations")
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.viz_dir, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Error creating directories: {str(e)}")

    def _create_error_result(self) -> Dict:
        """Create result dictionary for error cases"""
        return {
            'performance_metrics': {
                'makespan': float('inf'),
                'best_cost': float('inf'),
                'execution_time': float(time.time() - self.start_time),
                'iterations': 0,
                'violations': {'deadline': 0, 'resource': 0}
            },
            'schedule': []
        }

    def optimize(self) -> Dict:
        """Greedy optimization with constraint checking"""
        print("Starting Greedy optimization...")
        self.start_time = time.time()

        try:
            # Initialize tracking
            schedule = []
            task_timings = {}
            resource_usage = {r: [] for r in self.global_resources}  # List of tuples for each resource

            # Get tasks with dependencies ordered
            dependency_order = self._get_dependency_order()

            # Schedule each task
            for task_id in dependency_order:
                task = self.tasks[task_id]

                # Find feasible start time
                start_time = self._find_feasible_start_time(
                    task_id,
                    task,
                    task_timings,
                    resource_usage
                )

                # If found feasible time, schedule task
                if start_time is not None:
                    end_time = start_time + task['processing_time']

                    # Record timing
                    task_timings[task_id] = {
                        'start': start_time,
                        'end': end_time
                    }

                    # Update resource usage
                    self._update_resource_usage(
                        resource_usage,
                        start_time,
                        end_time,
                        task['resource_requirements']
                    )

                    schedule.append(task_id)

            # Calculate metrics
            makespan = max(timing['end'] for timing in task_timings.values())
            deadline_violations = self._count_deadline_violations(task_timings)
            resource_violations = self._count_resource_violations(resource_usage)

            results = {
                'performance_metrics': {
                    'makespan': float(makespan),
                    'best_cost': float(makespan),
                    'execution_time': float(time.time() - self.start_time),
                    'iterations': 1,
                    'violations': {
                        'deadline': deadline_violations,
                        'resource': resource_violations
                    }
                },
                'schedule': self._create_final_schedule(schedule, task_timings)
            }

            # Save results and create visualizations
            self._save_report(results)
            self.create_visualizations(results, resource_usage)

            return results

        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            return self._create_error_result()

    def _get_dependency_order(self) -> List[int]:
        """Get tasks ordered by dependencies"""
        # Build dependency graph
        graph = {i: set(self.tasks[i].get('dependencies', []))
                 for i in range(self.num_tasks)}

        # Calculate deadlines considering dependencies
        deadlines = {}
        for i in range(self.num_tasks):
            task = self.tasks[i]
            deadline = task['deadline']
            for dep in task.get('dependencies', []):
                deadline = min(deadline, self.tasks[dep]['deadline'] - self.tasks[dep]['processing_time'])
            deadlines[i] = deadline

        # Sort by adjusted deadline
        ordered = sorted(range(self.num_tasks), key=lambda x: deadlines[x])

        return ordered

    def _find_feasible_start_time(self, task_id: int, task: Dict,
                                  task_timings: Dict, resource_usage: Dict) -> Optional[int]:
        """Find earliest feasible start time that avoids violations"""
        # Get earliest time from dependencies
        start_time = 0
        for dep in task.get('dependencies', []):
            if dep in task_timings:
                start_time = max(start_time, task_timings[dep]['end'])

        # Try times until deadline
        deadline = task['deadline']
        processing_time = task['processing_time']

        while start_time + processing_time <= deadline:
            # Check resource availability
            if self._check_resource_availability(
                    start_time,
                    start_time + processing_time,
                    task['resource_requirements'],
                    resource_usage
            ):
                return start_time

            start_time += 1

        return None

    def _check_resource_availability(self, start: int, end: int,
                                     requirements: Dict, usage: Dict) -> bool:
        """Check if resources are available in time period"""
        for resource, amount in requirements.items():
            # Get current usage timeline
            timeline = usage[resource]

            # Check each time point
            for t in range(start, end):
                current_usage = sum(amt for s, e, amt in timeline
                                    if s <= t < e)
                if current_usage + amount > self.global_resources[resource]:
                    return False

        return True

    def _update_resource_usage(self, usage: Dict, start: int, end: int, requirements: Dict):
        """Update resource usage with new task"""
        for resource, amount in requirements.items():
            usage[resource].append((start, end, amount))

    def _count_deadline_violations(self, task_timings: Dict) -> int:
        """Count number of deadline violations"""
        violations = 0
        for task_id, timing in task_timings.items():
            if timing['end'] > self.tasks[task_id]['deadline']:
                violations += 1
        return violations

    def _count_resource_violations(self, resource_usage: Dict) -> int:
        """Count number of resource violations"""
        violations = 0
        for resource, timeline in resource_usage.items():
            if not timeline:
                continue

            max_time = max(end for _, end, _ in timeline)
            for t in range(max_time):
                usage = sum(amt for start, end, amt in timeline if start <= t < end)
                if usage > self.global_resources[resource]:
                    violations += 1
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

    def create_visualizations(self, results: Dict, resource_usage: Dict):
        """Generate all visualizations"""
        try:
            schedule = results['schedule']
            if not schedule:
                print("No schedule to visualize")
                return

            self._plot_schedule(schedule)
            self._plot_deadline_violations(schedule)
            self._plot_resource_utilization(schedule, resource_usage)

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

        plt.title('Greedy Schedule (with Deadlines)')
        plt.xlabel('Time')
        plt.ylabel('Task ID')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.viz_dir, 'schedule.png'))
        plt.close()

    def _plot_deadline_violations(self, schedule: List[Dict]):
        """Plot deadline violations analysis"""
        plt.figure(figsize=(15, 6))

        violations = []
        for task in schedule:
            violation = max(0, task['end_time'] - task['deadline'])
            violations.append(violation)

        plt.bar(range(len(violations)), violations, alpha=0.6, color='salmon')
        plt.title('Deadline Violations by Task')
        plt.xlabel('Task ID')
        plt.ylabel('Time Units Overdue')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.viz_dir, 'deadline_violations.png'))
        plt.close()

    def _plot_resource_utilization(self, schedule: List[Dict], resource_usage: Dict):
        """Plot resource utilization over time"""
        try:
            makespan = max(task['end_time'] for task in schedule)
            max_time = int(makespan) + 1

            plt.figure(figsize=(15, 8))

            # Plot each resource usage
            for resource_name, timeline in resource_usage.items():
                usage = [0] * max_time
                for start, end, amount in timeline:
                    for t in range(int(start), int(end)):
                        if t < max_time:
                            usage[t] += amount

                plt.plot(usage, label=f'{resource_name} Usage', alpha=0.7)
                plt.axhline(y=self.global_resources[resource_name],
                            color='red',
                            linestyle='--',
                            alpha=0.3,
                            label=f'{resource_name} Capacity')

            plt.title('Resource Utilization Over Time')
            plt.xlabel('Time')
            plt.ylabel('Resource Usage')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.viz_dir, 'resource_utilization.png'))
            plt.close()

        except Exception as e:
            print(f"Error plotting resource utilization: {str(e)}")


def main():
    dataset_path = os.path.join('advanced_task_scheduling_datasets',
                                'advanced_task_scheduling_large_dataset.json')

    scheduler = FastGreedyScheduler(dataset_path)
    results = scheduler.optimize()

    print("\nResults:")
    print(f"Makespan: {results['performance_metrics']['makespan']:.2f}")
    print(f"Best Cost: {results['performance_metrics']['best_cost']:.2f}")
    print(f"Execution Time: {results['performance_metrics']['execution_time']:.2f} seconds")
    print(f"Deadline Violations: {results['performance_metrics']['violations']['deadline']}")
    print(f"Resource Violations: {results['performance_metrics']['violations']['resource']}")


if __name__ == "__main__":
    main()