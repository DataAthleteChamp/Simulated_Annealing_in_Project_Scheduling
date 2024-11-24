from ortools.sat.python import cp_model
import json
from time import time
from typing import List, Dict
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import traceback


class ORToolsScheduler:
    def __init__(self, dataset_path: str):
        """Initialize the OR-Tools scheduler"""
        self._load_dataset(dataset_path)
        self._initialize_tracking()
        self._create_output_directories()

    def _load_dataset(self, dataset_path: str):
        """Load and validate dataset"""
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)

        # Convert all floating point values to integers by scaling
        self.scale_factor = 100  # Scale everything by 100 to preserve 2 decimal places

        # Convert tasks
        self.tasks = []
        for task in self.dataset['tasks']:
            scaled_task = {
                'task_id': task['task_id'],
                'processing_time': int(task['processing_time']),
                'deadline': int(task['deadline'] * self.scale_factor),
                'dependencies': task.get('dependencies', []),
                'resource_requirements': {
                    resource: int(amount * self.scale_factor)
                    for resource, amount in task['resource_requirements'].items()
                }
            }
            self.tasks.append(scaled_task)

        # Convert resource capacities
        self.global_resources = {
            resource: int(capacity * self.scale_factor)
            for resource, capacity in self.dataset['dataset_metadata']['global_resources'].items()
        }

        self.num_tasks = len(self.tasks)

        print(f"Loaded {self.num_tasks} tasks and {len(self.global_resources)} resource types")
        print(f"Using scale factor of {self.scale_factor} for floating point conversion")

    def _calculate_violations(self, schedule: List[Dict]) -> Dict:
        """Calculate constraint violations"""
        violations = {'deadline': 0, 'resource': 0}

        # Calculate resource usage timeline
        timeline = {}
        step = 0.1  # Check every 0.1 time units
        max_time = max(task['end_time'] for task in schedule)

        for t in np.arange(0, max_time + step, step):
            timeline[t] = {r: 0 for r in self.global_resources}

            # Find active tasks at time t
            for task in schedule:
                if task['start_time'] <= t < task['end_time']:
                    task_id = task['task_id']
                    # Add resource usage
                    for resource, amount in self.tasks[task_id]['resource_requirements'].items():
                        timeline[t][resource] += amount
                        if timeline[t][resource] > self.global_resources[resource]:
                            violations['resource'] += 1
                            break

        # Check deadline violations
        for task in schedule:
            task_id = task['task_id']
            if task['end_time'] > self.tasks[task_id]['deadline'] / self.scale_factor:
                violations['deadline'] += 1

        return violations

    def _save_report(self, results: Dict):
        """Save results to JSON file"""
        report_path = os.path.join(self.output_dir, 'ortools_report.json')

        # Ensure directory exists
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {report_path}")

    def _calculate_deadline_statistics(self, schedule: List[Dict]) -> Dict:
        """Calculate detailed deadline statistics"""
        stats = {
            'total_violations': 0,
            'max_delay': 0.0,
            'avg_delay': 0.0,
            'violations_by_time_period': {},
            'worst_violations': []
        }

        # Calculate delays and violations
        delays = []
        for task in schedule:
            if task['end_time'] > task['deadline']:
                delay = float(task['end_time'] - task['deadline'])
                delays.append(delay)

                # Track violation
                stats['total_violations'] += 1

                # Record detailed violation info
                time_period = int(task['end_time'] // 100) * 100  # Group by 100 time units
                if time_period not in stats['violations_by_time_period']:
                    stats['violations_by_time_period'][time_period] = 0
                stats['violations_by_time_period'][time_period] += 1

                # Add to worst violations
                stats['worst_violations'].append({
                    'task_id': task['task_id'],
                    'delay': delay,
                    'end_time': float(task['end_time']),
                    'deadline': float(task['deadline']),
                    'processing_time': float(task['processing_time'])
                })

        # Calculate aggregate statistics
        if delays:
            stats['max_delay'] = float(max(delays))
            stats['avg_delay'] = float(sum(delays) / len(delays))

            # Sort and keep top 5 worst violations
            stats['worst_violations'].sort(key=lambda x: x['delay'], reverse=True)
            stats['worst_violations'] = stats['worst_violations'][:5]

        # Add percentage metrics
        stats['violation_percentage'] = (stats['total_violations'] / len(schedule)) * 100

        # Add timing distribution
        if stats['violations_by_time_period']:
            stats['peak_violation_period'] = max(
                stats['violations_by_time_period'].items(),
                key=lambda x: x[1]
            )[0]

        return stats

    def create_visualizations(self, results: Dict):
        """Create visualization plots with enhanced deadline visualization"""
        os.makedirs(self.viz_dir, exist_ok=True)
        schedule = results['schedule']

        # 1. Gantt chart with deadline violations
        plt.figure(figsize=(15, 8))

        # Plot tasks with deadline status
        for task in schedule:
            task_id = task['task_id']
            is_delayed = task['end_time'] > task['deadline']

            # Task bar
            plt.barh(y=task_id,
                     width=task['processing_time'],
                     left=task['start_time'],
                     color='red' if is_delayed else 'skyblue',
                     alpha=0.6)

            # Deadline marker
            plt.axvline(x=task['deadline'],
                        ymin=(task_id - 0.4) / self.num_tasks,
                        ymax=(task_id + 0.4) / self.num_tasks,
                        color='red',
                        linestyle='--',
                        alpha=0.7)

            # Violation indicator
            if is_delayed:
                plt.hlines(y=task_id,
                           xmin=task['deadline'],
                           xmax=task['end_time'],
                           colors='red',
                           linestyles=':',
                           alpha=0.3)

        plt.title('Task Schedule (Red: Deadline Violations)')
        plt.xlabel('Time')
        plt.ylabel('Task ID')
        plt.grid(True, alpha=0.3)

        # Add legend
        plt.plot([], [], color='skyblue', label='On Time')
        plt.plot([], [], color='red', label='Delayed')
        plt.plot([], [], color='red', linestyle='--', label='Deadline')
        plt.legend()

        plt.savefig(os.path.join(self.viz_dir, 'schedule.png'))
        plt.close()

        # 2. Resource utilization
        plt.figure(figsize=(15, 8))
        max_time = max(task['end_time'] for task in schedule)
        time_points = np.arange(0, max_time + 0.1, 0.1)

        for resource in self.global_resources:
            usage = []
            for t in time_points:
                current_usage = sum(
                    self.tasks[task['task_id']]['resource_requirements'][resource]
                    for task in schedule
                    if task['start_time'] <= t < task['end_time']
                ) / self.scale_factor
                usage.append(current_usage)

            plt.plot(time_points, usage, label=f'{resource} Usage', alpha=0.7)
            plt.axhline(y=self.global_resources[resource] / self.scale_factor,
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

        # 3. Add Deadline Violations Distribution
        plt.figure(figsize=(15, 6))
        deadline_stats = results['performance_metrics']['deadline_statistics']
        if deadline_stats['violations_by_time_period']:
            periods = sorted(deadline_stats['violations_by_time_period'].keys())
            violations = [deadline_stats['violations_by_time_period'][p] for p in periods]

            plt.bar(periods, violations, alpha=0.7)
            plt.title('Deadline Violations Distribution Over Time')
            plt.xlabel('Time Period')
            plt.ylabel('Number of Violations')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.viz_dir, 'deadline_violations.png'))
        plt.close()

        print(f"Visualizations saved in: {self.viz_dir}")

    def optimize(self) -> Dict:
        """Run optimization using OR-Tools CP-SAT solver with improved deadline handling"""
        self.start_time = time()

        try:
            print("Creating optimization model...")
            model = cp_model.CpModel()

            # Calculate horizon
            horizon = sum(task['processing_time'] for task in self.tasks)
            print(f"Time horizon: {horizon}")

            # Create variables
            start_times = []
            end_times = []
            intervals = []
            deadline_violations = []  # Track deadline violations explicitly

            print("Creating variables...")

            for i in range(self.num_tasks):
                # Create start and end time variables
                start = model.NewIntVar(0, horizon, f'start_{i}')
                end = model.NewIntVar(0, horizon, f'end_{i}')

                # Create interval variable
                interval = model.NewIntervalVar(
                    start,
                    self.tasks[i]['processing_time'],
                    end,
                    f'interval_{i}'
                )

                # Deadline violation indicator
                violation = model.NewBoolVar(f'deadline_violation_{i}')

                start_times.append(start)
                end_times.append(end)
                intervals.append(interval)
                deadline_violations.append(violation)

                # Link end times with start times and processing times
                model.Add(end_times[i] == start_times[i] + self.tasks[i]['processing_time'])

                # Deadline constraints with violation tracking
                model.Add(end_times[i] <= self.tasks[i]['deadline']).OnlyEnforceIf(violation.Not())
                model.Add(end_times[i] > self.tasks[i]['deadline']).OnlyEnforceIf(violation)

            # Create makespan variable
            makespan = model.NewIntVar(0, horizon, 'makespan')

            print("Adding constraints...")

            # Basic constraints
            for i in range(self.num_tasks):
                # Task must finish before makespan
                model.Add(end_times[i] <= makespan)

                # Dependencies
                for dep in self.tasks[i]['dependencies']:
                    model.Add(start_times[i] >= end_times[dep])

            print("Adding resource constraints...")

            # Resource constraints
            for resource, capacity in self.global_resources.items():
                demands = [task['resource_requirements'][resource] for task in self.tasks]
                model.AddCumulative(intervals, demands, capacity)

            # Objective: minimize weighted sum of makespan and deadline violations
            deadline_penalty = 1000  # High penalty for deadline violations
            model.Minimize(makespan + deadline_penalty * sum(deadline_violations))

            print("\nSolving model...")

            # Create solver with solution printer
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 300  # 5 minute timeout

            class SolutionPrinter(cp_model.CpSolverSolutionCallback):
                def __init__(self):
                    cp_model.CpSolverSolutionCallback.__init__(self)
                    self.solutions = 0

                def on_solution_callback(self):
                    self.solutions += 1
                    print(f"Found solution {self.solutions} with makespan: {self.Value(makespan)}")

            solution_printer = SolutionPrinter()

            # Solve
            status = solver.Solve(model, solution_printer)

            print(f"\nSolver status: {solver.StatusName(status)}")

            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                # Extract solution
                schedule = []
                for i in range(self.num_tasks):
                    start = solver.Value(start_times[i])
                    schedule.append({
                        'task_id': i,
                        'start_time': float(start),
                        'processing_time': float(self.tasks[i]['processing_time']),
                        'end_time': float(start + self.tasks[i]['processing_time']),
                        'deadline': float(self.tasks[i]['deadline']) / self.scale_factor
                    })

                # Sort schedule by start time
                schedule.sort(key=lambda x: x['start_time'])

                # Calculate metrics including deadline statistics
                makespan_value = float(solver.Value(makespan))
                violations = self._calculate_violations(schedule)
                deadline_stats = self._calculate_deadline_statistics(schedule)

                results = {
                    'performance_metrics': {
                        'makespan': makespan_value,
                        'best_cost': makespan_value,
                        'execution_time': float(time() - self.start_time),
                        'iterations': solution_printer.solutions,
                        'violations': violations,
                        'deadline_statistics': deadline_stats  # Added deadline statistics
                    },
                    'schedule': schedule,
                    'optimization_status': {
                        'status': solver.StatusName(status),
                        'objective_value': float(solver.ObjectiveValue())
                    }
                }

                # Save results and create visualizations
                self._save_report(results)
                self.create_visualizations(results)

                # Print deadline statistics summary
                print("\nDeadline Statistics:")
                print(f"Total violations: {deadline_stats['total_violations']}")
                print(f"Maximum delay: {deadline_stats['max_delay']:.2f}")
                print(f"Average delay: {deadline_stats['avg_delay']:.2f}")
                if deadline_stats['worst_violations']:
                    print("\nWorst Deadline Violations:")
                    for violation in deadline_stats['worst_violations']:
                        print(f"Task {violation['task_id']}: "
                              f"Delay = {violation['delay']:.2f}, "
                              f"Deadline = {violation['deadline']:.2f}")

                return results

            else:
                print("\nNo solution found!")
                print("\nAnalyzing problem...")
                self._analyze_infeasibility()
                raise ValueError("No feasible solution found!")

        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            return {
                'performance_metrics': {
                    'makespan': float('inf'),
                    'best_cost': float('inf'),
                    'execution_time': float(time() - self.start_time),
                    'iterations': 0,
                    'violations': {'deadline': 0, 'resource': 0},
                    'deadline_statistics': {
                        'total_violations': 0,
                        'max_delay': 0,
                        'avg_delay': 0,
                        'violations_by_time': {},
                        'worst_violations': []
                    }
                },
                'schedule': [],
                'error': str(e)
            }

    def _initialize_tracking(self):
        self.best_makespan = float('inf')
        self.start_time = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _create_output_directories(self):
        self.output_dir = f"ortools_results_{self.timestamp}"
        self.viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)


def main():
    dataset_path = os.path.join('advanced_task_scheduling_datasets',
                                'advanced_task_scheduling_small_dataset.json')

    print("Initializing OR-Tools Scheduler...")
    scheduler = ORToolsScheduler(dataset_path)

    print("Starting optimization...")
    results = scheduler.optimize()

    if results.get('error'):
        print("No feasible solution found!")
    else:
        print("\nOptimization completed successfully!")
        print(f"Makespan: {results['performance_metrics']['makespan']}")
        print(f"Execution time: {results['performance_metrics']['execution_time']:.2f} seconds")


if __name__ == "__main__":
    main()