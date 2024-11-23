import numpy as np
import matplotlib.pyplot as plt


def visualize_scheduling_example():
    """Visualize different scheduling approaches for the same tasks"""
    # Example tasks
    tasks = [
        {'id': 1, 'time': 4, 'resources': {'cpu': 2, 'memory': 4}},
        {'id': 2, 'time': 3, 'resources': {'cpu': 1, 'memory': 2}},
        {'id': 3, 'time': 5, 'resources': {'cpu': 3, 'memory': 3}},
        {'id': 4, 'time': 2, 'resources': {'cpu': 1, 'memory': 1}}
    ]

    # Available resources
    resources = {'cpu': 4, 'memory': 8}

    # Create figure for different scheduling approaches
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

    # 1. Sequential Scheduling (like our current implementation)
    current_time = 0
    for task in tasks:
        ax1.barh(y=task['id'],
                 width=task['time'],
                 left=current_time,
                 alpha=0.6,
                 label=f'Task {task["id"]}')
        current_time += task['time']
    ax1.set_title('Sequential Scheduling (Current Approach)\nMakespan = 14')

    # 2. Parallel Scheduling - Greedy Approach
    schedule_greedy = {
        1: 0,  # Task 1 starts at time 0
        2: 0,  # Task 2 can start with Task 1 (enough resources)
        3: 3,  # Task 3 must wait for Task 2
        4: 4  # Task 4 can start during Task 3
    }

    for task in tasks:
        ax2.barh(y=task['id'],
                 width=task['time'],
                 left=schedule_greedy[task['id']],
                 alpha=0.6,
                 label=f'Task {task["id"]}')
    ax2.set_title('Parallel Scheduling - Greedy Approach\nMakespan = 8')

    # 3. Parallel Scheduling - Optimal Approach
    schedule_optimal = {
        1: 0,  # Task 1 starts at time 0
        2: 2,  # Task 2 starts at time 2
        3: 0,  # Task 3 starts with Task 1
        4: 5  # Task 4 starts after Task 3
    }

    for task in tasks:
        ax3.barh(y=task['id'],
                 width=task['time'],
                 left=schedule_optimal[task['id']],
                 alpha=0.6,
                 label=f'Task {task["id"]}')
    ax3.set_title('Parallel Scheduling - Optimal Approach\nMakespan = 7')

    # Add resource usage plots
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('Time')
        ax.set_ylabel('Task ID')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Example of how resource tracking works
def check_resource_availability(time_slot: int,
                                active_tasks: list,
                                new_task: dict,
                                resources: dict) -> bool:
    """Check if resources are available for a new task"""
    # Calculate current resource usage
    usage = {r: 0 for r in resources}
    for task in active_tasks:
        for resource, amount in task['resources'].items():
            usage[resource] += amount

    # Check if adding new task exceeds limits
    for resource, amount in new_task['resources'].items():
        if usage[resource] + amount > resources[resource]:
            return False
    return True


# Example of different algorithm approaches
def demonstrate_scheduling_approaches():
    """Show how different algorithms approach parallel scheduling"""
    print("Example Task Set:")
    print("Task 1: 4 time units, needs 2 CPU, 4 Memory")
    print("Task 2: 3 time units, needs 1 CPU, 2 Memory")
    print("Task 3: 5 time units, needs 3 CPU, 3 Memory")
    print("Task 4: 2 time units, needs 1 CPU, 1 Memory")
    print("\nAvailable Resources: 4 CPU, 8 Memory")

    print("\n1. Sequential (Current) Approach:")
    print("- Simply adds up all processing times")
    print("- Makespan = 4 + 3 + 5 + 2 = 14 units")

    print("\n2. Greedy Parallel Approach:")
    print("- Task 1 starts at 0")
    print("- Task 2 can start with Task 1 (resources available)")
    print("  CPU: 2+1=3 ≤ 4, Memory: 4+2=6 ≤ 8")
    print("- Task 3 must wait (not enough CPU)")
    print("- Task 4 can overlap with Task 3")
    print("- Makespan = 8 units")

    print("\n3. Optimal Parallel Approach:")
    print("- Tasks 1 and 3 start together")
    print("  CPU: 2+3=5 > 4 (wouldn't work in greedy)")
    print("- Task 2 starts after Task 1")
    print("- Task 4 starts after Task 3")
    print("- Makespan = 7 units")


visualize_scheduling_example()
demonstrate_scheduling_approaches()