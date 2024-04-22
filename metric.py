import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend


def plot_time_taken_and_steps(time_taken, steps_to_goal):
    """
    Plot the time taken and number of steps for each iteration.
    """
    num_iterations = len(time_taken)

    # creat 2 canvas/fig side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # plt time taken
    axes[0].plot(range(1, num_iterations + 1), time_taken, marker='o')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Time Taken (seconds)')
    axes[0].set_title('Time Taken for Each Iteration')

    # plt steps to goal
    axes[1].plot(range(1, num_iterations + 1), steps_to_goal, marker='o')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Steps to Goal')
    axes[1].set_title('Number of Steps Taken to Reach Goal for Each Iteration')

    plt.tight_layout()
    plt.show()
