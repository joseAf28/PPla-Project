import sys
import re

from typing import Dict


class Task:
    def __init__(self, name, duration, machines, resources):
        self.name = name
        self.duration = duration
        self.machines = machines
        self.resources = resources

    def __repr__(self):
        return 'Task %r with duration %d, requires machines %s and resources %s' % (self.name, self.duration, ', '.join(self.machines), ', '.join(self.resources))


class Machine:
    def __init__(self, name):
        self.name = name
        self.schedule = []

    def add_task(self, task_name, start_time, resources_used, duration):
        self.schedule.append((task_name, start_time, start_time + duration, resources_used))

    def get_schedule(self):
        return self.schedule


def parse_input(input_text):
    tasks = {}

    # Remove comments and split by lines
    lines = [line.strip() for line in input_text.split('\n') if line.strip() and not line.startswith('%')]

    # Parse tasks
    for line in lines:
        line_proc = re.sub('’', '', line)
        line_proc = re.sub("'", '', line_proc)
        line_proc = re.sub('\\)', '', line_proc)
        parts = line_proc.replace("test(", "").replace(")", "").replace("'", "").split()
        task_name = parts[0].strip(',')
        duration = int(parts[1].strip(','))
        machines = [m.strip() for m in re.sub(r'\]', '', re.sub(r'\[', '', parts[2].strip())).split(',')[:-1]] if parts[2].strip(',') != "[]" else []
        resources = [r.strip() for r in parts[3].strip(',').strip("[").strip("]").split()] if parts[3].strip(',') != "[]" else []
        tasks[task_name] = Task(task_name, duration, machines, resources)

    return tasks


def parse_output(output_text, tasks):
    machines = {}
    makespan = None

    # Split by lines
    lines = [line.strip() for line in output_text.split('\n') if line.strip()]

    # Extract makespan from the first line
    makespan = int(lines[0].split(':')[-1].strip())

    # Extract machine schedules from the rest
    for line in lines[1:]:
        # Clean and split the line into machine name and task info
        line_proc = re.sub('’', '', line)
        line_proc = re.sub("'", '', line_proc)
        parts = line_proc.replace("machine(", "").split(',', maxsplit=2)
        machine_name = parts[0].strip()
        n_tasks = int(parts[1].strip())
        tasks_data = re.split(r'\), *\(', re.sub(r'\]', '', re.sub(r'\[', '', parts[2].strip())))

        if n_tasks > 0:
            machines[machine_name] = Machine(machine_name)

            for task_data in tasks_data:
                task_data = task_data.replace("(", "").replace(")", "").replace("'", "").strip()

                # Split the task info (task_name, start_time, resources if available)
                task_info = task_data.split(',')
                task_name = task_info[0].strip()

                # Ensure start_time is correctly extracted
                if len(task_info) > 1:
                    try:
                        start_time = int(task_info[1].strip())
                    except ValueError:
                        raise ValueError(f"Invalid start time for task {task_name} on machine {machine_name}")
                else:
                    raise ValueError(f"Missing start time for task {task_name} on machine {machine_name}")

                # Extract resources if they exist
                if len(task_info) > 2:
                    resources_used = [r.strip() for r in task_info[2].strip("[]").split()] if task_info[2].strip() != '' else []
                else:
                    resources_used = []

                # Find the task's duration from the input task dictionary
                # print(tasks.keys())
                task_duration = tasks[task_name].duration
                machines[machine_name].add_task(task_name, start_time, resources_used, task_duration)

    return makespan, machines


def check_schedule(tasks, machines, given_makespan):
    used_resources = {}
    max_end_time = 0  # This will store the actual makespan
    tasks_machines: Dict[str, int] = dict([(t_name, 0) for t_name in tasks.keys()])

    # Check each machine's schedule
    for machine in machines.values():
        # If there are tasks scheduled on this machine, find the last task's end time
        if machine.schedule:
            # Check that all test starting times are valid
            times = sorted([(task[1], task[2]) for task in machine.schedule], key=lambda x: x[0])
            for i in range(len(times) - 1):
                if times[i][1] > times[i + 1][0]:
                    return False, f"Task {machine.schedule[i + 1][0]} is overlapping with task {machine.schedule[i][0]} is machine {machine.name}."

            # The last task in the schedule will have the latest start time (the schedule is already sorted by start time)
            last_task = machine.schedule[-1]  # Get the last task
            task_name, start_time, end_time, resources_used = last_task
            task_duration = tasks[task_name].duration

            # Calculate the end time of the last task (start_time + duration)
            last_task_end_time = start_time + task_duration

            # Update the max_end_time if this task's end time is later
            max_end_time = max(max_end_time, last_task_end_time)

        # Check if the task uses resources and whether they overlap in time
        for task in machine.schedule:
            task_name, start_time, end_time, resources_used = task
            tasks_machines[task_name] += 1

            # **Check if the resources used by the task are valid**
            for resource in resources_used:
                print(resource)
                if resource not in tasks[task_name].resources[0]:
                    
                    print(tasks[task_name].resources)
                    return False, f"Resource {resource} used by task {task_name} on machine {machine.name} is not a valid resource for this task."

                # **Check for resource overlap globally**
                if resource in used_resources:
                    resource_intervals = used_resources[resource]
                    for (resource_start, resource_end, task_user, machine_user) in resource_intervals:
                        if not (end_time <= resource_start or start_time >= resource_end):
                            return False, f"Resource {resource} is used by {task_name} on machine {machine.name} and overlaps with task {task_user} on machine {machine_user}."

                    # Add the current task's resource usage to the global resource tracking
                    used_resources[resource].append((start_time, end_time, task_name, machine.name))
                else:
                    # Create a new entry for the resource
                    used_resources[resource] = [(start_time, end_time, task_name, machine.name)]

            # Check if the task is assigned to valid machines
            if tasks[task_name].machines and machine.name not in tasks[task_name].machines:
                return False, f"Task {task_name} is assigned to machine {machine.name}, but it is not a valid machine for this task."

    # Validate that each test is assigned only to one machine
    if any([val > 1 for val in tasks_machines.values()]):
        return False, f"At least one task is assigned to multiple machines."

    # Validate the makespan by comparing it with the actual max end time
    if max_end_time != given_makespan:
        return False, f"Given makespan {given_makespan} does not match the actual machine makespan {max_end_time}."

    return True, "Solution is valid."


def main(input_file, output_file):
    # Read input and output files
    with open(input_file, 'r') as f:
        input_text = f.read()

    with open(output_file, 'r') as f:
        output_text = f.read()

    # Parse input and output
    tasks = parse_input(input_text)
    makespan, machines = parse_output(output_text, tasks)

    # Validate the schedule
    is_valid, message = check_schedule(tasks, machines, makespan)
    print(message)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python checker.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main(input_file, output_file)