import sys
import re
from minizinc import Instance, Model, Solver
import time
import itertools
import heapq
from itertools import groupby
from operator import itemgetter

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



class Problem:
    
    def __init__(self, input_file_name, output_file_name):
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        
        
    @staticmethod
    def parse_instance():
        # Check if the correct number of arguments is provided
        if len(sys.argv) != 3:
            print(f"Usage: python3.11 {sys.argv[0]} <input-file-name> <output-file-name>")
            sys.exit(1)

        # Get the input and output file names from the command-line arguments
        input_file_name = sys.argv[1]
        output_file_name = sys.argv[2]
        
        return Problem(input_file_name, output_file_name)
    
    
    @staticmethod
    def create_pattern_read(input_string):
        
        pattern = r"\s*'([^']*)'\s*|\s*(\d+)\s*|\[([^\]]*)\]"
        ## elements 'tn' or number or [ elements ]
        
        matches = re.findall(pattern, input_string)
        
        arguments = []
        for match in matches:
            if match[0]:         # string argument
                arguments.append(match[0])
            elif match[1]:      # integer argument
                arguments.append(int(match[1]))
            elif match[2]:      # list argument
                list_items = [item.strip().strip("'") for item in match[2].split(",")]
                arguments.append(list_items)
            else:               # empty argument
                arguments.append(['e'])
                
        return arguments
    
    
    
    def read_input_data(self):
        
        with open(self.input_file_name, 'r') as input_file:
            content = [line.rstrip() for line in input_file]
            
        commented_data = []
        test_data = []
        
        for line in content:
            if line.startswith("%"):
                commented_data.append(line)
            elif line.startswith('test('):
                test_data.append(line)
            else:
                print("Error: data not in correct format")
                
        
        commented_values = [int(commented_data[i].split(" ")[-1]) for i in range(len(commented_data))]
        tests_values = [Problem.create_pattern_read(test) for test in test_data]
        
        ## store data for the model
        self.num_tests = commented_values[0]
        self.num_machines = commented_values[1]
        self.num_resources = commented_values[2]
        
        self.durations = [tests_values[i][1] for i in range(len(tests_values))]
        
        self.machines = [tests_values[i][2] for i in range(len(tests_values))]
        self.resources = [tests_values[i][3] for i in range(len(tests_values))]
    
    
    ## function not used by now
    @staticmethod
    def filter_sets(array_of_sets):
        # Remove duplicates by converting sets to frozensets and using a set comprehension
        unique_sets = list(set(frozenset(s) for s in array_of_sets))

        # Sort sets by length in descending order to handle larger sets first
        unique_sets.sort(key=len, reverse=True)

        filtered_sets = []
        seen_sets = set()  # To keep track of sets we have already added

        for s in unique_sets:
            # Check if the current set `s` is a subset of any set already in `filtered_sets`
            if not any(s < existing_set for existing_set in filtered_sets):
                filtered_sets.append(s)
                seen_sets.add(s)

        # Convert frozensets back to regular sets for the final result
        return [set(s) for s in filtered_sets]


    ## function not used by now
    @staticmethod
    def common_elements_sets(sets_array):
        
        # Check all pairs of sets
        # Variable to store the maximum size of the intersection
        max_intersection_size = 0

        # Iterate over all combinations of the sets (at least 2 sets)
        for r in range(2, len(sets_array) + 1):
            for comb in itertools.combinations(sets_array, r):
                # Find the intersection of the current combination of sets
                intersection = set.intersection(*comb)
                # Update the maximum size if the current intersection is larger
                max_intersection_size = max(max_intersection_size, len(intersection))

        
        return intersection
    
    
    def input_data_modelA(self):
        
        ##! Input data for all the tests
        dictionaire_machines = {f"m{i+1}": i+1 for i in range(self.num_machines)}
        setAllMachines = {i+1 for i in range(self.num_machines)}
        
        self.machines_allowed = []
        for i in range(self.num_tests):
            if self.machines[i] == ['e']:
                self.machines_allowed.append(setAllMachines) ## probably this ones is default, no need to add it
            else:
                self.machines_allowed.append({dictionaire_machines[machine] for machine in self.machines[i]})
        
        
        self.resources_allowed = [ [] for _ in range(self.num_resources)]
        self.have_resources = [False for _ in range(self.num_tests)]
        
        for i in range(self.num_tests):
            for j in range(self.num_resources):
                if f'r{j+1}' in self.resources[i]:
                    self.resources_allowed[j].append(i+1)
                    self.have_resources[i] = True
        
        
        ###! Input data for model A
        ## It includes only the tests that have resources despite it has restrictions on the machines or not
        
        self.tests_modelA = [i+1 for i in range(self.num_tests) if self.have_resources[i] == True]
        self.num_tests_modelA = len(self.tests_modelA)
        
        ## convert the tests to a dictionary to have the index of the test (to start from 1)
        dictionaire_tests_modelA = {self.tests_modelA[i]:i+1  for i in range(self.num_tests_modelA)}
        
        self.durations_modelA = [self.durations[i-1] for i in self.tests_modelA]
        
        self.machines_allowed_modelA = [self.machines_allowed[i-1] for i in self.tests_modelA]
        resources_allowed_modelA = [{dictionaire_tests_modelA[j] for j in self.resources_allowed[i]} for i in range(self.num_resources)]
        self.resources_effective_modelA = Problem.filter_sets(resources_allowed_modelA)
        
        self.num_resources_effective = len(self.resources_effective_modelA)
        self.have_resources_modelA = [self.have_resources[i-1] for i in self.tests_modelA]
        
        
        ##! machine pre-assigned
        ##! not very smart yet
        
        print()
        print("effective resources: ", self.resources_effective_modelA)
        print()
        
        ## assign first tests with machines restrictions
        counter_machine_assignment = [0 for _ in range(self.num_machines)]
        
        self.machines_pre_assigned = [0 for _ in range(self.num_tests_modelA)]
        
        for i in range(self.num_tests_modelA):
            if len(self.machines_allowed_modelA[i]) < self.num_machines:
                (self.machines_pre_assigned[i]) = list(self.machines_allowed_modelA[i])[0]
        
        
        machine_value = 1
        
        for i in range(self.num_tests_modelA):
            if len(self.machines_allowed_modelA[i]) == self.num_machines:
                (self.machines_pre_assigned[i]) = machine_value
                
                machine_value += 1
                if machine_value > self.num_machines:
                    machine_value = 1
        
        
        ##! assign values for the start time in the first resource
        
        lenghts_resources = [len(self.resources_effective_modelA[i]) for i in range(self.num_resources_effective)]
        index_max_resource = lenghts_resources.index(max(lenghts_resources))
        
        tests_resource = list(self.resources_effective_modelA[index_max_resource])
        
        
        self.start_pre_assigned = [0 for _ in range(self.num_tests_modelA)]
        
        self.start_pre_assigned[tests_resource[0]-1] = 0
        self.start_pre_assigned[tests_resource[1]-1] = self.durations_modelA[tests_resource[0]-1]
        
        old_task = tests_resource[1]
        for i, task in enumerate(tests_resource):
            if i > 1:
                self.start_pre_assigned[task-1] = self.durations_modelA[task-2] + self.start_pre_assigned[old_task-1] 
                
                old_task = task
        
        for i in range(self.num_tests_modelA):
            if self.start_pre_assigned[i] == 0:
                self.start_pre_assigned[i] = -2
                
        self.start_pre_assigned[tests_resource[0]-1] = 0
        
        print()
        print("durations: ", self.durations_modelA)
        print("start pre-assigned: ", self.start_pre_assigned)
        print()
        
        
        self.new_tasks_modelA = [i+1 for i in range(self.num_tests_modelA) if self.start_pre_assigned[i] == -2]
        self.num_new_tasks_modelA = len(self.new_tasks_modelA)
        self.new_duarations_modelA = [self.durations_modelA[i-1] for i in self.new_tasks_modelA]
        self.machines_pre_assigned_A = [self.machines_pre_assigned[i-1] for i in self.new_tasks_modelA]
        self.resources_effective_modelA_A = [self.resources_effective_modelA[i] for i in range(self.num_resources_effective) if i != index_max_resource]
        
        
        end_pre_assigned = [self.start_pre_assigned[i] + self.durations_modelA[i] if self.start_pre_assigned[i] > -2 else -2 for i in range(self.num_tests_modelA)]
        
        start_pre_assigned_2 = [self.start_pre_assigned[i] for i in range(self.num_tests_modelA) if self.start_pre_assigned[i] > -2]
        end_pre_assigned_2 = [end_pre_assigned[i] for i in range(self.num_tests_modelA) if self.start_pre_assigned[i] > -2]
        
        max_number = sum(self.durations)
        print(max_number)
        
        gaps_start = list()
        
        for i in range(len(start_pre_assigned_2)-1):
            if start_pre_assigned_2[i+1] - end_pre_assigned_2[i] > 0:
                gaps_start.append([end_pre_assigned_2[i], start_pre_assigned_2[i+1]])
                
        if start_pre_assigned_2[0] > 0:
            gaps_start.append([0, start_pre_assigned_2[0]])
            
        if end_pre_assigned_2[-1] < max_number:
            gaps_start.append([end_pre_assigned_2[-1], max_number])
        
        print()
        print("start pre-assigned: ", start_pre_assigned_2)
        print("end pre-assigned: ", end_pre_assigned_2)
        print("gaps: ", gaps_start)
        print()
        
    
        
        
        
        # ##! offset in the global resources in the machines that have only one machine allowed - have that into account when checking the global resources
        # self.tests_unique_machines_no_resources = [i+1 for i in range(self.num_tests) if (self.have_resources[i] == False and len(self.machines_allowed[i]) == 1)]
        
        # self.offset_machine = [set() for _ in range(self.num_machines)]
        # for test in self.tests_unique_machines_no_resources:
        #     self.offset_machine[list(self.machines_allowed[test-1])[0]-1].add(self.durations[test-1])
        
        # self.offset_machine = [sum(offset) for offset in self.offset_machine]
        # self.offset_which_machine = [i+1 if self.offset_machine[i] > 0 else 0 for i in range(self.num_machines)]
        
        # ##!? describe the case where there is at at least with unique restricton for each machine
        # ##!? by default the offset from the last machine is zero
        # if len(self.offset_which_machine) == self.num_machines:
        #     self.offset_which_machine[-1] = 0
        
        
        # ##!? Idea - restrict the number of machines to the number of tests for the global resources
        # ##!? seems not improve the efficiency

        # self.set_non_common_machines = set()
        # for i in range(self.num_tests_modelA):
        #     if len(self.machines_allowed_modelA[i]) < self.num_machines:
        #         self.set_non_common_machines.update(self.machines_allowed_modelA[i])
        
        # set_diff_elements = setAllMachines - self.set_non_common_machines
        
        # while len(self.set_non_common_machines) < self.num_resources:
        #     self.set_non_common_machines.add(set_diff_elements.pop())
        
        # self.machines_effective_allowed_modelA = [self.set_non_common_machines if len(self.machines_allowed_modelA[i]) == len(setAllMachines) else self.machines_allowed_modelA[i] for i in range(self.num_tests_modelA)]
        
        
        ##! Ideas that seem not to improve the efficiency, but worth to keep in mind and explore later
        # self.test_resources_modelA_no_restriction_machines = set(i+1 for i in range(self.num_tests_modelA) if len(self.machines_allowed_modelA[i]) == self.num_machines)
        # print(self.test_resources_modelA_no_restriction_machines)
        
        # self.tests_no_model_A = [i+1 for i in range(self.num_tests) if i+1 not in self.tests_modelA]
        
        # self.tests_restriction_machine_no_model_A = [i+1 for i in range(self.num_tests) if (i+1 in self.tests_no_model_A and len(self.machines_allowed[i]) != self.num_machines)]
        # self.machines_restriction_no_model_A = set.union(*[self.machines_allowed[i-1] for i in self.tests_restriction_machine_no_model_A])
        # self.machines_effective_per_resource = [[self.machines_allowed_modelA[i-1] for i in self.resources_effective_modelA[j]] for j in range(self.num_resources_effective)]
        
        # # print(self.machines_effective_per_resource[0])
        
        # self.machines_effective_per_resource_common = [Problem.common_elements_sets(self.machines_effective_per_resource[i]) for i in range(self.num_resources_effective)]
        
        # union_machines_effective_per_resource_common = set.union(*self.machines_effective_per_resource_common) - self.machines_restriction_no_model_A
        
        # print("union: ", union_machines_effective_per_resource_common)
    

    def load_modelA_1(self, solver_name="cbc"):
        
        ## load the model and the solver
        model = Model('./model/modelA.mzn')
        solver = Solver.lookup(solver_name)
        instance = Instance(solver, model)
        
        ## load the data into the model
        instance["num_tests"] = self.num_tests_modelA
        instance["num_machines"] = self.num_machines
        instance["num_resources"] = self.num_resources_effective
        
        instance["durations"] = self.durations_modelA
        
        # instance["machines_allowed"] = self.machines_allowed_modelA
        instance["machines_allowed"] = self.machines_allowed_modelA
        instance["machines_pre_assigned"] = self.machines_pre_assigned
        
        
        instance["start_pre_assigned"] = self.start_pre_assigned
        
        # instance["machines_allowed"] = self.machines_effective_allowed_modelA
        
        instance["resources_allowed"] = self.resources_effective_modelA
        
        instance["offset_machine"] = self.offset_machine
        instance["offset_which_machine"] = self.offset_which_machine
        
        ## solve the model
        self.result = instance.solve()
        
    
    def load_modelA(self, solver_name="cbc"):
        
        ## load the model and the solver
        model = Model('./model/modelA.mzn')
        solver = Solver.lookup(solver_name)
        instance = Instance(solver, model)
        
        ## load the data into the model
        instance["num_tests"] = self.num_new_tasks_modelA
        instance["num_machines"] = self.num_machines
        instance["num_resources"] = self.num_resources_effective-1
        
        instance["durations"] = self.new_duarations_modelA 
        
        # instance["machines_allowed"] = self.machines_allowed_modelA
        # instance["machines_allowed"] = self.machines_allowed_modelA[:self.num_new_tasks_modelA]
        instance["machines_pre_assigned"] = self.machines_pre_assigned_A
        
        
        instance["gaps_start"] = self.gaps_start
        instance["num_gaps"] = len(self.gaps_start)
        instance["max_makespan"] = sum(self.durations)
        
        # instance["start_pre_assigned"] = self.start_pre_assigned
        
        # instance["machines_allowed"] = self.machines_effective_allowed_modelA
        
        # instance["resources_allowed"] = self.resources_effective_modelA_A
        
        # instance["offset_machine"] = self.offset_machine
        # instance["offset_which_machine"] = self.offset_which_machine
        
        ## solve the model
        self.result = instance.solve()
    
    
    ##! Not used with the greedy algorithm
    def load_modelB(self, solver_name="cbc"):
        
        self.makespan_A = self.result["makespan"]
        self.machines_assigned_A = self.result["machine_assigned"]
        self.start_times_A = self.result["start"]
        
        print("model A results")
        
        print("effective resources: ", self.resources_effective_modelA)
        print("makespan A: ", self.makespan_A)
        print("machines assigned A: ", self.machines_assigned_A)
        print("start times A: ", self.start_times_A)
        
        print()
        
        ## list tests with machine restrictions and not included in model A
        self.tests_modelB = [i+1 for i in range(self.num_tests) if i+1 not in self.tests_modelA]
        self.num_tests_modelB = len(self.tests_modelB)
        
        self.durations_modelB = [self.durations[i-1] for i in self.tests_modelB]
        self.machines_allowed_modelB = [self.machines_allowed[i-1] for i in self.tests_modelB]
        
        ## intial times of tests from model A 
        self.s_init_vec = self.start_times_A
        self.s_end_vec = [self.start_times_A[i] + self.durations_modelA[i] for i in range(self.num_tests_modelA)]
        self.n_s = self.num_tests_modelA
        
        
        ##! load model
        model = Model('./model/modelB.mzn')
        solver = Solver.lookup(solver_name)
        instance = Instance(solver, model)
        
        ## load the data into the model
        instance["num_tests_A"] = self.num_tests_modelA
        instance["num_tests"] = self.num_tests_modelB
        instance["num_machines"] = self.num_machines
        
        instance["durations"] = self.durations_modelB
        
        instance["machines_allowed"] = self.machines_allowed_modelB
        
        instance["n_s"] = self.n_s
        
        instance["s_init_vec"] = self.s_init_vec
        instance["s_end_vec"] = self.s_end_vec
        
        instance["machine_assigned_A"] = self.machines_assigned_A
        
        
        self.result_B = instance.solve()
        
        print("results model B: ", self.result_B)
        
    
    
    ##! for the second part of the problem, instead of using the minizinc model, I will use a greedy algorithm
    ##! assigning for each the task the machine that has that has the slot earlier available, it is allowed and obeys the restrictions
    def gready_algorithm_modelB(self):
    
        self.makespan_A = self.result["makespan"]
        self.machines_assigned_A = self.result["machine_assigned"]
        self.start_times_A = self.result["start"]
        
        print("model A results")
        
        print("effective resources: ", self.resources_effective_modelA)
        print("makespan A: ", self.makespan_A)
        print("machines assigned A: ", self.machines_assigned_A)
        print("start times A: ", self.start_times_A)
        
        print()
        
        ## list tests with machine restrictions and not included in model A
        self.tests_modelB = [i+1 for i in range(self.num_tests) if i+1 not in self.tests_modelA]
        self.num_tests_modelB = len(self.tests_modelB)
        
        self.durations_modelB = [self.durations[i-1] for i in self.tests_modelB]
        self.machines_allowed_modelB = [self.machines_allowed[i-1] for i in self.tests_modelB]
        
        ## intial times of tests from model A 
        self.s_init_vec = self.start_times_A
        self.s_end_vec = [self.start_times_A[i] + self.durations_modelA[i] for i in range(self.num_tests_modelA)]
        self.n_s = self.num_tests_modelA
        

        #! create the list of the machines assigned in model A and sort the data
        s_data_sorted = sorted([(self.s_init_vec[i], self.s_end_vec[i], self.machines_assigned_A[i]) for i in range(self.num_tests_modelA)]\
            , key=lambda x: x[2])

        # Group the sorted data based on the third component and maintain internal order
        s_data_grouped = []
        for key, group in groupby(s_data_sorted, key=lambda x: x[2]):
            # For each group, convert groupby object to a list to maintain sub-order
            group_list = list(group)
            group_list = sorted(group_list, key=lambda x: x[0])
            s_data_grouped.append(group_list)
        
        
        ##! the maximum width of the time schedule for a machine
        max_machine_time = sum(self.durations)
        
        ##! create the list of the available machines and their free time
        machine_availability = []
        for group in s_data_grouped:
            
            ###! find the free slots between the ordered assigned tasks
            for i in range(len(group)-1):
                if group[i][1] != group[i+1][0]:
                    machine_availability.append((group[i][1], group[i+1][0], group[i][2]))
                else:
                    pass
            
            if len(group) == 1 and group[0][0] != 0:
                machine_availability.append((0, group[0][0], group[0][2]))
                machine_availability.append((group[0][1], max_machine_time, group[0][2])) 
            elif len(group) == 1 and group[0][0] == 0:
                machine_availability.append((group[0][1], max_machine_time, group[0][2])) 
            else:
                machine_availability.append((group[-1][1], max_machine_time, group[-1][2]))
        
        
        ##! add the machines that have not been assigned any task: they are available from the beginning till the end
        for i in range(1, self.num_machines+1):
            if not any(machine[2] == i for machine in machine_availability):
                machine_availability.append((0, max_machine_time, i))
        
        
        ##! list of tasks, its id, duration and machines allowed that are up to be assigned
        tasks_modelB = [ {'id': number, 'duration': duration, 'machines': machines}\
            for number, duration, machines in zip(self.tests_modelB, self.durations_modelB, self.machines_allowed_modelB)]
        
        
        ##!? greedy algorithm
        ##! idea: start to assign machines with restrictions in a hierarchical way, first the ones that have more restrictions
        ##! sort first by the number of machines allowed, then by the longest duration, but separately
        
        tasks_modelB_restrictions_sorted = sorted([task for task in tasks_modelB if len(task['machines']) != self.num_machines], key=lambda x: (len(x['machines']), x['duration']), reverse=False)
        tasks_modelB_no_restrictions_sorted = sorted([task for task in tasks_modelB if len(task['machines']) == self.num_machines], key=lambda x: x['duration'], reverse=True)
        
        tasks_modelB_sorted = tasks_modelB_restrictions_sorted + tasks_modelB_no_restrictions_sorted
        
        # print([task['id'] for task in tasks_modelB_restrictions_sorted])
        # print()
        # print([task['id'] for task in tasks_modelB_no_restrictions_sorted])
        # print()
        # print([task['id'] for task in tasks_modelB_sorted])

        ##! create the heap with the available machines
        heapq.heapify(machine_availability)
        
        ##! dictionary with the task id and the machine assigned: {task_id: (machine_id, start_time)}
        tasks_assignment_B = {} 
        
        for task_id, task in enumerate(tasks_modelB_sorted):
            assigned = False
            task_duration = task['duration']
            machines_available = task['machines']
            
            slots_to_add = []
            while assigned == False:
                
                start_slot, end_slot, machine_id = heapq.heappop(machine_availability)

                if (machine_id in machines_available) and (end_slot - start_slot >= task_duration):
                    task_start_time = start_slot
                    tasks_assignment_B[task['id']] = (machine_id, task_start_time)

                    new_slots = []
                    if start_slot < task_start_time:
                        slots_to_add.append((start_slot, task_start_time, machine_id))
                    if task_start_time + task_duration < end_slot:
                        slots_to_add.append((task_start_time + task_duration, end_slot, machine_id))
                    
                    assigned = True
                else:
                    ##! it does not fit, save the slot to send back to the heap later
                    slots_to_add.append((start_slot, end_slot, machine_id))
            
            
            for slot in slots_to_add:
                ###! after the assignment, we add the slots back to the heap
                heapq.heappush(machine_availability, slot)


        tasks_assignment_A = {self.tests_modelA[i]: (self.machines_assigned_A[i], self.start_times_A[i]) for i in range(self.num_tests_modelA)}
        self.tasks_assignment = {**tasks_assignment_A, **tasks_assignment_B}
        self.total_makespan = max([self.tasks_assignment[task][1] + self.durations[task-1] for task in self.tasks_assignment])
        
        
        if self.total_makespan > self.makespan_A:
            print("best solution NOT garanteed")
            print("makespan A: ", self.makespan_A)
            print("makespan Total: ", self.total_makespan)
        else: 
            print("best solution garanteed")
            print("makespan A: ", self.makespan_A)
            print("makespan Total: ", self.total_makespan)
            
            ## but still we could minimize packing
        
        
        print("final tasks' assignment: ", self.tasks_assignment)
        print()
    
    
    def checker_solution(self):
        
        ##! check if the solution is correct
        for task in self.tasks_assignment:
            machine_id, start_time = self.tasks_assignment[task]
            duration = self.durations[task-1]
            
            if start_time < 0:
                print("Error: start time is negative")
                return False
            
            if machine_id not in self.machines_allowed[task-1]:
                print("Error: machine not allowed")
                
                print(task, machine_id)
                print(self.machines_allowed[task-1])
                return False
            
            if start_time + duration > self.total_makespan:
                print("Error: makespan is not correct")
                return False
        
            ##! overlpaping tasks in the same machine
            for task2 in self.tasks_assignment:
                if task != task2:
                    machine_id2, start_time2 = self.tasks_assignment[task2]
                    duration2 = self.durations[task2-1]
                    
                    if machine_id == machine_id2:
                        if (start_time >= start_time2 and start_time < start_time2 + duration2) or (start_time + duration > start_time2 and start_time + duration <= start_time2 + duration2):
                            print("Error: overlapping tasks")
                            print(task, task2)
                            print(start_time, start_time2)
                            print(duration, duration2)
                            print(machine_id, machine_id2)
                            return False
            
            ##! overlappint tasks that consume the same resource
            for resource in self.resources[task-1]:
                for task2 in self.tasks_assignment:
                    if task != task2:
                        machine_id2, start_time2 = self.tasks_assignment[task2]
                        duration2 = self.durations[task2-1]
                        
                        if resource in self.resources[task2-1] and resource != 'e':
                            if (start_time >= start_time2 and start_time < start_time2 + duration2) or (start_time + duration > start_time2 and start_time + duration <= start_time2 + duration2):
                                print("Error: overlapping tasks")
                                print(task, task2)
                                print(start_time, start_time2)
                                print(duration, duration2)
                                print(machine_id, machine_id2)
                                return False
        
        return True
    
    
    
    def create_output_file(self):
        
        with open(self.output_file_name, 'w') as output_file:
            output_file.write(f"% Makespan : {self.total_makespan}\n")
            
            for machine_id in range(1, self.num_machines+1):
                output_file.write(f"machine( 'm{machine_id}', ")
                
                tasks_machine = [(f"t{task}", self.tasks_assignment[task][1], self.resources[task-1]) for task in self.tasks_assignment if self.tasks_assignment[task][0] == machine_id]
                tasks_machine = sorted(tasks_machine, key=lambda x: x[1])
                tasks_machine = [(f"{task[0]}", task[1]) if task[2] == ['e'] else task for task in tasks_machine]
                num_tasks = len(tasks_machine)
                
                output_file.write(f"{num_tasks}, {tasks_machine})\n")
        
        print(f"Output file created: {self.output_file_name}")
        print()



    def crete_plot_file(self):
        
        tasks_data = {task: (self.tasks_assignment[task][0], self.tasks_assignment[task][1], self.durations[task-1], self.resources[task-1]) for task in self.tasks_assignment}

        machines = [f"m{tasks_data[task][0]}" for task in tasks_data]
        start_times = [tasks_data[task][1] for task in tasks_data]
        durations = [tasks_data[task][2] for task in tasks_data]
        tasks = [f"t{task}" for task in tasks_data]
        resources = [tasks_data[task][3] if tasks_data[task][3] != ['e']  else [''] for task in tasks_data]
        
        df = pd.DataFrame({
            'Machine': machines,
            'Task': tasks,
            'Start': start_times,
            'Duration': durations,
            'Resource': resources
        })

        fig, ax = plt.subplots(figsize=(20, 20))

        for i, task in enumerate(df.itertuples()):
            ax.barh(task.Machine, task.Duration, left=task.Start, align='center')
            ax.text(task.Start + task.Duration / 2, task.Machine, f"{task.Task} - {task.Resource}", va='center', ha='center', color='black', fontweight='bold')

        # Set labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Machines')
        ax.set_title('Job-Task Schedule')

        # Set x-axis to start at zero and extend based on the task durations
        ax.set_xlim(0, max(df['Start'] + df['Duration']))
        
        plt.tight_layout()
        plt.savefig(self.output_file_name + '.png')
        
        

if __name__ == "__main__":
    
    ##! using the Greedy algorithm
    
    time_start = time.time()
    
    problem = Problem.parse_instance()
    problem.read_input_data()
    
    
    problem.input_data_modelA()
    # problem.load_modelA()
    
    time_partial = time.time()
    
    # problem.gready_algorithm_modelB()
    
    # print("Is solution:", problem.checker_solution())
    
    # problem.create_output_file()
    # problem.crete_plot_file()
    
    time_end = time.time()
    
    print(f"Time A: {time_partial - time_start} secs")
    print(f"Time B: {time_end - time_partial} secs")
    print(f"Total time: {time_end - time_start} secs")