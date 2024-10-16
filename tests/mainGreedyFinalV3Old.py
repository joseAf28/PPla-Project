import sys 
import re                       # for regular expressions to parse the input data
from minizinc import Instance, Model, Solver
import time
import heapq                    # for the heap data structure used in the greedy algorithm
from itertools import groupby   # for grouping the data in the greedy algorithm

from collections import defaultdict

import matplotlib.pyplot as plt # for plotting the results



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
    
    
    ##! reduce the number of resources to the effective ones. for instance, if r1 and r2 are used in the same test, we can reduce it to [r1, r2]
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
    
    
    
    def input_data_modelA_500(self):
        
        ###! Input data for model A
        self.tests_modelA_1 = [i+1 for i in range(self.num_tests) if self.have_resources[i] == True]
        self.num_tests_modelA_1 = len(self.tests_modelA_1)
        
        ##? heuristic for t500 tests - works better than simply packing all the global tests in the same machine and 
        ##? then distribute the remaining tests among the machines with gaps
        
        tests_modelA_not_grouped = []
        tests_modelA_grouped = []
        
        for test in self.tests_modelA_1:
            if len(self.machines_allowed[test-1]) == self.num_machines:
                tests_modelA_grouped.append(test)
            else:
                tests_modelA_not_grouped.append(test)
                
                
        grouped_super_tests = defaultdict(set)
        
        for test in tests_modelA_grouped:
            resources = frozenset(self.resources[test-1])
            grouped_super_tests[resources].add(test)
        
        
        grouped_super_tests_duration = [sum(self.durations[i-1] for i in tests) for resources, tests in grouped_super_tests.items()]
        grouped_super_tests_machines = [setAllMachines for _ in range(len(grouped_super_tests))]
        grouped_super_tests_resources = [set(resources) for resources, tests in grouped_super_tests.items()]
        
        not_grouped_tests_duration = [self.durations[i-1] for i in tests_modelA_not_grouped]
        not_grouped_tests_machines = [self.machines_allowed[i-1] for i in tests_modelA_not_grouped]
        not_grouped_tests_resources = [set(self.resources[i-1]) for i in tests_modelA_not_grouped]
        
        tests_modelA_grouped = [len(tests_modelA_not_grouped)+i+1 for i in range(len(grouped_super_tests))]
        
        dict_non_grouped = {tests_modelA_not_grouped[i]:i+1 for i in range(len(tests_modelA_not_grouped))}
        dict_grouped = {}
        
        for i, resources in enumerate(grouped_super_tests):
            for test in grouped_super_tests[resources]:
                dict_grouped[test] = len(tests_modelA_not_grouped) + i + 1
        
        dict_total = {**dict_non_grouped, **dict_grouped}
        
        resources_allowed_modelA = [{dict_total[j] for j in self.resources_allowed[i]} for i in range(self.num_resources)]
        
        
        
        self.tests_modelA = not_grouped_tests_resources + grouped_super_tests_resources
        self.num_tests_modelA = len(self.tests_modelA)
        
        self.durations_modelA = not_grouped_tests_duration + grouped_super_tests_duration
        self.machines_allowed_modelA = not_grouped_tests_machines + grouped_super_tests_machines
        
        self.resources_effective_modelA = [set() for _ in range(self.num_resources)]
        for i in range(len(self.tests_resources_total)):
            for j in range(self.num_resources):
                if f'r{j+1}' in self.tests_resources_total[i]:
                    self.resources_allowed_total[j].add(i+1)
        
        self.num_resources_effective = len(self.resources_effective_modelA)
        
        #####? ends here the heuristic
    
    
    def input_data_modelA_init(self):
        ##! Input data for all the tests
        dictionaire_machines = {f"m{i+1}": i+1 for i in range(self.num_machines)}
        setAllMachines = {i+1 for i in range(self.num_machines)}
        
        self.machines_allowed = []
        for i in range(self.num_tests):
            if self.machines[i] == ['e']:
                self.machines_allowed.append(setAllMachines) ## default one
            else:
                self.machines_allowed.append({dictionaire_machines[machine] for machine in self.machines[i]})
        
        
        self.resources_allowed = [ [] for _ in range(self.num_resources)]
        self.have_resources = [False for _ in range(self.num_tests)]
        
        for i in range(self.num_tests):
            for j in range(self.num_resources):
                if f'r{j+1}' in self.resources[i]:
                    self.resources_allowed[j].append(i+1)
                    self.have_resources[i] = True
                    
        
        
    def input_data_modelA_middle_up_to_100(self):
        
        ###! Input data for model A
        ## It includes only the tests that have resources despite it has restrictions on the machines or not
        
        self.tests_modelA = [i+1 for i in range(self.num_tests) if self.have_resources[i] == True]
        self.num_tests_modelA = len(self.tests_modelA)
        
        ## convert the tests to a dictionary to have the index of the test (to start from 1)
        self.dictionaire_tests_modelA = {self.tests_modelA[i]:i+1  for i in range(self.num_tests_modelA)}
        self.dictionaire_undo_tests_modelA = {i+1: self.tests_modelA[i] for i in range(self.num_tests_modelA)}
        
        self.durations_modelA = [self.durations[i-1] for i in self.tests_modelA]
        self.machines_allowed_modelA = [self.machines_allowed[i-1] for i in self.tests_modelA]
        
        self.resources_allowed_modelA = [{self.dictionaire_tests_modelA[j] for j in self.resources_allowed[i]} for i in range(self.num_resources)]
        self.resources_effective_modelA = Problem.filter_sets(self.resources_allowed_modelA)
        
        self.resources_effective_modelA_real = [ [self.dictionaire_undo_tests_modelA[j] for j in i] for i in self.resources_effective_modelA]
        
        
        self.num_resources_effective = len(self.resources_effective_modelA)
        self.have_resources_modelA = [self.have_resources[i-1] for i in self.tests_modelA]
    
    
    
    def input_data_modelA_middle_500(self):
        
        setAllMachines = {i+1 for i in range(self.num_machines)}
        
        ###! Input data for model A
        self.tests_modelA_1 = [i+1 for i in range(self.num_tests) if self.have_resources[i] == True]
        self.num_tests_modelA_1 = len(self.tests_modelA_1)
        
        ##? heuristic for t500 tests - works better than simply packing all the global tests in the same machine and 
        ##? then distribute the remaining tests among the machines with gaps
        
        tests_modelA_not_grouped = []
        tests_modelA_grouped = []
        
        for test in self.tests_modelA_1:
            if len(self.machines_allowed[test-1]) == self.num_machines:
                tests_modelA_grouped.append(test)
            else:
                tests_modelA_not_grouped.append(test)
                
                
        self.grouped_super_tests = defaultdict(set)
        
        for test in tests_modelA_grouped:
            resources = frozenset(self.resources[test-1])
            self.grouped_super_tests[resources].add(test)
        
        
        self.grouped_super_tests_duration = [sum(self.durations[i-1] for i in tests) for resources, tests in self.grouped_super_tests.items()]
        self.grouped_super_tests_machines = [setAllMachines for _ in range(len(self.grouped_super_tests))]
        self.grouped_super_tests_resources = [set(resources) for resources, tests in self.grouped_super_tests.items()]
        
        not_grouped_tests_duration = [self.durations[i-1] for i in tests_modelA_not_grouped]
        not_grouped_tests_machines = [self.machines_allowed[i-1] for i in tests_modelA_not_grouped]
        not_grouped_tests_resources = [set(self.resources[i-1]) for i in tests_modelA_not_grouped]
        
        tests_modelA_grouped = [len(tests_modelA_not_grouped)+i+1 for i in range(len(self.grouped_super_tests))]
        
        self.dict_non_grouped = {tests_modelA_not_grouped[i]:i+1 for i in range(len(tests_modelA_not_grouped))}
        self.dict_undo_non_grouped = {i+1: tests_modelA_not_grouped[i] for i in range(len(tests_modelA_not_grouped))}
        
        self.len_tests_modelA_not_grouped = len(tests_modelA_not_grouped)
        
        self.dict_grouped = {}
        for i, resources in enumerate(self.grouped_super_tests):
            for test in self.grouped_super_tests[resources]:
                self.dict_grouped[test] = self.len_tests_modelA_not_grouped + i + 1
                
        
        self.dictionaire_tests_modelA = {**self.dict_non_grouped, **self.dict_grouped}
        
        resources_allowed_modelA = [{self.dictionaire_tests_modelA[j] for j in self.resources_allowed[i]} for i in range(self.num_resources)]
        
        self.tests_modelA = not_grouped_tests_resources + self.grouped_super_tests_resources
        self.num_tests_modelA = len(self.tests_modelA)
        
        self.durations_modelA = not_grouped_tests_duration + self.grouped_super_tests_duration
        self.machines_allowed_modelA = not_grouped_tests_machines + self.grouped_super_tests_machines
        
        self.resources_effective_modelA = [set() for _ in range(self.num_resources)]
        for i in range(self.num_tests_modelA):
            for j in range(self.num_resources):
                if f'r{j+1}' in self.tests_modelA[i]:
                    self.resources_effective_modelA[j].add(i+1)
        
        self.num_resources_effective = len(self.resources_effective_modelA)
    
    
    
    def input_data_modelA_end(self, nb_max_non_ordering):
        
        ##! find the the tests with global resources that can be superposed in time
        ### 1 - start with the tests that have the least number of resources
        ### 2 - compare them with all the other tests to check if they can be superposed
        ### 3 - if they can be superposed, add them to the list of tests that can be superposed
        ### 4 - limit the number of tests that can be superposed: it will be considered an hyperparameter of the model that can be changed
        ### 5 - to break further simmetries, we only consider the first element of the pair to superpose from all pairs available to cut further the search space
        
        resources_allowed_per_testA = [ set() for _ in range(self.num_tests_modelA)]
        
        for i in range(self.num_tests):
            for j in range(self.num_resources):
                if f'r{j+1}' in self.resources[i]:
                    resources_allowed_per_testA[self.dictionaire_tests_modelA[i+1]-1].add(j+1)
        
        
        len_resources_allowed_per_testA = [len(resources) for resources in resources_allowed_per_testA]
    
        
        min_resources = min(len_resources_allowed_per_testA)
        max_resources = max(len_resources_allowed_per_testA)
        cap = 1.0
        
        min_check = min_resources
        max_check = round(min_resources + cap*(max_resources - min_resources))
        
        queue_tests_to_superpose = [i+1 for i in range(self.num_tests_modelA) if (len_resources_allowed_per_testA[i] >= min_check and len_resources_allowed_per_testA[i] <= max_check)]
        
        
        if len(queue_tests_to_superpose) > nb_max_non_ordering:
            queue_tests_to_superpose = sorted(queue_tests_to_superpose, key=lambda x: len_resources_allowed_per_testA[x-1], reverse=False)
            queue_tests_to_superpose = queue_tests_to_superpose[:nb_max_non_ordering]
        
        
        tests_allowed_to_superpose = []
        for element in queue_tests_to_superpose:
            superposition_set = set()
            for j in range(len(resources_allowed_per_testA)):
                
                if (element-1 != j) and  (resources_allowed_per_testA[element-1].isdisjoint(resources_allowed_per_testA[j])):
                    superposition_set.add(j+1)
                    
            if len(superposition_set) > 1:
                tests_allowed_to_superpose.append(superposition_set)
            else: 
                tests_allowed_to_superpose.append(set())
                
        
        self.pairs_allowed_to_superpose = []
        if len(tests_allowed_to_superpose) > 0:
            for i in range(len(queue_tests_to_superpose)):
                if len(tests_allowed_to_superpose[i]) > 0:
                    for test in tests_allowed_to_superpose[i]:
                        self.pairs_allowed_to_superpose.append([queue_tests_to_superpose[i], test])
        else:
            pass             
        
        
        ##! choose only the first element of the pairs to superpose
        self.unique_pairs_allowed_to_superpose = []
        seen_first_element = set()
        
        for pair in self.pairs_allowed_to_superpose:
            if pair[0] not in seen_first_element:
                self.unique_pairs_allowed_to_superpose.append(pair)
                seen_first_element.add(pair[0])
            else:
                pass
        
        
        ###! tests that can't be superposed in time are ordered in a way that they don't overlap in time to break simmetries
        ### By default, we consider a window of size 5 to order the tests that use the same resource 
        window_size = 5
        
        pairs_ordering_same_resource = []
        for i in range(len(self.resources_effective_modelA)):
            for ele1 in self.resources_effective_modelA[i]:
                for ele2 in self.resources_effective_modelA[i]:
                    if ele1 < ele2 and ele2 <= ele1 + window_size:
                        pairs_ordering_same_resource.append([ele1, ele2])
        
        
        ##! Finnally we exlude the pairs that are allowed to superpose so that we send two different order restrictions to the miniZinc model
        self.pairs_ordering_same_resource_with_superpose = []
        queue_aux = list(set([pair[0] for pair in self.unique_pairs_allowed_to_superpose]))
        
        # print("queue aux: ", queue_aux)
        
        for pair in pairs_ordering_same_resource:
            if pair[0] in queue_aux or pair[1] in queue_aux:
                pass
            else:
                self.pairs_ordering_same_resource_with_superpose.append(pair)
        
        
        ##! Bu pre-assigning the machines to the tests with the heursitic of miniming the use of the overlapping tasks in the same machine condition in
        ##! in Minizinc model, we break further simmetries and cut further the search space
        
        self.machines_pre_assigned = [0 for _ in range(self.num_tests_modelA)]
        
        ## assign the first purpose machine for the tests with machine restrictions
        for i in range(self.num_tests_modelA):
            if len(self.machines_allowed_modelA[i]) < self.num_machines:
                self.machines_pre_assigned[i] = list(self.machines_allowed_modelA[i])[0]
        
        ## remaining global tests are distributed evenly among the machines
        machine_value = 1
        for i in range(self.num_tests_modelA):
            if len(self.machines_allowed_modelA[i]) == self.num_machines:
                self.machines_pre_assigned[i] = machine_value
                
                machine_value += 1
                if machine_value > self.num_machines:
                    machine_value = 1
    
    
    
    def load_modelA(self, new_baseline, solver_name="cbc"):
        
        ## load the model and the solver
        model = Model('./model/modelAFinal.mzn')
        solver = Solver.lookup(solver_name)
        instance = Instance(solver, model)
        
        ## load the data into the model
        instance["num_tests"] = self.num_tests_modelA
        instance["num_machines"] = self.num_machines
        instance["num_resources"] = self.num_resources_effective

        
        instance["durations"] = self.durations_modelA
        instance["machines_pre_assigned"] = self.machines_pre_assigned
        instance["resources_allowed"] = self.resources_effective_modelA
        
        ##! baseline makespan
        baseline = sum(self.durations_modelA)
        if new_baseline < baseline and new_baseline > 0:
            
            baseline = new_baseline
        
        instance["num_makespan"] = baseline
        
        instance["nb_pairs_allowed_to_superpose"] = len(self.unique_pairs_allowed_to_superpose)
        instance["pairs_allowed_to_superpose"] = self.unique_pairs_allowed_to_superpose
        
        
        instance["nb_pairs_ordering_same_resource_with_superpose"] = len(self.pairs_ordering_same_resource_with_superpose)
        instance["pairs_ordering_same_resource_with_superpose"] = self.pairs_ordering_same_resource_with_superpose
        
        ## solve the model
        result = instance.solve()
        
        self.makespan_A = result["makespan"]
        self.machines_assigned_A = result["machine_assigned"]
        self.start_times_A = result["start"]
    
    
    
    def baseline_solution_modelA(self):
        
        ##! all tests that have global resoources are distributed consecutively in time without overlapping regardless of the resource
        
        self.machines_assigned_A = self.machines_pre_assigned
        
        start_times = [0 for _ in range(self.num_tests_modelA)]
        for i in range(1, self.num_tests_modelA):
            start_times[i] = start_times[i-1] + self.durations_modelA[i-1]
        
        self.start_times_A = start_times
        self.makespan_A = max([self.start_times_A[i] + self.durations_modelA[i] for i in range(self.num_tests_modelA)])
    
    
    
    ##! for the second part of the problem, instead of using the minizinc model, we will use a greedy algorithm
    ##! assigning for each the task the machine that has that has the slot earlier available, it is allowed and obeys the restrictions
    def gready_algorithm_modelB(self):
        
        ## list tests with machine restrictions and not included in model A
        if self.num_tests < 500:
            self.tests_modelB = [i+1 for i in range(self.num_tests) if i+1 not in self.tests_modelA]
        else:
            self.tests_modelB = [i+1 for i in range(self.num_tests) if i+1 not in self.tests_modelA_1]
        
        
        self.num_tests_modelB = len(self.tests_modelB)
        
        self.durations_modelB = [self.durations[i-1] for i in self.tests_modelB]
        self.machines_allowed_modelB = [self.machines_allowed[i-1] for i in self.tests_modelB]
        
        ## intial times of tests from model A 
        self.s_init_vec = self.start_times_A
        self.s_end_vec = [self.start_times_A[i] + self.durations_modelA[i] for i in range(self.num_tests_modelA)]
        self.n_s = self.num_tests_modelA
        

        ## create the list of the machines assigned in model A and sort the data
        s_data_sorted = sorted([(self.s_init_vec[i], self.s_end_vec[i], self.machines_assigned_A[i]) for i in range(self.num_tests_modelA)]\
            , key=lambda x: x[2])

        ## Group the sorted data based on the third component and maintain internal order
        s_data_grouped = []
        for key, group in groupby(s_data_sorted, key=lambda x: x[2]):
            # For each group, convert groupby object to a list to maintain sub-order
            group_list = list(group)
            group_list = sorted(group_list, key=lambda x: x[0])
            s_data_grouped.append(group_list)
        
        
        ## the maximum width of the time schedule for a machine
        max_machine_time = sum(self.durations)
        
        ## create the list of the available machines and their free time
        machine_availability = []
        for group in s_data_grouped:
            
            ### find the free slots between the ordered assigned tasks
            for i in range(len(group)-1):
                if group[i][1] != group[i+1][0]:
                    machine_availability.append((group[i][1], group[i+1][0], group[i][2]))
                else:
                    pass
            
            ## add the first slot if it is not 0
            if len(group) - 1 > 0 and group[0][0] != 0: 
                machine_availability.append((0, group[0][0], group[0][2]))
            
            ## add the slots if grup has only one element
            if len(group) == 1 and group[0][0] != 0:
                machine_availability.append((0, group[0][0], group[0][2]))
                machine_availability.append((group[0][1], max_machine_time, group[0][2])) 
            elif len(group) == 1 and group[0][0] == 0:
                machine_availability.append((group[0][1], max_machine_time, group[0][2])) 
            else:
                machine_availability.append((group[-1][1], max_machine_time, group[-1][2]))
        
        
        ## add the machines that have not been assigned any task: they are available from the beginning till the end
        for i in range(1, self.num_machines+1):
            if not any(machine[2] == i for machine in machine_availability):
                machine_availability.append((0, max_machine_time, i))
        
        
        print("machine availability BEGIN: ", machine_availability)
        
        
        print("--"*40)
        print()
        
        
        ## list of tasks, its id, duration and machines allowed that are up to be assigned
        tasks_modelB = [ {'id': number, 'duration': duration, 'machines': machines}\
            for number, duration, machines in zip(self.tests_modelB, self.durations_modelB, self.machines_allowed_modelB)]
        
        
        ##! greedy algorithm - part B
        ## this remaining task has no complexity in terms of restrictions as we already distribute the tests with restrictions in model A
        ## idea: start to assign machines with restrictions in a hierarchical way, first the ones that have more restrictions
        ## goal: minimize the makespan by maintaining the machines as balanced as possible
        ## Moreover, we sort the tasks by the duration in descending order and the gaps let by distributed tasks in model A are huge and almost always enough to fit the remaining tasks without 
        ## further increase the makespan of part A
        ## sort first by the number of machines allowed, then by the longest duration, but separately
        
        tasks_modelB_restrictions_sorted = sorted([task for task in tasks_modelB if len(task['machines']) != self.num_machines], key=lambda x: (len(x['machines']), x['duration']), reverse=False)
        tasks_modelB_no_restrictions_sorted = sorted([task for task in tasks_modelB if len(task['machines']) == self.num_machines], key=lambda x: x['duration'], reverse=True)
        
        tasks_modelB_sorted = tasks_modelB_restrictions_sorted + tasks_modelB_no_restrictions_sorted
        

        ## create the heap with the available machines
        heapq.heapify(machine_availability)
        
        ## dictionary with the task id and the machine assigned: {task_id: (machine_id, start_time)}
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
                    ## it does not fit, save the slot to send back to the heap later
                    slots_to_add.append((start_slot, end_slot, machine_id))
            
            
            for slot in slots_to_add:
                ### after the assignment, we add the slots back to the heap
                heapq.heappush(machine_availability, slot)


        print("machine availability: ", machine_availability)
        print("tasks assignment B: ", tasks_assignment_B)
        
        print("*"*50)


        if self.num_tests < 500:
            
            tasks_assignment_A = {self.tests_modelA[i]: (self.machines_assigned_A[i], self.start_times_A[i]) for i in range(self.num_tests_modelA)}
            self.tasks_assignment = {**tasks_assignment_A, **tasks_assignment_B}
            self.total_makespan = max([self.tasks_assignment[task][1] + self.durations[task-1] for task in self.tasks_assignment])
        else:
            
            tasks_assignment_A = defaultdict(list)
            
            ##! ungrouped A
            for i in range(self.len_tests_modelA_not_grouped):
                tasks_assignment_A[self.dict_undo_non_grouped[i+1]] = [self.machines_assigned_A[i], self.start_times_A[i]]
            
            
            ##! grouped A
            ## we have the same ordered as previously defined for the suprtests groups 
            counter = self.len_tests_modelA_not_grouped
            for key, value in self.grouped_super_tests.items():
                
                tests = list(value)
                start_time = self.start_times_A[counter]
                
                start_time_vec = [start_time for _ in range(len(tests))]
                for i in range(1, len(tests)):
                    start_time_vec[i] = start_time_vec[i-1] + self.durations[tests[i-1]-1]
                    
                for i, test in enumerate(tests):
                    tasks_assignment_A[test] = [self.machines_assigned_A[counter], start_time_vec[i]]
                
                counter += 1
            
            
            ###! grouped B
            self.tasks_assignment = {**tasks_assignment_A, **tasks_assignment_B}
            
            self.total_makespan = max([self.tasks_assignment[task][1] + self.durations[task-1] for task in self.tasks_assignment])
        
        
        
        ##! check if the solution is has the minimum makespan available
        if self.total_makespan > self.makespan_A:
            print("best solution NOT garanteed")
            print("makespan A: ", self.makespan_A)
            print("makespan Total: ", self.total_makespan)
            print("baseline makespan: ", sum(self.durations_modelA))
            
            ##! improve the solution
            
            ##! internal gaps
            print("machine availability: ", machine_availability)
            
            frontier_gaps = [gap for gap in machine_availability if gap[1] == max_machine_time]
            
            tupple_makespan = [element for element in frontier_gaps if element[0] == self.total_makespan][0]
            
            print("tupple makespan: ", tupple_makespan)
            
            tasks_frontier = []
            machine_frontier = []
            time_limit_frontier = []
            
            for task in tasks_assignment_B:
                for gap in frontier_gaps:
                    if tasks_assignment_B[task][0] == gap[2]:
                        if tasks_assignment_B[task][1] + self.durations[task-1] == gap[0]:
                            tasks_frontier.append(task)
                            machine_frontier.append(gap[2])
                            time_limit_frontier.append(gap[0])
                            
            print("tasks frontier: ", tasks_frontier)
            print("machine frontier: ", machine_frontier)
            
            
            index_task_makespan = machine_frontier.index(tupple_makespan[2])
            duration_task_makespan = self.durations[tasks_frontier[index_task_makespan]-1]
            task_makespan = tasks_frontier[index_task_makespan]
            machine_makespan = machine_frontier[index_task_makespan]
            
            print("task makespan: ", tasks_frontier[index_task_makespan])
            
            machine_frontier_without_makespan = machine_frontier[:index_task_makespan] + machine_frontier[index_task_makespan+1:]
            tasks_frontier_without_makespan = tasks_frontier[:index_task_makespan] + tasks_frontier[index_task_makespan+1:]
            time_limit_frontier_without_makespan = time_limit_frontier[:index_task_makespan] + time_limit_frontier[index_task_makespan+1:]
            
            ##! check if an exchange of tasks can improve the makespan and store the information a new_makespan list to later choose the better
            new_makespan_list = []
            
            for i in range(len(tasks_frontier_without_makespan)):
                # print("machine allowed: ", self.machines_allowed[tasks_frontier_without_makespan[i]-1])
                # print("machine frontier: ", machine_frontier)
                
                # print("checker:" , machine_makespan in list(self.machines_allowed[tasks_frontier_without_makespan[i]-1]))
                
                if machine_makespan in list(self.machines_allowed[tasks_frontier_without_makespan[i]-1]):
                    new_makespan0 = self.total_makespan - duration_task_makespan + self.durations[tasks_frontier_without_makespan[i]-1]
                    new_makespan1 = time_limit_frontier_without_makespan[i] - self.durations[tasks_frontier_without_makespan[i]-1] + duration_task_makespan
                        
                    new_makespan = max(new_makespan0, new_makespan1)
                    new_makespan_list.append((tasks_frontier_without_makespan[i], time_limit_frontier_without_makespan[i], machine_frontier_without_makespan[i], i))
                        
            
            chosen_change_minimal_makespan = min(new_makespan_list, key=lambda x: x[0])
            
            # tasks_assignment_B[task_makespan] = (chosen_change_minimal_makespan[2], time_limit_frontier_without_makespan[chosen_change_minimal_makespan[3]] - self.durations[chosen_change_minimal_makespan[0]-1])
            # tasks_assignment_B[chosen_change_minimal_makespan[0]] = (tupple_makespan[2], self.total_makespan - duration_task_makespan)
            
            
            # self.tasks_assignment = {**tasks_assignment_A, **tasks_assignment_B}
            # self.total_makespan = max([self.tasks_assignment[task][1] + self.durations[task-1] for task in self.tasks_assignment])
            
            print(" total makespan after improvement: ", self.total_makespan)
            # print("totla makespan before improvement: ", chosen_change_minimal_makespan[0])
            
            print("tasks assignment B chosen: ", tasks_assignment_B[chosen_change_minimal_makespan[0]])
            print("tasks assignment B makespan: ", tasks_assignment_B[task_makespan])
            
            print("chosen change minimal makespan: ", chosen_change_minimal_makespan)
            # print("tasks frontier without makespan: ", task_makespan)
            
            # print()
            # print("frontier gaps: ", frontier_gaps)
            # print("tasks frontier: ", new_makespan_list)
            
            
            
            
        else: 
            print("best solution garanteed")
            print("makespan A: ", self.makespan_A)
            print("makespan Total: ", self.total_makespan)
            print("baseline makespan: ", sum(self.durations_modelA))
        
        
        # print("final tasks' assignment: ", self.tasks_assignment)
        # print()
        
        return self.total_makespan
    
    
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



    def create_plot_file(self):
    
        tasks_data = {task: (self.tasks_assignment[task][0], self.tasks_assignment[task][1], self.durations[task-1], self.resources[task-1]) for task in self.tasks_assignment}

        machines = [f"m{tasks_data[task][0]}" for task in tasks_data]
        start_times = [tasks_data[task][1] for task in tasks_data]
        durations = [tasks_data[task][2] for task in tasks_data]
        tasks = [f"t{task}" for task in tasks_data]
        resources = [tasks_data[task][3] if tasks_data[task][3] != ['e']  else [''] for task in tasks_data]
        
        
        fig, ax = plt.subplots(figsize=(25, 25))
        
        for i in range(len(machines)):
            ax.barh(machines[i], durations[i], left=start_times[i], align='center')
            ax.text(start_times[i] + durations[i]/ 2, machines[i], f"{tasks[i]} - {resources[i]}", va='center', ha='center', color='black', fontweight='bold', fontsize=5)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Machines')
        ax.set_title('Machine-Test Schedule')

        limit_max = max(start_times) + max(durations)
        ax.set_xlim(0, limit_max)
        
        plt.tight_layout()
        plt.savefig(self.output_file_name.split('.')[0] + '.png')
        
        print(f"Plot file created: {self.output_file_name.split('.')[0] + '.png'}")
        print()



if __name__ == "__main__":
    
    time_start = time.time()

    ##e hyperparameter variable
    nb_max_non_ordering = 0
    
    max_time = 60 * 5 # 5 minutes
    total_make_span = 0
    old_make_span = 0
    
    counter_same_makespan = 0
    counter_nb_zero = 0
    
    counter_t500_larger = 0
    thereshold_t500_larger = 65 ##! 65 tests that cannot overlap in time
    
    time_partial = time.time()
    
    counter_extra = 0
    
    ##! iterate over the hyperparameter till the solution is not further improved
    while time_partial > max_time:
        
        problem = Problem.parse_instance()
        problem.read_input_data()
        
        problem.input_data_modelA_init()
        
        ## set the initial value for the hyperparameter based on the number of tests
        if problem.num_tests < 500 and counter_nb_zero == 0:
            nb_max_non_ordering = 6
            counter_nb_zero = 1
        elif problem.num_tests == 500 and counter_nb_zero == 0:
            nb_max_non_ordering = 2
            counter_nb_zero = 1
        
        
        print("-"*30)
        print("Model with nb_max_non_ordering: ", nb_max_non_ordering)
        
        if problem.num_tests < 500:
            problem.input_data_modelA_middle_up_to_100()
        else:
            problem.input_data_modelA_middle_500()
        
        problem.input_data_modelA_end(nb_max_non_ordering=nb_max_non_ordering)
        
        
        if problem.num_tests == 500 and problem.num_tests_modelA > thereshold_t500_larger and counter_t500_larger == 0:
            ##! ensure that the larger t500 tests have at least one solution
            
            problem.baseline_solution_modelA()
            nb_max_non_ordering = 0
            counter_t500_larger = 1
        else:
            problem.load_modelA(total_make_span)
        
        
        now_makespan = problem.gready_algorithm_modelB()
        
        # print("Is solution:", problem.checker_solution())
        
        problem.create_output_file()
        problem.create_plot_file()
        
        time_partial = time.time()
        
        print("Nb elements in model A: ", problem.num_tests_modelA)
        print("Time elapsed: ", round((time_partial - time_start)/60,3), " mins")
        print()
        
        
        old_make_span = total_make_span
        total_make_span = now_makespan

        
        if old_make_span == total_make_span:
            counter_same_makespan += 1
            

        ##! convergence: if the same value for makespan is obtained twice, we stop the search
        ##! or the maximum number of tests in model A has been reached and we the solution cannot be further improved
        if nb_max_non_ordering > problem.num_tests_modelA or counter_same_makespan == 2:
            break
        
        if nb_max_non_ordering < 10:
            nb_max_non_ordering += 2
        else:
            nb_max_non_ordering += 1
        
            
        # if counter_extra == 0:
        #     break
        
        
    print("Total time elapsed: ", round((time.time() - time_start)/60,3), " mins")