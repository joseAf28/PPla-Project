import sys
import re
from minizinc import Instance, Model, Solver
import time
import itertools
import heapq

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
    
    
    def convert_input_models(self):
        
        dictionaire_machines = {f"m{i+1}": i+1 for i in range(self.num_machines)}
        dictionaire_resources = {f"r{i+1}": i for i in range(self.num_resources)}
        
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
        
        
        
        # ##! not including the tests that have resources
        # self.machine_restrictions = [True if (self.machines_allowed[i] != setAllMachines and self.have_resources[i] == False) else False for i in range(self.num_tests)]
        
        ## Input data for model A
        # self.tests_modelA = [i+1 for i in range(self.num_tests) if (self.have_resources[i] == True or self.machines_allowed[i] != setAllMachines)] # true tests name
        self.tests_modelA = [i+1 for i in range(self.num_tests) if self.have_resources[i] == True]
        self.num_tests_modelA = len(self.tests_modelA)
        
        dictionaire_tests_modelA = {self.tests_modelA[i]:i+1  for i in range(self.num_tests_modelA)}
        
        self.durations_modelA = [self.durations[i-1] for i in self.tests_modelA]
        self.machines_allowed_modelA = [self.machines_allowed[i-1] for i in self.tests_modelA]
        self.resources_allowed_modelA = [{dictionaire_tests_modelA[j] for j in self.resources_allowed[i]} for i in range(self.num_resources)]
        self.resources_effective_modelA = Problem.filter_sets(self.resources_allowed_modelA)
        
        self.num_resources_effective = len(self.resources_effective_modelA)
        
        self.have_resources_modelA = [self.have_resources[i-1] for i in self.tests_modelA]
        
        
        self.tests_unique_machines_no_resources = [i+1 for i in range(self.num_tests) if (self.have_resources[i] == False and len(self.machines_allowed[i]) == 1)]
        
        self.offset_machine = [set() for _ in range(self.num_machines)]
        for test in self.tests_unique_machines_no_resources:
            self.offset_machine[list(self.machines_allowed[test-1])[0]-1].add(self.durations[test-1])
        
        self.offset_machine = [sum(offset) for offset in self.offset_machine]
        self.offset_which_machine = [i+1 if self.offset_machine[i] > 0 else 0 for i in range(self.num_machines)]
        
        
        ## Ideas that seem not to improve the efficiency
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
    

    def load_modelA(self, solver_name="cbc"):
        
        ## load the model and the solver
        model = Model('./model/modelA.mzn')
        solver = Solver.lookup(solver_name)
        instance = Instance(solver, model)
        
        ## load the data into the model
        instance["num_tests"] = self.num_tests_modelA
        instance["num_machines"] = self.num_machines
        instance["num_resources"] = self.num_resources_effective
        
        instance["durations"] = self.durations_modelA
        
        instance["machines_allowed"] = self.machines_allowed_modelA
        instance["resources_allowed"] = self.resources_effective_modelA
        instance["have_resources"] = self.have_resources_modelA
        
        instance["offset_machine"] = self.offset_machine
        instance["offset_which_machine"] = self.offset_which_machine
        
        # instance["test_resources_no_restriction_machines"] = self.test_resources_modelA_no_restriction_machines
        
        print("num_resources: ", self.num_resources_effective)
        print("resources: ", self.resources_effective_modelA)
        print("machines: ", self.machines_allowed_modelA)
        
        ## solve the model
        self.result = instance.solve()
        print(self.result)
    
    
    def input_data_model_B(self):
    
        self.makespan_A = self.result["makespan"]
        self.machines_assigned_A = self.result["machine_assigned"]
        self.start_times_A = self.result["start"]

        # print(self.machines_assigned_A)
        # print(self.start_times_A)
        # print(self.makespan_A)
        
        
        
        ## list tests with machine restrictions and not included in model A
        self.tests_modelB = [i+1 for i in range(self.num_tests) if i+1 not in self.tests_modelA]
        # self.tests_modelB_restrictions = [i+1 for i in range(self.num_tests) if i+1 not in self.tests_modelA and len(self.machines_allowed[i]) != self.num_machines]
        
        self.num_tests_modelB = len(self.tests_modelB)
        
        self.durations_modelB = [self.durations[i-1] for i in self.tests_modelB]
        self.machines_allowed_modelB = [self.machines_allowed[i-1] for i in self.tests_modelB]
        
        ## intial times of tests from model A 
        self.s_init_vec = self.start_times_A
        self.s_end_vec = [self.start_times_A[i] + self.durations_modelA[i] for i in range(self.num_tests_modelA)]
        self.n_s = self.num_tests_modelA
        
        # ## machines assignes model A
        # self.machines_assigned_A = [self.machines_assigned_A[i-1] for i in self.tests_modelA]
        
    
    ##! not used by now
    def load_modelB(self, solver_name="cbc"):
        
        ## load the model and the solver
        model = Model('./model/modelB.mzn')
        solver = Solver.lookup(solver_name)
        instance = Instance(solver, model)
        
        ## load the data into the model
        instance["num_tests_A"] = self.num_tests_modelA
        instance["num_tests"] = self.num_tests_modelB
        instance["num_machines"] = self.num_machines
        
        instance["durations"] = self.durations_modelB
        instance["machines_allowed"] = self.machines_allowed_modelB
        
        instance["s_init_vec"] = self.s_init_vec
        instance["s_end_vec"] = self.s_end_vec
        instance["n_s"] = self.n_s
        
        instance["machine_assigned_A"] = self.machines_assigned_A
        
        ## solve the model
        self.result = instance.solve()
        print(self.result)
    
    
    ##! for the second part of the problem, instead of using the minizinc model, I will use a greedy algorithm
    ##! assigning for each the task the machine that has that has the slot earlier available, it is allowed and obeys the restrictions
    def gready_algorithm_modelB(self):
    
        self.makespan_A = self.result["makespan"]
        self.machines_assigned_A = self.result["machine_assigned"]
        self.start_times_A = self.result["start"]

        
        ## list tests with machine restrictions and not included in model A
        self.tests_modelB = [i+1 for i in range(self.num_tests) if i+1 not in self.tests_modelA]
        # self.tests_modelB_restrictions = [i+1 for i in range(self.num_tests) if i+1 not in self.tests_modelA and len(self.machines_allowed[i]) != self.num_machines]
        
        self.num_tests_modelB = len(self.tests_modelB)
        
        self.durations_modelB = [self.durations[i-1] for i in self.tests_modelB]
        self.machines_allowed_modelB = [self.machines_allowed[i-1] for i in self.tests_modelB]
        
        ## intial times of tests from model A 
        self.s_init_vec = self.start_times_A
        self.s_end_vec = [self.start_times_A[i] + self.durations_modelA[i] for i in range(self.num_tests_modelA)]
        self.n_s = self.num_tests_modelA
        
        
        tests_sorted = [i for _, i in sorted(zip(self.durations_modelB, self.tests_modelB), reverse=True)]
        
        #! create the list of the available machines and their free time
        machine_loads_aux = [(self.s_init_vec[i], self.s_end_vec[i], self.machines_assigned_A[i]) for i in range(self.num_tests_modelA)]

        from itertools import groupby
        from operator import itemgetter

        sorted_data = sorted(machine_loads_aux, key=itemgetter(2))

        # Group the sorted data based on the third component and maintain internal order
        result = []
        for key, group in groupby(sorted_data, key=itemgetter(2)):
            # For each group, convert groupby object to a list to maintain sub-order
            group_list = list(group)
            result.append(group_list)
        
        new_sorted_data = []
        for group in result:
            group = sorted(group, key=lambda x: x[0])
            new_sorted_data.append(group)
        
        
        self.max_number = 100000  ##! maximum size number for now
        
        
        ##! create the list of the available machines and their free time
        machine_availability = []
        for group in new_sorted_data:
            
            ###! find the free slots between the ordered assigned tasks
            for i in range(len(group)-1):
                if group[i][1] != group[i+1][0]:
                    machine_availability.append((group[i][1], group[i+1][0], group[i][2]))
                else:
                    pass
            
            if len(group) == 1 and group[0][0] != 0:
                machine_availability.append((0, group[0][0], group[0][2]))
                machine_availability.append((group[0][1], self.max_number, group[0][2])) ## 9999 not important
            elif len(group) == 1 and group[0][0] == 0:
                machine_availability.append((group[0][1], self.max_number, group[0][2])) ## 9999 not important
            else:
                machine_availability.append((group[-1][1], self.max_number, group[-1][2]))
        
        
        ##! add the machines that have not been assigned any task: they are available from the beginning till the end
        for i in range(1, self.num_machines+1):
            if not any(machine[2] == i for machine in machine_availability):
                machine_availability.append((0, self.max_number, i))
        
        
        ##! list of tasks, its id, duration and machines allowed that are up to be assigned
        tasks_modelB = [ {'id': number, 'duration': duration, 'machines': machines} for number, duration, machines in zip(self.tests_modelB, self.durations_modelB, self.machines_allowed_modelB)]
        
        print("machine_availability")
        print(machine_availability)
        print()
        
        ##!? greedy algorithm
        ##! idea: start for assigment machines with restrictions in a hierarchical way, first the ones that have more restrictions
        
        ##! sorting tasks by the longest duration
        ##! sort first by the number of machines allowed, then by the duration, but separately
        
        tasks_modelB_restrictions_sorted = sorted([task for task in tasks_modelB if len(task['machines']) != self.num_machines], key=lambda x: (len(x['machines']), x['duration']), reverse=False)
        tasks_modelB_no_restrictions_sorted = sorted([task for task in tasks_modelB if len(task['machines']) == self.num_machines], key=lambda x: x['duration'], reverse=True)
        
        tasks_modelB_sorted = tasks_modelB_restrictions_sorted + tasks_modelB_no_restrictions_sorted
        
        print([task['id'] for task in tasks_modelB_restrictions_sorted])
        print()
        print([task['id'] for task in tasks_modelB_no_restrictions_sorted])
        print()
        print([task['id'] for task in tasks_modelB_sorted])

        ##! create the heap with the available machines
        heapq.heapify(machine_availability)
        
        tasks_assignment_B = {} ##! dictionary with the task id and the machine assigned: {task_id: (machine_id, start_time)}
        
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
        
        print(tasks_assignment_A)
        print()
        
        self.tasks_assignment = {**tasks_assignment_A, **tasks_assignment_B}
        
        print(self.tasks_assignment)
        
        
        self.total_makespan = max([self.tasks_assignment[task][1] + self.durations[task-1] for task in self.tasks_assignment])
    
    
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
    
    
    
    def create_output_model(self):
        
        # convert the output data from the file to the minizinc model
        pass



if __name__ == "__main__":
    
    time_start = time.time()
    
    problem = Problem.parse_instance()
    problem.read_input_data()
    problem.convert_input_models()
    
    problem.load_modelA()
    
    # problem.input_data_model_B()
    
    # problem.load_modelB()
    problem.gready_algorithm_modelB()
    
    print("Is solution:", problem.checker_solution())
    
    
    
    time_end = time.time()
    
    print(f"Time: {time_end - time_start}")
    
    # print(problem.input_file_name)
    # print(problem.output_file_name)

