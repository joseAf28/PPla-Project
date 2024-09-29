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
        
        
        
        find_gaps = []
        for group in new_sorted_data:
            for i in range(len(group)-1):
                if group[i][1] != group[i+1][0]:
                    find_gaps.append((group[i][1], group[i+1][0], group[i][2]))
        
            if len(group) == 1 and group[0][0] != 0:
                find_gaps.append((0, group[0][0], group[0][2]))
                find_gaps.append((group[0][1], self.max_number, group[0][2])) ## 9999 not important
            elif len(group) == 1 and group[0][0] == 0:
                find_gaps.append((group[0][1], self.max_number, group[0][2])) ## 9999 not important
            else:
                find_gaps.append((group[-1][1], self.max_number, group[-1][2]))
        
        machine_availability = {machine: [] for machine in range(1, self.num_machines+1)}
        
        for gap in find_gaps:
            machine_availability[gap[2]].append(gap)
        
        ## insert the remaining available time for each machine
        for machine in machine_availability:
            if len(machine_availability[machine]) == 0:
                machine_availability[machine].append((0, self.max_number, machine))
        
        
        print("machine_availability: ", machine_availability)
        
        ## assign tests with restrions fisrst
        
        tasks_machine_restrictions_modelB = [ {'id': number, 'duration': duration, 'machines': machines} for number, duration, machines in zip(self.tests_modelB, self.durations_modelB, self.machines_allowed_modelB) if len(machines) != self.num_machines]
        tasks_machine_no_restrictions_modelB = [ {'id': number, 'duration': duration, 'machines': machines} for number, duration, machines in zip(self.tests_modelB, self.durations_modelB, self.machines_allowed_modelB) if len(machines) == self.num_machines]
        
        tasks_machine_modelB = tasks_machine_restrictions_modelB + tasks_machine_no_restrictions_modelB
    
    
    
    
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
    
    
    
    time_end = time.time()
    
    print(f"Time: {time_end - time_start}")
    
    # print(problem.input_file_name)
    # print(problem.output_file_name)

