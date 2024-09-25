import sys
import re
from minizinc import Instance, Model, Solver


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
    
    
    
    def convert_input_models(self):
        
        dictionaire_machines = {f"m{i+1}": i+1 for i in range(self.num_machines)}
        dictionaire_resources = {f"r{i+1}": i for i in range(self.num_resources)}
        
        setAllMachines = {i+1 for i in range( self.num_machines)}
        
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
        self.have_resources_modelA = [self.have_resources[i-1] for i in self.tests_modelA]
        
        print(self.tests_modelA)
        print(self.num_tests_modelA)
        print(dictionaire_tests_modelA)
        print(self.durations_modelA)
        print(self.machines_allowed_modelA)
        print(self.resources_allowed_modelA)
        print(self.have_resources_modelA)
        
        
        self.tests_unique_machines_no_resources = [i+1 for i in range(self.num_tests) if (self.have_resources[i] == False and len(self.machines_allowed[i]) == 1)]
        
        self.offset_machine = [set() for _ in range(self.num_machines)]
        for test in self.tests_unique_machines_no_resources:
            print(self.machines_allowed[test-1])
            self.offset_machine[list(self.machines_allowed[test-1])[0]-1].add(self.durations[test-1])
        
        self.offset_machine = [sum(offset) for offset in self.offset_machine]
        self.offset_which_machine = [i+1 if self.offset_machine[i] > 0 else 0 for i in range(self.num_machines)]
        
        print()
        print(self.tests_unique_machines_no_resources)
        print(self.offset_machine)
        print(self.offset_which_machine)
        print()


    def load_modelA(self, solver_name="cbc"):
        
        ## load the model and the solver
        model = Model('./model/modelA.mzn')
        solver = Solver.lookup(solver_name)
        instance = Instance(solver, model)
        
        ## load the data into the model
        instance["num_tests"] = self.num_tests_modelA
        instance["num_machines"] = self.num_machines
        instance["num_resources"] = self.num_resources
        
        instance["durations"] = self.durations_modelA
        
        instance["machines_allowed"] = self.machines_allowed_modelA
        instance["resources_allowed"] = self.resources_allowed_modelA
        instance["have_resources"] = self.have_resources_modelA
        
        instance["offset_machine"] = self.offset_machine
        instance["offset_which_machine"] = self.offset_which_machine
        
        ## solve the model
        self.result = instance.solve()
        
        print(self.result)
        
        
        self.makespan_A = self.result["makespan"]
        self.machines_assigned_A = self.result["machines_assigned"]
        self.start_times_A = self.result["start_times"]
    
    
    
    def create_output_model(self):
        
        # convert the output data from the file to the minizinc model
        pass



if __name__ == "__main__":
    
    problem = Problem.parse_instance()
    problem.read_input_data()
    problem.convert_input_models()
    
    problem.load_modelA()
    
    # print(problem.input_file_name)
    # print(problem.output_file_name)

