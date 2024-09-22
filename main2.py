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
        
    
    
    def convert_input_model(self):
        
        dictionaire_machines = {f"m{i+1}": i+1 for i in range(self.num_machines)}
        dictionaire_resources = {f"r{i+1}": i for i in range(self.num_resources)}
        
        setAllMachines = {i+1 for i in range( self.num_machines)}
        self.machines_allowed = []
        
        for i in range(self.num_tests):
            if self.machines[i] == ['e']:
                self.machines_allowed.append(setAllMachines) ## probably this ones is default, no need to add it
            else:
                self.machines_allowed.append({dictionaire_machines[machine] for machine in self.machines[i]})
    
        
        self.resources_allowed = [ set() for _ in range(self.num_resources)]
        self.have_resources = [False for _ in range(self.num_tests)]
        
        for i in range(self.num_tests):
            for j in range(self.num_resources):
                if f'r{j+1}' in self.resources[i]:
                    self.resources_allowed[j].add(i+1)
                    self.have_resources[i] = True


    def load_model(self, solver_name="cbc"):
        
        ## load the model and the solver
        model = Model('./model/machineScheduling2.mzn')
        solver = Solver.lookup(solver_name)
        instance = Instance(solver, model)
        
        ## load the data into the model
        instance["num_tests"] = self.num_tests
        instance["num_machines"] = self.num_machines
        instance["num_resources"] = self.num_resources
        
        instance["durations"] = self.durations
        
        instance["machines_allowed"] = self.machines_allowed
        instance["resources_allowed"] = self.resources_allowed
        instance["have_resources"] = self.have_resources
        
        ## solve the model
        self.result = instance.solve()
        
        print(self.result)
    
    
    
    def create_output_model(self):
        
        # convert the output data from the file to the minizinc model
        pass


if __name__ == "__main__":
    
    problem = Problem.parse_instance()
    problem.read_input_data()
    problem.convert_input_model()
    
    problem.load_model()
    
    # print(problem.input_file_name)
    # print(problem.output_file_name)

