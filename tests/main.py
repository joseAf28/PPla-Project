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
        self.non_machine = [tests_values[i][2] for i in range(len(tests_values))]
        self.resources = [tests_values[i][3] for i in range(len(tests_values))]
    
    
    
    def convert_input_model(self):
        
        string_machines = [f"m{i+1}" for i in range(self.num_machines)]
        string_resources = [f"r{i+1}" for i in range(self.num_resources)]
        
        self.matrix_machines_allowed = [[False for _ in range(self.num_machines)] for _ in range(self.num_tests)]
        self.matrix_resources = [[False for _ in range(self.num_resources)] for _ in range(self.num_tests)]
        
        for i in range(self.num_tests):
            for j in range(self.num_machines):
                if string_machines[j] in self.non_machine[i] or self.non_machine[i] == ['e']:
                    self.matrix_machines_allowed[i][j] = True
        
        for i in range(self.num_tests):
            for j in range(self.num_resources):
                if string_resources[j] in self.resources[i]:
                    self.matrix_resources[i][j] = True
                    
                    
        print(self.matrix_machines_allowed)
        print("\n")
        print(self.matrix_resources)
    
    
    
    def load_model(self, solver_name="cbc"):
        
        ## load the model and the solver
        model = Model('./model/machineScheduling.mzn')
        solver = Solver.lookup(solver_name)
        instance = Instance(solver, model)
        
        ## load the data into the model
        instance["num_tests"] = self.num_tests
        instance["num_machines"] = self.num_machines
        instance["num_resources"] = self.num_resources
        
        instance["durations"] = self.durations
        instance["machine_allowed"] = self.matrix_machines_allowed
        instance["resource_required"] = self.matrix_resources
        
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
    
    # print(problem.input_file_name)
    # print(problem.durations)
    # print(problem.non_machine)
    # print(problem.resources)
    # print(problem.commented_values)
    # print(problem.tests_values)
    
    problem.load_model()
    
    # print(problem.input_file_name)
    # print(problem.output_file_name)

