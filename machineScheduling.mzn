% Test Scheduling Problem in MiniZinc

% Data
int: num_tests;    % Number of tests
int: num_machines; % Number of machines
int: num_resources; % Number of global resources
int: num_makespan;

array[1..num_tests] of int: durations; % Duration of each test

array[1..num_tests, 1..num_machines] of bool: machine_allowed; % machine_allowed[i][j] is 1 if test i can be assigned to machine j

array[1..num_tests, 1..num_resources] of bool: resource_required; % resource_required[i][r] is True if test i requires resource r

% Variables
array[1..num_tests] of var 0..num_machines: machine_assigned; % machine_assigned[i] is the machine to which test i is assigned
% LATER restrict the domain

array[1..num_tests, 1..num_resources] of var bool: resource_used; % resource_used[i] indicates the resource for test i (if required)
array[1..num_tests] of var 0..num_makespan: start; % Start time for each test

var 0..num_makespan: makespan; % The makespan (time when the last test finishes)


% input data
%include "dataMachineScheduling.dzn";


% Constraints
%Each test must be assigned to an allowed machine
constraint
    forall(i in 1..num_tests) (
        machine_allowed[i, machine_assigned[i]] = true
    );

%Each resource must be used as specified
% F -> T == T and F -> F == T (solver doesn't care about it if first is false) and T -> F == F, while T -> T
constraint
    forall(i in 1..num_tests, r in 1..num_resources)(
        resource_required[i,r] -> resource_used[i,r] = true
    );

% Tests cannot overlap on the same machine
constraint
    forall(i, j in 1..num_tests where i < j) (
        if machine_assigned[i] = machine_assigned[j] then
            (start[i] + durations[i] <= start[j] \/ start[j] + durations[j] <= start[i])
        else
            true
        endif
    );
    

% Global resources: tests cannot overlap on the same resource
constraint
    forall(r in 1..num_resources, i, j in 1..num_tests where i < j /\ resource_required[i,r] /\ resource_required[j,r]) (
        (start[i] + durations[i] <= start[j] \/ start[j] + durations[j] <= start[i])
    );

% Makespan is the end time of the last test
constraint
    forall(i in 1..num_tests) (
        makespan >= start[i] + durations[i]
    );

%% All the optimization is misssig 

% Objective: Minimize the makespan
solve minimize makespan;
