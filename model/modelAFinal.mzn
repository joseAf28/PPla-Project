include "disjunctive.mzn";

%%! decomposition heuristic
%% It only solves the tests with global resources
%% We leave the remaining tests to be solved using a different approach

% Input Data
int: num_tests;         % Number of global tests
int: num_machines;      % Number of machines
int: num_resources;     % Number of global resources

array[1..num_tests] of int: durations;                         % Duration of each global test

int: num_makespan;      % Makespan estimate: all tests with global resources occur in the same machine

array[1..num_resources] of set of int: resources_allowed;       % (t)[{1, 2, 5}, {5, 7}, ..., {9}]
array[1..num_tests] of int: machines_pre_assigned;              % machine_pre_assigned[i] is the machine to which test i is pre-assigned


int: nb_pairs_allowed_to_superpose; % Number of pairs of tests that are allowed to superpose
array[1..nb_pairs_allowed_to_superpose, 1..2] of int: pairs_allowed_to_superpose; % pairs_allowed_to_superpose[i, 1] and pairs_allowed_to_superpose[i, 2] are the indexes of the tests that are allowed to superpose

int: nb_pairs_ordering_same_resource_with_superpose;
array[1..nb_pairs_ordering_same_resource_with_superpose, 1..2] of int: pairs_ordering_same_resource_with_superpose; % pairs_ordering_same_resource_with_superpose[i, 1] and pairs_ordering_same_resource_with_superpose[i, 2] are the indexes of the tests that are allowed to superpose


%% Results
array[1..num_tests] of var 1..num_machines: machine_assigned;  % machine_assigned[i] is the machine to which test i is assigned
array[1..num_tests] of var 0..num_makespan: start;             % Start time for each test

var 0..num_makespan: makespan; % The makespan (time when the last test finishes)


predicate exclusive(set of int: s) =
    disjunctive([start[i] | i in s], [durations[i] | i in s]);



%%! Constraints

constraint
        forall(i in 1..num_tests)(
            machine_assigned[i] == machines_pre_assigned[i]
        );


%% Global resources: tests cannot overlap on the same resource
constraint
        forall(r in 1..num_resources)(
            exclusive(resources_allowed[r]));


%% tests cannot overlap in the same machine
constraint 
        forall(t1 in 1..num_tests, t2 in 1..num_tests where t1 < t2)(
            (machine_assigned[t1] == machine_assigned[t2]) -> 
            (start[t1] + durations[t1] <= start[t2] \/ start[t2] + durations[t2] <= start[t1])
);


%% set of tests that are allowed to superpose in time
constraint
        forall(i in 1..nb_pairs_allowed_to_superpose)(
            start[pairs_allowed_to_superpose[i, 1]] >= start[pairs_allowed_to_superpose[i, 2]] \/
            start[pairs_allowed_to_superpose[i, 1]] + durations[pairs_allowed_to_superpose[i, 1]] <= start[pairs_allowed_to_superpose[i, 2]] + durations[pairs_allowed_to_superpose[i, 2]]
        );


%% set of tests that are not allowed to superpose in time and must be ordered
constraint
        forall(i in 1..nb_pairs_ordering_same_resource_with_superpose)(
            start[pairs_ordering_same_resource_with_superpose[i, 1]] <= start[pairs_ordering_same_resource_with_superpose[i, 2]]
        );


constraint makespan = max([start[i] + durations[i] | i in 1..num_tests]);

constraint makespan <= num_makespan;


% Objective: Minimize the makespan
solve minimize makespan;