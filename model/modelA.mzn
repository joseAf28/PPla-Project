include "globals.mzn";

%% solve the global resources
%% only includes tests with global resources
%% leave the remaining tests to be solved using a different approach
%% decomposition heuristic

% Input Data
int: num_tests;    % Number of global tests
int: num_machines; % Number of machines
int: num_resources; % Number of global resources

array[1..num_tests] of int: durations;                         % Duration of each global test

int: num_makespan = sum(i in 1..num_tests)(durations[i]);      % Makespan estimate: all tests occur in the same machine
% Think about a more advanced way of calculating the makespan

array[1..num_tests] of set of int: machines_allowed;            % ex: [{1, 2}, {1, 2, 3}, {3}, .. ]
array[1..num_resources] of set of int: resources_allowed;       % (t)[{1, 2, 5}, {5, 7}, ..., {9}]

array[1..num_machines] of int: offset_machine;
array[1..num_machines] of int: offset_which_machine;

%% output the tests that have global resources
array[1..num_tests] of var 1..num_machines: machine_assigned;  % machine_assigned[i] is the machine to which test i is assigned
array[1..num_tests] of var 0..num_makespan: start;             % Start time for each test

var 0..num_makespan: makespan; % The makespan (time when the last test finishes)



predicate exclusive(set of int: s) =
    disjunctive([start[i] | i in s], [durations[i] | i in s]);


% Constraints
%Each test must be assigned to an allowed machine
constraint 
        forall(i in 1..num_tests)(
            machine_assigned[i] in machines_allowed[i]
        );


% Global resources: tests cannot overlap on the same resource
constraint
        forall(r in 1..num_resources)(
            exclusive(resources_allowed[r]));


% offset in the machine schedule in the case that a test must be assigned to a specific machine
constraint 
        forall(i in 1..num_tests, m in 1..num_machines)(
            (machine_assigned[i] == offset_which_machine[m]) ->
            (start[i] >= offset_machine[m]));


% tests cannot overlap in the same machine
constraint 
        forall(t1 in 1..num_tests, t2 in 1..num_tests where t1 < t2)(
            (machine_assigned[t1] == machine_assigned[t2]) -> 
            (start[t1] + durations[t1] <= start[t2] \/ start[t2] + durations[t2] <= start[t1])
);


% constraint forall(i in 1..num_tests-1)(machine_assigned[i] <= machine_assigned[i+1]);


constraint makespan = max([start[i] + durations[i] | i in 1..num_tests]);



% Objective: Minimize the makespan
% solve :: int_search(start, first_fail, indomain_min, complete) minimize makespan;
solve minimize makespan;



%% Problems / Ideas to think about later

%% modelA also fails to solve the problem when the number of resources inscreases a lot
%% - grouping resources in a single set

%% think about the symmetry breaking of the machines