include "globals.mzn";

% solve the global resources and restricted resources/machines tests first
%% only includes tests with global resources
%% include later also the tests with restricted resources
%% leave the tests thaht can occur in any machine for the end
%% decomposition heuristic

% Input Data
int: num_tests;    % Number of global tests
int: num_machines; % Number of machines
int: num_resources; % Number of global resources

array[1..num_tests] of int: durations;                         % Duration of each global test

int: num_makespan = sum(i in 1..num_tests)(durations[i]);      % Makespan estimate: all tests occur in the same machine
% possible conflict with the makespan

array[1..num_tests] of set of int: machines_allowed;            % ex: [{1, 2}, {1, 2, 3}, {3}, .. ]
array[1..num_resources] of set of int: resources_allowed;       % (t)[{1, 2, 5}, {5, 7}, ..., {9}]

array[1..num_tests] of bool: have_resources;

array[1..num_machines] of int: offset_machine;
array[1..num_machines] of int: offset_which_machine;


% set of int: test_resources_no_restriction_machines;

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
    forall(r in 1..num_resources)(exclusive(resources_allowed[r]));


constraint forall(i in 1..num_tests, m in 1..num_machines)(
            if machine_assigned[i] == offset_which_machine[m] then 
                start[i] >= offset_machine[m]
            else
            true
            endif
        );       

% constraint forall(i in 1..num_tests-1)(machine_assigned[i] <= machine_assigned[i+1]);


% tests cannot overlap in the same machine, not used as we only consider the restriction in which the tests can be assigned to a single machine
% possible problem if all machines have a restriction
% constraint
%     forall(i, j in 1..num_tests where i < j) (
%         if machine_assigned[i] = machine_assigned[j] /\ have_resources[i] = false /\ have_resources[j] = false  then
%             start[i] + durations[i] <= start[j]
%         elseif machine_assigned[i] = machine_assigned[j] then
%             (start[i] + durations[i] <= start[j] \/ start[j] + durations[j] <= start[i])
%         else
%             true
%         endif
%     );

constraint
    makespan = max([start[i] + durations[i] | i in 1..num_tests]);



% Objective: Minimize the makespan
% solve :: int_search(start, first_fail, indomain_min, complete) minimize makespan;

solve minimize makespan;


%% modelA also fails to solve the problem when the number of resources inscreases a lot
%% - grouping resources in a single set


%% think about the symmetry breaking of the machines