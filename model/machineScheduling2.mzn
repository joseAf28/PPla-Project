include "globals.mzn";

% Data
int: num_tests;    % Number of tests
int: num_machines; % Number of machines
int: num_resources; % Number of global resources

array[1..num_tests] of int: durations;                         % Duration of each test
int: num_makespan = sum(i in 1..num_tests)(durations[i]);      % Makespan estimate: all tests occur in the same machine

array[1..num_tests] of set of int: machines_allowed;            % ex: [{1, 2}, {1, 2, 3}, {3}, .. ]
array[1..num_resources] of set of int: resources_allowed;       % (t)[{1, 2, 5}, {5, 7}, ..., {9}]


array[1..num_tests] of var 1..num_machines: machine_assigned;  % machine_assigned[i] is the machine to which test i is assigned
array[1..num_tests] of var 0..num_makespan: start;             % Start time for each test

var 0..num_makespan: makespan; % The makespan (time when the last test finishes)


array[1..num_tests] of bool: have_resources; % use to check if we can define an order inside each machine

%% example as sample for now:
array[1..2] of set of int: groups = [{1, 6}, {2, 4}];


predicate exclusive(set of int: s) =
    let {array[int] of var int: x = [start[i] | i in s],
         array[int] of  int : d = [durations[i] | i in s]} in
     disjunctive(x, d);


% Constraints
%Each test must be assigned to an allowed machine
constraint 
        forall(i in 1..num_tests)(
            machine_assigned[i] in machines_allowed[i]
        );


% Tests cannot overlap on the same machine
% it can improved by specifying an order of the tests as long as they are assigned to the same machine and don't share resources
%constraint
%    forall(i, j in 1..num_tests where i < j) (
%        if machine_assigned[i] = machine_assigned[j] then
%            (start[i] + durations[i] <= start[j] \/ start[j] + durations[j] <= start[i])
%        else
%            true
%        endif
%    );
    

constraint
    forall(i, j in 1..num_tests where i < j) (
        if machine_assigned[i] = machine_assigned[j] /\ have_resources[i] = false /\ have_resources[j] = false then
            (start[i] + durations[i] <= start[j])
        elseif machine_assigned[i] = machine_assigned[j] then
              (start[i] + durations[i] <= start[j] \/ start[j] + durations[j] <= start[i])
        else
            true
        endif
    );

% Global resources: tests cannot overlap on the same resource
% disjunctive constraint
constraint
    forall(r in 1..num_resources)(exclusive(resources_allowed[r]));
    
    

%% seems not to help at least so far
%constraint
%   forall(i in 1..2)(
%        forall (j, k in groups[i] where j < k)(
%         start[j] + durations[j] <= start[k])
%   );


constraint
    makespan = max([start[i] + durations[i] | i in 1..num_tests]);


%%! All the optimization is misssig 
%% tasks in the same machine and don't share resources can be ordered
%% tasks that share resources can be ordered if they are similar (same duration, same machine and/or resources)
%% restrictions on the domain of the variables
%% dominance empty spaces

% Objective: Minimize the makespan
solve minimize makespan;