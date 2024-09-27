% solve the non global resources part of the problem
%% decomposition heuristic

% Input Data
int: num_tests_A;
int: num_tests; 
int: num_machines; % Number of machines
array[1..num_tests] of int: durations;                         % Duration of each global test

int: num_makespan = sum(i in 1..num_tests)(durations[i]);      % Makespan estimate: all tests occur in the same machine
array[1..num_tests] of set of int: machines_allowed;            % ex: [{1, 2}, {1, 2, 3}, {3}, .. ]

int: n_s;

array[1..n_s] of int: s_init_vec;
array[1..n_s] of int: s_end_vec;

array[1..n_s] of int: machines_assigned_A;

%% output the tests that have global resources
array[1..num_tests] of var 1..num_machines: machine_assigned;  % machine_assigned[i] is the machine to which test i is assigned
array[1..num_tests] of var 0..num_makespan: start;             % Start time for each test

var 0..num_makespan: makespan; % The makespan (time when the last test finishes)



% Constraints

% Each test can be assigned to all machines
constraint 
        forall(i in 1..num_tests)(
            machine_assigned[i] in machines_allowed[i]
        );

% Tests cannot overlap in the same machine
% constraint
%     forall(i, j in 1..num_tests where i < j) (
%         if machine_assigned[i] == machine_assigned[j]  then
%             start[i] + durations[i] <= start[j] \/ start[j] + durations[j] <= start[i]
%         else
%             true
%         endif
%     );

constraint
    forall(i, j in 1..num_tests where i < j) (
        if machine_assigned[i] == machine_assigned[j]  then
            start[i] + durations[i] <= start[j]
        else
            true
        endif
    );


constraint
    forall(i in 1..num_tests, j in 1..num_tests_A) (
        if machine_assigned[i] == machines_assigned_A[j] then 
            start[i] + durations[i] <= s_init_vec[j] \/ s_end_vec[j] <= start[i]
        else 
            true
        endif
        );

% constraint
%     forall(i in 1..num_tests, j in 1..num_tests_A) (
%         if machine_assigned[i] == machine_assigned_A[j] then 
                    
%             let {
            
%             int: len = card(s_init_vec[i]),
%             array[1..len] of int: s_init_ordered = [e | e in s_init_vec[m]],
%             array[1..len] of int: s_end_ordered = [e | e in s_end_vec[m]]

%             } in
%             exists(s in 1..len)(start[i] >= s_init_ordered[s] /\ start[i] + durations[i] <= s_end_ordered[s])
%         else 
%             true
%         endif
%         );


%% pseudo-makespan: the quantity of interest is the mininum between this and the model A makespan
constraint
    makespan = max([start[i] + durations[i] | i in 1..num_tests]);



% Objective: Minimize the makespan
solve minimize makespan;