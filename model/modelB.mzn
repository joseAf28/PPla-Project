% solve the symmetric blocks now, which have no restrictions
%% decomposition heuristic

% Input Data
int: num_tests;    % Number of global tests
int: num_machines; % Number of machines
int: num_makespan;

array[1..num_tests] of int: durations;                         % Duration of each global test


%% output the tests that have global resources
array[1..num_tests] of var 1..num_machines: machine_assigned;  % machine_assigned[i] is the machine to which test i is assigned
array[1..num_tests] of var 0..num_makespan: start;             % Start time for each test

var 0..num_makespan: makespan; % The makespan (time when the last test finishes)

int: n_s;

array[1..n_s] of set of int: s_init_vec;
array[1..n_s] of set of int: s_end_vec;


% Constraints

% Each test can be assigned to all machines


% Tests cannot overlap in the same machine
constraint
    forall(i, j in 1..num_tests where i < j) (
        if machine_assigned[i] == machine_assigned[j]  then
            start[i] + durations[i] <= start[j] \/ start[j] + durations[j] <= start[i]
        else
            true
        endif
    );


constraint
    forall(i in 1..num_tests, m in 1..num_machines) (
        if machine_assigned[i] == m then 
                    
            let {
            
            int: len = card(s_init_vec[i]),
            array[1..len] of int: s_init_ordered = [e | e in s_init_vec[m]],
            array[1..len] of int: s_end_ordered = [e | e in s_end_vec[m]]

            } in
            exists(s in 1..len)(start[i] >= s_init_ordered[s] /\ start[i] + durations[i] <= s_end_ordered[s])
        else 
            true
        endif
        );
        
        

%% pseudo-makespan: the quantity of interest is the mininum between this and the model A makespan
constraint
    makespan = max([start[i] + durations[i] | i in 1..num_tests]);



% Objective: Minimize the makespan
solve minimize makespan;