% Machine Learning HW#3
% By: Seyed Mohammad Hossein Oloomi

function policy_iteration()

data = get(gcf,'UserData');
% Initialization
policy = zeros(data.rownum, data.colnum, 4);
% Random policy
for i = 1 : data.rownum
    for j = 1 : data.colnum
        if(data.cell_type(i, j) == 2)
            continue;
        end
        policy(i, j, 1) = rand();   % Up action
        remainedProb = 1 - policy(i, j, 1);
        policy(i, j, 2) = remainedProb * rand();  % Right
        remainedProb = 1 - (policy(i, j, 1) + policy(i, j, 2));
        policy(i, j, 3) = remainedProb * rand();  % Down
        policy(i, j, 4) = 1 - (policy(i, j, 1) + policy(i, j, 2) + policy(i, j, 3));    % Left
    end
end

policy_stable = 0;
eval_iter = 0;
improve_iter = 0;
while(policy_stable == 0)
    improve_iter = improve_iter + 1;
    disp(improve_iter);
%% ---------------- Policy Evaluation ---------------
    % V(s)=0 for all s in S
    state_value = zeros(data.rownum, data.colnum);
    delta = 0.011;    
    while(delta > 0.01)
        eval_iter = eval_iter + 1;
        delta = 0;
        % for each s in S
        for i = 1 : data.rownum
            for j = 1 : data.colnum
                if(data.cell_type(i, j) == 2)
                    continue;
                end
                v = state_value(i, j);
                state_value(i, j) = 0;
                % Up Right Down Left
                for action = 1 : 4
                    Qsa = Q_value(i, j, (action + 1), state_value);
                    state_value(i, j) = state_value(i, j) + policy(i, j, action) * Qsa;
                end
                delta = max(delta, abs(v - state_value(i,j)));
            end
        end
    end
%% ---------------- Policy Improvement ---------------
    policy_stable = 1;
    % for each s in S
    for i = 1 : data.rownum
        for j = 1 : data.colnum
            if(data.cell_type(i, j) == 2)
                continue;
            end
            [prob action] = max(policy(i, j, :));
            b = action(1);
            policy(i, j, :) = 0;
            Q = zeros(1,4);
            % Up Right Down Left
            for action = 1 : 4
                Q(action) = Q_value(i, j, (action + 1), state_value);
            end
            [q_val action] = max(Q);
            policy(i, j, action(1)) = 1;
            if( b ~= action(1))
                policy_stable = 0;
            end
        end
    end
end
disp('Policy Evaluation iterations: ');
disp(eval_iter);
disp('Policy Improvement iterations: ');
disp(improve_iter);
% disp('Policy: ');
% disp(policy);
%% ---------------- Output a deterministic policy ----------------
for row = 1 : data.rownum
    for col = 1 : data.colnum        
        if(data.cell_type(row, col) == 2)
            continue;
        end
        [prob action] = max(policy(row, col, :));
        switch action(1)
            case 1
                set(data.cell_handle(row, col), 'String', '<html>&uarr;</html>');
            case 2
                set(data.cell_handle(row, col), 'String', '<html>&rarr;</html>');
            case 3
                set(data.cell_handle(row, col), 'String', '<html>&darr;</html>');
            case 4
                set(data.cell_handle(row, col), 'String', '<html>&larr;</html>');
        end
    end
end


fig2_h=figure;
set(fig2_h,'Units','points')
surf(state_value);
%set(gcf, 'UserData', data);
end


function [qValue] = Q_value(row, col, action, state_value)
x_dir = [0; 1; 0; -1; 0];  %NoMove Up Right Down Left
y_dir = [0; 0; 1; 0; -1];
data = get(gcf,'UserData');
qValue = 0;
for next_state = 1:5
    if(next_state == action)
        Pss = 1 - 4/5 * data.sParameter;
    else
        Pss = data.sParameter / 5;
    end
    % Going out of maze
    if(row + x_dir(next_state) < 1 || row + x_dir(next_state) > data.rownum...
            || col + y_dir(next_state) < 1 || col + y_dir(next_state) > data.colnum)
        qValue = qValue + Pss * (data.rewards(1) + data.discountFactor * state_value(row, col));
        % Normal
    else
        next_state_row = row + x_dir(next_state);
        next_state_col = col + y_dir(next_state);
        next_state_type = data.cell_type(next_state_row, next_state_col);
        % Wall
        if(next_state_type == 2)
            qValue = qValue + Pss * (data.rewards(2) + data.discountFactor * state_value(row, col));
        else
            qValue = qValue + Pss * (data.rewards(next_state_type) + ...
                data.discountFactor * state_value(next_state_row, next_state_col));
        end
    end
end
end