% Machine Learning HW#3
% By: Seyed Mohammad Hossein Oloomi

function value_iteration()
data = get(gcf,'UserData');

% V(s)=0 for all s in S
state_value = zeros(data.rownum, data.colnum);
delta = 0.011;
iter = 0;
while(delta > 0.01)
    iter = iter + 1;
    delta = 0;
    % for each s in S
    for i = 1 : data.rownum
        for j = 1 : data.colnum
            if(data.cell_type(i, j) == 2)
                continue;
            end
            v = state_value(i, j);
            Q = zeros(1,4);
            % Up Right Down Left
            for action = 1 : 4
                Q(action) = Q_value(i, j, (action + 1), state_value);
            end
            state_value(i, j) = max(Q);
            delta = max(delta, abs(v - state_value(i,j)));
        end
    end
%     disp(state_value);
%     disp('----------------------------');
end
disp('Number of iterations = ');
disp(iter);

% for i = 1 : data.rownum
%     for j = 1 : data.colnum
%         set(data.cell_handle(i, j), 'String' , num2str(state_value(i, j)));
%     end
% end
% set (gcf, 'UserData', data);

% Output a deterministic policy
x_dir = [1; 0; -1; 0];  %Up Right Down Left
y_dir = [0; 1; 0; -1];
for row = 1 : data.rownum
    for col = 1 : data.colnum
        if(data.cell_type(row, col) == 2)
            continue;
        end
        q_value = zeros(1,4);
        for action = 1 : 4
            next_row = row + x_dir(action);
            next_col = col + y_dir(action);
            if(next_row < 1 || next_row > data.rownum || next_col < 1 || next_col > data.colnum)
                q_value(action) = - realmax;
            else
                q_value(action) = state_value(next_row, next_col);
            end
        end
        [max_q max_a] = max(q_value);
        switch max_a(1)
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
set(gcf, 'UserData', data);

fig2_h=figure;
set(fig2_h,'Units','points')
surf(state_value);
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

