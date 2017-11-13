% Machine Learning HW#3
% By: Seyed Mohammad Hossein Oloomi

function monte_carlo()
data = get(gcf,'UserData');

% Q(s,a) = 0 for all s in S, a in A(S)
Qsa = zeros(data.rownum, data.colnum, 4);
QsaCount = zeros(data.rownum, data.colnum, 4);

% Random e-soft policy
policy = zeros(data.rownum, data.colnum, 4);
for i = 1 : data.rownum
    for j = 1 : data.colnum
        if(data.cell_type(i, j) == 2)
            continue;
        end
        policy(i, j, 1) = 0.25;        
        policy(i, j, 2) = 0.25;
        policy(i, j, 3) = 0.25;
        policy(i, j, 4) = 0.25;
    end
end

x_dir = [0; 1; 0; -1; 0];  %NoMove Up Right Down Left
y_dir = [0; 0; 1; 0; -1];

% Repeat forever
for iter = 1 : 10000
    % Generating an episode using policy
    % Episode length
    len = ceil(rand() * data.episodLength);
    % Episode States list
    states_row = zeros(1, len);
    states_col = zeros(1, len);
    % Episode Actions list
    actions = zeros(1, len - 1);
    % Episode Rewards
    rewards = zeros(1, len - 1);    
    % S1
    row = ceil(rand() * data.rownum);
    col = ceil(rand() * data.colnum);
    % if start state is selected in wall, change it
    while(data.cell_type(row, col) == 2)
        row = ceil(rand() * data.rownum);
        col = ceil(rand() * data.colnum);
    end
    states_row(1) = row;
    states_col(1) = col;
    for s = 1 : (len - 1)
        state_policy = [policy(row,col,1) policy(row,col,2) policy(row,col,3) policy(row,col,4)];
        state_policy = state_policy * triu(ones(4,4));
        action = find(rand() < state_policy);
        % action
        actions(s) = action(1);
        
        next_state_prob = zeros(1,5);
        next_state_prob(:) = data.sParameter / 5;
        next_state_prob(action(1) + 1) = 1 - 4/5 * data.sParameter;
        next_state_prob = next_state_prob * triu(ones(5,5));        
        next_state = find(rand() < next_state_prob);        
        
        row = row + x_dir(next_state(1));
        col = col + y_dir(next_state(1));
        if(row < 1 || row > data.rownum || col < 1 || col > data.colnum || data.cell_type(row, col) == 2)
            row = row - x_dir(next_state(1));
            col = col - y_dir(next_state(1));
        end
        % next state
        states_row(s + 1) = row;
        states_col(s + 1) = col;        
        % reward
        rewards(s) = data.rewards(data.cell_type(row, col));
    end
    
    power = 0 : 1 : len-2;
    discount_vector = data.discountFactor .^ power;       
    returns = zeros(1, len-1);    
    QsaFirstVisit = ones(data.rownum, data.colnum, 4);
    % For each pair s,a appearing in the episode:
    for s = 1 : len-1
        if(QsaFirstVisit(states_row(s), states_col(s), actions(s)) == 0)
            continue;
        end
        prevReturn = Qsa(states_row(s), states_col(s), actions(s));
        prevCount = QsaCount(states_row(s), states_col(s), actions(s));
        returns(s) = sum((rewards(s : len-1) .* discount_vector(1 : len-s)));
        % Q(s,a) = average(Returns(s,a))
        Qsa(states_row(s), states_col(s), actions(s)) = (prevCount * prevReturn + returns(s)) / (prevCount + 1);
        QsaCount(states_row(s), states_col(s), actions(s)) = prevCount + 1;
        QsaFirstVisit(states_row(s), states_col(s), actions(s)) = 0;
    end
    
    epsilon = 0.1;
    % for each s in the episode
    for s = 1 : len-1
        [q i] = max(Qsa(states_row(s), states_col(s), :));
        policy(states_row(s), states_col(s), :) = epsilon / 4;
        policy(states_row(s), states_col(s), i(1)) = 1 - 3/4 * epsilon;
    end
    iter
end

% Calculating states values
state_value = zeros(data.rownum, data.colnum);
for row = 1 : data.rownum
    for col = 1 : data.colnum
        if(data.cell_type(row, col) == 2)
            continue;
        end
        [q a] = max(Qsa(row, col, :));
        state_value(row, col) = (1-epsilon) * q + epsilon / 4 * sum(Qsa(row, col, :));
    end
end

for row = 1 : data.rownum
    for col = 1 : data.colnum
        if(data.cell_type(row, col) == 2)
            continue;
        end
        [q a] = max(Qsa(row, col, :));
        switch a(1)
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