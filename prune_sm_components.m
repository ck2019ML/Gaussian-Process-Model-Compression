function [hyps,selected] = prune_sm_components(hyps, mode)
% Perform pruning on trained SM kernel.
%
% Inputs:
%        hyps: hyperparameters as a decoded struct
%        mode: how to prune?
%          'no':         no pruning
%          'threshold':  keep components with weight >= 1
%          'energy':     keep components with 99% of total weight
%          'auto':       automatically decide between the two (default)
%
% Outputs:
%        hyps: hyperparameters after pruning
%        selected: indicator vector of components to keep

train_w = hyps.w;
Q = numel(train_w);

% Energy based pruning

energeSum = 0;
init_win_max = zeros(1, Q);
[sort_w, sort_ind] = sort(train_w, 'descend');
wPer = sort_w ./ sum(train_w);

for q=[1: Q]
    if energeSum > 0.99
        disp([num2str(q-1), ' components containing 99% variance'])
        break
    else
        energeSum = wPer(:, q) + energeSum;
        init_win_max(:, sort_ind(:, q)) = 1;
    end
end
init_win_max = logical(init_win_max);

% Threshold based pruning

init_win_zero = train_w >= 1;

% Automatic choice

if isequal(mode, 'auto') || isequal(mode, 1)
    sum_zero = sum(init_win_zero);
    sum_max = sum(init_win_max);
    fprintf('Pruning: %d to %d (threshold) or %d (energy based)\n', Q, sum_zero, sum_max);
    if (sum_max > 3 && sum_max < sum_zero) || sum_zero==0
        mode = 'energy'
    else
        mode = 'threshold';
    end
end

if isequal(mode, 'no') || isequal(mode, 0)
    % No pruning
elseif isequal(mode, 'energy')
    disp('Prune: keep top 99% of total weight');
    selected = init_win_max;
elseif isequal(mode, 'threshold')
    disp('Prune: keep components with weight >= 1');
    selected = init_win_zero;
else
    error(['Unknown pruning mode: ', mode]);
end

hyps.w = hyps.w(:, selected);
hyps.m = hyps.m(:, selected);
hyps.v = hyps.v(:, selected);
end
