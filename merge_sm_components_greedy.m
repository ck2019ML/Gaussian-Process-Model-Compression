function [hyps] = merge_sm_components_greedy(hyps, merge_threshold)
% Perform component merging, in a greedy way
%
% Inputs:
%        hyps: hyperparameters of the kernel: struct with w,m,v matrices
%              see decode_hyperparameters
%        merge_threshold: similarity threshold for merger, components are merged if their similarity is >= this threshold
%
% Outputs:
%        hyps: hyperparameters after merging

w = hyps.w;
m = hyps.m;
v = hyps.v;

similarities = component_similarities(w,m,v);
similarities = tril(similarities, -1);

while length(similarities) > 1
    % Find most similar components
    [maxsim,is] = max(similarities);
    [sim,j] = max(maxsim);
    i = is(j);
    % Merge?
    if sim >= merge_threshold
        assert(i ~= j);
        if j < i
            temp = i; i = j; j = temp;
        end
        % Component i becomes merged component
        [w_m,m_m,v_m] = merge_components(w(:,i),m(:,i),v(:,i), w(:,j),m(:,j),v(:,j));
        w(:,i) = w_m;
        m(:,i) = m_m;
        v(:,i) = v_m;
        w(:,j) = [];
        m(:,j) = [];
        v(:,j) = [];
        % Update similarities
        similarities(j,:) = [];
        similarities(:,j) = [];
        new_sims = component_similarities(w_m,m_m,v_m, w,m,v);
        similarities(i,1:i-1)   = new_sims(1:i-1);
        similarities(i+1:end,i) = new_sims(i+1:end)';
    else
        break
    end
end

% Result
hyps.w = w;
hyps.m = m;
hyps.v = v;
end

function [w_m,m_m,v_m] = merge_components(w1,m1,v1, w2,m2,v2)
    % Merge two components
    w_m = w1 + w2;
    m_m = (w1.*m1 + w2.*m2) ./ w_m;
    v_m = (w1.*(v1 + (m1-m_m).^2) + w2.*(v2 + (m2-m_m).^2)) ./ w_m;
end

function [K1] = component_similarities(w1,m1,v1, w2,m2,v2)
    % Compute similarities between all pairs of kernel components
    if nargin < 4
      w2 = w1;
      m2 = m1;
      v2 = v1;
    end

    D = size(m1,1);
    Q1 = size(m1,2);
    Q2 = size(m2,2);

    mij  = reshape(bsxfun(@minus, reshape(m1,D,Q1,1), reshape(m2,D,1,Q2)), D, Q1, Q2);
    vij  = reshape(bsxfun(@times, reshape(v1,D,Q1,1), reshape(v2,D,1,Q2)), D, Q1, Q2);
    vivj = reshape(bsxfun(@plus,  reshape(v1,D,Q1,1), reshape(v2,D,1,Q2)), D, Q1, Q2);
    K1 = sqrt(prod(sqrt(4*vij)./vivj, 1)) .* exp(-0.25.*sum((mij.^2)./vivj, 1));
    disp(size(K1));
    K1 = reshape(K1, Q1, Q2);
end

