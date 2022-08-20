function [hyps, Qlabel] = merge_sm_components_multi_pass(hyps, dep_thd, max_passes)
% Perform component merging
%
% Inputs:
%        hyps: hyperparameters of the kernel, as a decoded struct
%        dep_thd: similarity threshold for merger
%        max_passes: maximum number of passes
%
% Outputs:
%        hyps: hyperparameters after merging

if nargin < 3, max_passes = inf; end

changes = 1;
pass = 0;

while changes && pass < max_passes
    changes = 0;
    pass = pass + 1;
    
    [D,Q] = size(hyps.m);

    Qlabel = ones(1, Q);
    alldep = reshape(component_similarities(hyps.w, hyps.m, hyps.v), Q, []);
    tril_dep = tril(alldep) - diag(Qlabel);

    [ind_r, ind_c] = find(tril_dep > dep_thd);
    bigdep = alldep(tril_dep > dep_thd);
    [~, sort_ind] = sort(bigdep, 'descend');

    ind_r = ind_r(sort_ind);
    ind_c = ind_c(sort_ind);

    hyp_w = hyps.w;
    hyp_m = hyps.m;
    hyp_v = hyps.v;

    if Q>2
        for di = [1:numel(sort_ind)]
            
            i = ind_r(sort_ind(di));
            w1 = hyp_w(i);
            m1 = hyp_m(:, i);
            v1 = hyp_v(:, i);
            
            j = ind_c(sort_ind(di));
            w2 = hyp_w(j);
            m2 = hyp_m(:, j);
            v2 = hyp_v(:, j);
            
            if Qlabel(i)==1 && Qlabel(j)==1
                w_m = w1 + w2;
                m_m = (w1.*m1 + w2.*m2) ./ w_m;
                v_m = (w1.*(v1 + (m1-m_m).^2) + w2.*(v2 + (m2-m_m).^2)) ./ w_m;
                
                changes = 1;
                Qlabel(i) = 2;  %% label 2 has merged hyp
                Qlabel(j) = 0;  %% label 0 has hyp to be merged
                %% label 1 has raw hyp
            end
        end
    end

    Qlabel = logical(Qlabel);
    hyps.w = hyps.w(:, Qlabel);
    hyps.m = hyps.m(:, Qlabel);
    hyps.v = hyps.v(:, Qlabel);
    
    if changes
        fprintf('Merging to %d components\n', length(hyps.w));
    end
end
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

    wij = reshape( bsxfun(@times, reshape(w1,Q1,1),   reshape(w2,1,Q2)),   1, Q1, Q2);
    mij = reshape( bsxfun(@minus, reshape(m1,D,Q1,1), reshape(m2,D,1,Q2)), D, Q1, Q2);
    vij = reshape( bsxfun(@times, reshape(v1,D,Q1,1), reshape(v2,D,1,Q2)), D, Q1, Q2);
    vivj = reshape(bsxfun(@plus,  reshape(v1,D,Q1,1), reshape(v2,D,1,Q2)), D, Q1, Q2);
    K1 = sqrt(prod(sqrt(4*vij)./vivj, 1)) .* exp(-0.25.*sum((mij.^2)./vivj, 1));
end
