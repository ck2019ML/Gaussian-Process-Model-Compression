function [hyps] = decode_sm_hyperparameters(hyp, Q, D)
    % Decode hyperparameters of covSM kernel
    if nargin < 3
        D = floor(numel(hyp) / (2*Q+1));
    end
    hyps = struct();
    hyps.w = exp(reshape(  hyp(         1:1*Q) , 1, Q));             % mixture weights
    hyps.m = exp(reshape(  hyp(1*Q+    (1:D*Q)), D, Q));             % spectral means
    hyps.v = exp(reshape(2*hyp(1*Q+D*Q+(1:D*Q)), D, Q));             % spectral variances
end
