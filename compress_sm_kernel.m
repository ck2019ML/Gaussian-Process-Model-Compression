function [hyp_cp, Q_cp, info] = compress_sm_kernel(hyp, Q, opts, varargin)
% [hyp_cp, Q_cp, info] = compress_sm_kernel(hyp, Q, opts...)
%
% Compress the hyperparameters of an Spectral Mixture kernel (@covSM)
%
% Inputs:
%        hyp:     hyperparameters of SM kernel
%        Q:       number of components
%        opts:    A struct with options; or a list of 'key',value pairs
%
% Options:
%        merge_threshold: merge component that have at least this much similarity. Default: 0.65
%        pruning:         use pruning?
%          0:               no pruning
%          1:               yes (default)
%        merging:         use merging?
%          0 / 'no':        no merging
%          1 / 'pairwise':  iterative pairwise
%          2 / 'greedy':    greedy merging (default)
%
% Outputs:
%        hyp_cp:  compressed hyperparameters
%        Q_cp:    number of components after compression
%        info:    struct with
%          .sorted:  sorted hyperparameters, as a struct with .w, .m, .v
%          .pruned:  pruned hyperparameters
%          .merged:  merged hyperparameters
%          .compressed: final hyperparameters
%
% Hyperparameters are represented as in GPML's covSM:
%    hyp = [ log(w(:)), log(m(:)), log(sqrt(v(:))) ]

if nargin < 3
    opts = struct();
elseif nargin > 3
    opts = struct(opts, varargin{:});
end
if ~isfield(opts,'merge_threshold'), opts.merge_threshold = 0.65; end
if ~isfield(opts,'pruning'), opts.pruning = 'auto'; end
if ~isfield(opts,'merging'), opts.merging = 'greedy'; end

% Decode hyperparameters
hyps = decode_sm_hyperparameters(hyp, Q);

info = struct();
info.merge_threshold = opts.merge_threshold;
info.pruning = opts.pruning;
info.merging = opts.merging;

% Sort by weight
[~,order] = sort(hyps.w, 'descend');
hyps.w = hyps.w(:,order);
hyps.m = hyps.m(:,order);
hyps.v = hyps.v(:,order);
info.sorted = hyps;

% Pruning
if isequal(opts.pruning, 0) || isequal(opts.pruning, 'no')
    % no pruning
else
    hyps = prune_sm_components(hyps, opts.pruning);
    info.pruned = hyps;
end

% Merging
if isequal(opts.merging, 1) || isequal(opts.merging, 'pairwise')
    hyps = merge_sm_components_multi_pass(hyps, opts.merge_threshold);
    info.merged = hyps;
elseif isequal(opts.merging, 2) || isequal(opts.merging, 'greedy')
    hyps = merge_sm_components_greedy(hyps, opts.merge_threshold);
    info.merged = hyps;
elseif isequal(opts.merging, 0) || isequal(opts.merging, 'no')
    % no merging
else
    error(['Unknown merging mode: ', opts.merging]);
end

info.compressed = hyps;

% Re-encode hyperparameters
[hyp_cp, Q_cp] = encode_sm_hyperparameters(hyps);
end


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

function [hyp, Q] = encode_sm_hyperparameters(hyps)
    Q = numel(hyps.w);
    hyp = [ log(hyps.w(:)),
            log(hyps.m(:)),
            log(hyps.v(:)) * 0.5 ];
end

