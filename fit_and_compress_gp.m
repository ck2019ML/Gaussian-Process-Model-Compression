function [hyp_opt, inference, info] = fit_and_compress_gp(inputdata, opts, varargin)
% [hyp_opt, inference, info] = fit_and_compress_gp(inputdata, opts...)
%
% Tune hyperparameters for SM kernel GP on a dataset, starting with multiple random initializations.
% 
% Inputs:
%        inputdata: Training data
%        opts:      A struct with options; or a list of 'key',value pairs
%
% Options:
%        covfunc:          Covariance function, for example @covSM
%        Q                 Number of components
%        num_init          Number of times to try random initialization
%        num_opt           Number of optimization steps to perform
%        pruning           Use pruning? See compress_sm_kernel
%        merging           Use merging? See compress_sm_kernel
%        merge_threshold   Merging threshold. See compress_sm_kernel
%
% Outputs:
%        hyp_opt:   final optimized hyperparameters
%        inference: function to call to run inference
%        info:      extra results for debug purposes.
%         .compression:  info from compress_sm_kernel
%         .init:         info from init_sm
%         .hyp_init:     initial hyperparameters
%         .hyp_init_opt: hyperparameters after initial optimization
%         .NLML_init:    negative log marginal likelilhood on training set, after initial optimization
%         .NLML_opt:     negative log marginal likelilhood on training set, after final optimization

% Options
if nargin < 2
    opts = struct();
elseif nargin > 2
    opts = struct(opts, varargin{:});
end
if ~isfield(opts,'Q'), opts.Q = 10; end
if ~isfield(opts,'num_init'), opts.num_init = 2; end
if ~isfield(opts,'LSM_mode'), opts.LSM_mode = 1; end
if ~isfield(opts,'init_mode'), opts.init_mode = 1; end
if ~isfield(opts,'covfunc'), opts.covfunc = @covSM; end
if ~isfield(opts,'num_opt'), opts.num_opt = -1000; end

Q        = opts.Q;
num_init = opts.num_init;
num_opt  = opts.num_opt;
init_mode = opts.init_mode;

% data
xtrain = inputdata.xtrain;
ytrain = inputdata.ytrain;
[xn, xD] = size(xtrain);

% log
info = struct();
info.opts = opts;

%% 2. repeated random initialization to avoid local optima as much as possible
nlml = Inf;
hyp_init = struct();
hyp_init.lik = log(opts.sn);
for j = 1:num_init
    if init_mode>0
        % TODO: clean up this flag mess:
        win_flag = rem(j, 2);
        
        if Q > 10
            lf = [1 2];
        else
            lf = [1 2];
        end
        log_flag = lf(rem(j, 2) + 1);
        
        if xD>1
            fft_mode=rem(j, 2);
        else
            fft_mode=0;
        end
        
        spec_opt.win_flag = win_flag;
        spec_opt.log_flag = log_flag;
        spec_opt.fft_mode = fft_mode;
        spec_opt.LSM_mode = 1;
        
        [hyp_init.cov, init_info_try] = init_sm(Q, xtrain, ytrain, spec_opt);
        [hyp_init_try, nlml_try, ~] = fit_gp_model({opts.covfunc,Q}, hyp_init, -200, xtrain, ytrain);
        
        if nlml_try < nlml
            info.init = init_info_try;
            info.hyp_init = hyp_init;
            hyp_train = hyp_init_try;
            nlml = nlml_try;
        end
        
    else
        [hyp_init_try, nlml_try, ~] = fit_gp_model({opts.covfunc,Q}, hyp_init, -200, xtrain, ytrain);
        if nlml_try < nlml
            hyp_train = hyp_init_try;
            nlml = nlml_try;
        end
        
        % Use random initialization for next iterations
        hyp_init.lik = log(rand(1));
        hyp_init.cov = log(rand(size(hyp_init.cov)));
        
        s_range = 0;
        win_spec = 0;
    end
end
info.NLML_init = nlml;

% Perform pruning and merging
[hyp_train.cov, Q, info.compression] = compress_sm_kernel(hyp_train.cov, Q, opts);

% (re)optimizing the GP model hyperparameters
[hyp_opt, info.NLML_opt, inference] = fit_gp_model({opts.covfunc,Q}, hyp_train, num_opt, xtrain, ytrain);
hyp_opt.Q = Q;

end


