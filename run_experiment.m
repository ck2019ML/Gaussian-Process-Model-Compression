function [result, info] = run_experiment(dataset_name, opts, varargin)
% [result, info] = run_experiment(dataset_name, opts...)
%
% Run an experiment.
% For example:
%        run_experiment('CO2data', 'Q',10, 'pruning',0);
%
% Inputs:
%        dataset_name: Name of the dataset
%        opts:         A struct with options; or a list of 'key',value pairs
%
% Options:
%        Q:                Number of kernel components
%        pruning           Use pruning? See compress_sm_kernel
%        merging           Use merging? See compress_sm_kernel
%        merge_threshold   Merging threshold. See compress_sm_kernel
%        num_init:         Number of time to repeat initialization
%        train_fraction:   Fraction of the dataset to use for training
%        split_method:     Data split method. See load_dataset

%% Options
if nargin < 2
    opts = struct();
elseif nargin > 2
    opts = struct(opts, varargin{:});
end

% data options
opts.normalize_x = 1;
opts.normalize_y = 2;
if ~isfield(opts, 'split_method'), opts.split_method = 0; end
if ~isfield(opts, 'train_fraction')
    if isequal(dataset_name, 'CO2data')
        opts.train_fraction = 0.6;
    else
        opts.train_fraction = 0.7;
    end
end

% optimization options
opts.covfunc = @covSM;
opts.sn = 0.5;
if ~isfield(opts, 'num_init'), opts.num_init = 2; end
if ~isfield(opts, 'num_opt'),  opts.num_opt  = -1000; end
if ~isfield(opts, 'Q'),        opts.Q = 10; end;
if ~isfield(opts, 'pruning'),  opts.pruning = 'auto'; end;
if ~isfield(opts, 'merging'),  opts.merging = 'greedy'; end;

%date_flag = datestr(now,'-yyyymmddHHMMSS');
disp([dataset_name, ', Q=', num2str(opts.Q)]);

%% Load data
inputdata = load_dataset(dataset_name, opts);

%% Fit model
[hyp, inference, info] = fit_and_compress_gp(inputdata, opts);

%% Compute prediction and metrics
[mean_pred, Var_pred] = inference(inputdata.xtest);
result = metrics(inputdata, hyp, mean_pred, Var_pred);
result.NLML_init = info.NLML_init;
result.NLML_opt  = info.NLML_opt;
result.Q_init    = opts.Q;
result.Q_final   = hyp.Q;

%% (optional) display/visualize/save results
disp(result);

end


function [testResults] = metrics(inputdata, hyp_opt, mean_pred, Var_pred)
    % Compute metrics
    ytrain = inputdata.ytrain;
    ytest = inputdata.ytest;

    % un-normalize y values
    ytrain = inputdata.y_unorm(ytrain);
    ytest = inputdata.y_unorm(ytest);
    mean_pred = inputdata.y_unorm(mean_pred);
    Var_pred = inputdata.yVar_unorm(Var_pred);

    mae = @(a,b) mean(abs(a-b));
    mse = @(a,b) mean((a-b).^2);

    MAE_test = mae(mean_pred, ytest);
    MSE_test = mse(mean_pred, ytest);
    SMSE_test = MSE_test / var(ytest);
    MSLL_test = msll(ytrain, ytest, mean_pred, Var_pred, hyp_opt.lik);

    testResults.MAE_test = MAE_test;
    testResults.MSE_test = MSE_test;
    testResults.SMSE_test = SMSE_test;
    testResults.MSLL_test = MSLL_test;
end

function [MSLL_test] = msll(ytrain, ytest, mPred, VarPred, sn)
    % Compute the standardized log loss (SLL) using predictive loss and empirical loss

    mean_tr = mean(ytrain);
    var_tr = var(ytrain);

    n = length(ytest);

    % the predictive loss
    pred_var = VarPred + exp(2*sn);
    pred_loss = -log(normpdf(ytest, mPred, sqrt(pred_var)) + realmin);
    pred_loss = sum(pred_loss);

    empirical_loss = sum(log(normpdf(ytest, mean_tr, sqrt(var_tr))));
    MSLL_test = (pred_loss + empirical_loss) / n;
end

