function [inputdata] = load_dataset(dataset_name, opts, varargin)
% [inputdata] = load_dataset(dataset_name, opts...)
%
% Load a dataset
%
% Inputs:
%        dataset_name: Name of the dataset
%
% Options:
%        train_fraction: fraction of the dataset to use for training (default: 0.7)
%        split_method: train/test split method
%          0: first part for training (default)
%          1: last part for training
%          2: middle part for training
%          ...
%        normalize_x: normalize the X values?
%          0: no normalization
%          1: standardize (subtract mean, divide by stddev)  (default for D>1)
%        normalize_y: normalize the Y values?
%          0: no normalization
%          1: standardize
%          2: only scale by powers of 10  (default)
%
% Outputs:
%        inputdata.xfull:  all x values / input values
%        inputdata.xtrain: x values for training
%        inputdata.xtest:  x values for testing
%        inputdata.yfull:  y values / output values
%        inputdata.ytrain: y values for training
%        inputdata.ytest:  y values for testing
%        inputdata.x_unorm: function to unnormalize x values

if nargin < 2
    opts = struct();
elseif nargin > 2
    opts = struct(opts, varargin{:});
end
if ~isfield(opts,'train_fraction'), opts.train_fraction = 0.7; end
if ~isfield(opts,'split_method'), opts.split_method = 0; end
if ~isfield(opts,'NormX') opts.NormX = 1; end
if ~isfield(opts,'NormY') opts.NormY = 2; end

prefile  = ['data/', dataset_name, '-pre.mat'];
filename = ['data/', dataset_name, '-sp', num2str(opts.split_method), '-tr', num2str(opts.train_fraction), '.mat'];

if exist(['data/', dataset_name, '.mat'], 'file') == 2
    load(['data/', dataset_name, '.mat'])
elseif exist(prefile, 'file') == 2
    load(prefile)
    labelSplit = 0;
elseif exist(filename, 'file') == 2
    load(filename)
    labelSplit = 1;
else
    M = csvread(['data/', dataset_name, '.csv'], 1, 0);
    X = M(:, 1:end-1);
    Y = M(:, end);
    labelSplit = 0;
end

if exist('xfull', 'var')
    X = xfull;
    Y = yfull;
elseif exist('xtrain', 'var') && exist('xtest', 'var')
    X = [xtrain; xtest];
    Y = [ytrain; ytest];
else
    labelSplit = 0;
end

[n, D] = size(X);
if D==1 && min(X) > 100
    x_start = min(X)-1;
else
    x_start = 0;
end

if exist('X', 'var')
    % Split the data
    X = X - x_start;
    extrapolateRatio = 0.1;
    trlen = floor(n * opts.train_fraction);
    
    if opts.split_method==0   % ordinary sampling, the first part for training
        xtrain = X(1: trlen, :); ytrain = Y(1: trlen, :);
        xtest = X(trlen+1:end, :); ytest = Y(trlen+1:end, :);
        
    elseif opts.split_method==1  % the last part for training
        xtrain = X(n-trlen+1:n, :); ytrain = Y(n-trlen+1:n, :);
        xtest = X(1: end-trlen, :); ytest = Y(1: end-trlen, :);
        
    elseif opts.split_method==2  % the middle part for training
        xtrain = X(floor((n-trlen)/2)+1:floor((n+trlen)/2), :);
        ytrain = Y(floor((n-trlen)/2)+1:floor((n+trlen)/2), :);
        xtest = X([1: floor((n-trlen)/2), floor((n+trlen)/2)+1:n], :);
        ytest = Y([1: floor((n-trlen)/2), floor((n+trlen)/2)+1:n], :);
        
    elseif opts.split_method==3  % the first and last parts for training
        xtrain = X([1: floor(trlen*0.5), n-floor(trlen*0.5)+1:n], :);
        ytrain = Y([1: floor(trlen*0.5), n-floor(trlen*0.5)+1:n], :);
        xtest = X(floor(trlen*0.5)+1:n-floor(trlen*0.5), :);
        ytest = Y(floor(trlen*0.5)+1:n-floor(trlen*0.5), :);
        
    elseif opts.split_method==4  % random samples for training and extrapolation
        rnglen = floor(n * (1 - extrapolateRatio));
        rng('default')
        randInd = floor(abs(rands(5*rnglen, 1))*rnglen);
        randInd = unique(randInd(randInd~=0), 'stable');  %unique(randInd(randInd~=0));
        %     sortedInd = sort(randInd(1:floor(trlen)));
        sortedInd = sort(randInd(1:floor(trlen)));
        
        xtrain = X(sortedInd, :);
        ytrain = Y(sortedInd, :);
        test_ind = ~ismember(X, sortedInd);
        xtest = X(test_ind, :);
        ytest = Y(test_ind, :);
        
    elseif opts.split_method==5  % really random samples for training
        rng('default')
        randInd = floor(abs(rands(5*n, 1))*n);
        randInd = unique(randInd(randInd~=0), 'stable');  %unique(randInd(randInd~=0));
        sortedInd = randInd(1:floor(trlen));
        
        xtrain = X(sortedInd, :);
        ytrain = Y(sortedInd, :);
        test_ind = ~ismember(X, sortedInd);
        xtest = X(test_ind, :);
        ytest = Y(test_ind, :);
        
    else
        error(['Unknown split_method: ', opts.split_method]);
    end
    
    xfull = X;
    yfull = Y;
end

xtrain_mean = zeros(1, D); xtrain_std = ones(1, D);
ytrain_mean = 0; ytrain_std = 1;

if isequal(opts.normalize_x, 1)
    if D > 1
        xtrain_mean = mean(xtrain);
        xtrain_std = std(xtrain);
    end
end

if isequal(opts.normalize_y, 0)
    % no normalization
elseif isequal(opts.normalize_y, 1)
    ytrain_mean = mean(ytrain);
    ytrain_std = std(ytrain);
elseif isequal(opts.normalize_y, 2)
    % just scale the absolute value of y
    ytrain_mean = 0;
    ytrain_scale = fix(log10(min(ytrain(:))));
    if ytrain_scale > 2, ytrain_std = 10^(ytrain_scale-1); end
end

x_norm = @(xx) (xx - xtrain_mean) ./ xtrain_std;
y_norm = @(yy) (yy - ytrain_mean) ./ ytrain_std;

x_unorm = @(xx) xx .* xtrain_std + xtrain_mean + x_start;
y_unorm = @(yy) yy .* ytrain_std + ytrain_mean;
yVar_unorm = @(yVar) yVar .* (ytrain_std^2);

inputdata.xtrain = x_norm(xtrain);
inputdata.xtest = x_norm(xtest);
inputdata.xfull = x_norm(xfull);

inputdata.ytrain = y_norm(ytrain);
inputdata.ytest = ytest;
inputdata.yfull = yfull;

inputdata.x_unorm = x_unorm;
inputdata.y_unorm = y_unorm;
inputdata.yVar_unorm = yVar_unorm;
end
