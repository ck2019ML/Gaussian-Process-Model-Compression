function [hyp_opt, NLML_opt, predict_fun] = fit_gp_model(covfunc, hyps, opt_num, xtrain, ytrain)
% Hyper-parameter optimization and inference for a Gaussian Process model.
% For large datasets (>=1000 samples) uses Aggeration GP.
%
% Inputs:
%        covfunc: covariance function
%        hyps:    initial values of hyperparameters
%        opt_num: number of optimization iterations
%        xtrain: n*d matrix of training set x values
%        ytrain: 1*d vector of training set y values
%
% Outputs:
%        hyp_opt: optimal hyperparameters
%        NLML_opt: negative log marginal likelihood on the train set (objective value)
%        predict_fun: function such that
%          [mean_pred, Var_pred] = predict_fun(xtest)
%          are predicted mean and variance on a test set

[xn, ~] = size(xtrain);
lik = {@likGauss};

if xn>=1000
    % train, Y is yes for normalizing
    BCMopts.Xnorm = 'N' ; BCMopts.Ynorm = 'N' ; 
    partitionCriterion = 'random' ; BCMopts.Ms = floor(xn / 200); 
    BCMopts.partitionCriterion = partitionCriterion ;
    BCMopts.cov = hyps.cov ; BCMopts.covfunc = covfunc;
    % opts.covfunc = @covSM;
    
    criterion = 'RBCM' ;
    BCMopts.meanfunc = [];
    BCMopts.likfunc = lik; BCMopts.inffunc = @infGaussLik;
    BCMopts.numOptFC = opt_num; BCMopts.sn = hyps.lik; 
        
    [models, NLML_opt, hyp_opt, ~] = aggregation_train(xtrain, ytrain, BCMopts);    
    predict_fun = @(xtest) aggregation_predict(xtest, models, criterion);
    
else
    hyps.cov(isinf(hyps.cov))=1e-8;
    hyps.cov(isnan(hyps.cov))=1e-8;
    
    [hyp_opt, fX, ~] = minimize(hyps, @gp, opt_num, @infExact, [], covfunc, lik, xtrain, ytrain);
    NLML_opt = gp(hyp_opt, @infExact, [], covfunc, lik, xtrain, ytrain);
     
    predict_fun = @(xtest) gp(hyp_opt, @infExact, [], covfunc, lik, xtrain, ytrain, xtest);
end
end
