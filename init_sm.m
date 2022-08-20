function [hypinit, spec_fit] = init_sm(Q, xtrain, ytrain, options)
% [hypinit, spec_fit] = init_sm(Q, xtrain, ytrain, options)
%
% Initialize hyperparameters of the SM kernel, based on empirical spectral density
%
% Inputs:
%        Q: number of components
%        xtrain: X values of training set
%        ytrain: Y values of training set
%        options: struct of options
%          .log_flag: Use log of empirical density?
%            0: Use empirical density without log
%            1: Average the weight (signal variance of signal) and randomly initialize mean position
%            2: fitting the whole log empirical spectral density using mixture of mode
%          .win_flag: Windowing?
%            0: No windowing
%            1: Blackman window
%            2: Guassian window
%            3: Hamming window
%            4: Hann window
%            5: Rectangular window
%            6: Triangular window
%          .fft_mode: apply FFT on each dimension seperately or jointly(image)
%            0: 
%            1: 
%          .LSM_mode: kernel type
%
% Outputs:
%        hypinit: initial values for hyperparameters
%        spec_fit: Information about intialization procedure

%% flags

log_flag = options.log_flag;
win_flag = options.win_flag;
fft_mode = options.fft_mode;
lsm_mode = options.LSM_mode;

[N, D] = size(xtrain); M = floor(N/2);
freq_range = [[0:M],[-M+1:1:-1]]'/N;  % frequency range
freq_range = freq_range(1:M+1);

fftdata = zeros(N, D);
the_log_density = zeros(M+1, D);
the_win_density = zeros(M+1, D);
avg_weight = std(ytrain)./Q;

%% empirical spectral densities
if win_flag==1
    win = exp(blackman(N))/exp(1); % blackman window
elseif win_flag==2
    win = exp(gausswin(N))/exp(1); % gausswin(N) window
elseif win_flag==3
    win = exp(hamming(N))/exp(1); % hamming(L) window
elseif win_flag==4
    win = exp(hann(N))/exp(1); % hann(L) window
elseif win_flag==5
    win = exp(rectwin(N))/exp(1); % Computes a rectangular window.
elseif win_flag==6
    win = exp(triang(N))/exp(1); % Computes a triangular window.
else
    win = 1;  % emp_win = abs(fft(ytrain)).^2/N;
end

w = zeros(1, Q) + 1e-8; % create hypers
m = zeros(D, Q) + 1e-8;
s = zeros(D, Q) + 1e-8;
sk = rand(D, Q) + 1e-8;

if lsm_mode ~= 3
    hypinit = zeros(Q+2*D*Q, 1);  % create initialisation vector of all hypers
end

%% apply FFT on each dimension seperately or jointly(image), which means dimensions are independent
%  three kinds of GMM on spectral density, 1:y, 2:joint d like image, 3:each d
%  if fft_mode==2
for d=[1:D]
    fftdata_d = ytrain;
    if D==1     % univariate data or multivariate data
        big_var = d;
    else        % multivariate data
        disp('FFT on each dimension of multivariate data')
        xtrain_std = std(xtrain);
        big_var = find(xtrain_std == max(xtrain_std));
    end
    
    %% Fourier transform for the observed values
    fftdata_f = fft(fftdata_d);
    emp_spec = fftdata_f.*conj(fftdata_f)./N;
    %     emp_spec = abs(fft(fftdata_d)).^2/N;  % /N; does not necessary divide N, especially for large N
    log_emp_spec = abs(log(emp_spec + 1));
    
    fftdata_wf = fft(fftdata_d .* win);
    emp_win = fftdata_wf.*conj(fftdata_wf)./N;  % /N; does not necessary divide N, especially for large N
    log_emp_win = abs(log(emp_win));
    
    emp_spec = emp_spec(1:M+1);
    log_emp_spec = log_emp_spec(1:M+1);
    log_emp_win = log_emp_win(1:M+1);
    
    if log_flag == 0
        disp('Empirical density used!')
        the_log_density_d = emp_spec;
        the_win_density_d = log_emp_win;
        
    elseif log_flag > 0 && win_flag == 0  % variation in magnitude is so large, show the log of the magnitude in the Fourier transform.
        disp('Log of empirical density used!')
        the_log_density_d = log_emp_spec;
        the_win_density_d = log_emp_win;
        
    elseif log_flag > 0 && win_flag ~= 0
        disp('Log of windowed FFT used!')
        the_log_density_d = log_emp_win;
        the_win_density_d = log_emp_spec;
    end
    
    the_log_density(:, d) = the_log_density_d;
    the_win_density(:, d) = the_win_density_d;
    
    if log_flag == 1  % (log_flag == 1 || log_flag == 2) && Q>1
        %% Average the weight (signal variance of signal) and randomly initialize mean position
        %rng('shuffle')
        w = w + ones(1, Q) .* rand(1,Q); % create hypers
        d2 = sqrt(sq_dist(xtrain(:,d)'));   % get distances for each input dimension
        
        if (N>1)
            d2(d2 == 0) = d2(1,2);
        else
            d2(d2 == 0) = 1;
        end
        
        minshift = min(min(d2));
        nyquist = 0.5/minshift;
        %rng('shuffle')
        m(d,:) = nyquist*rand(1,Q);    % draw frequencies from Uniform(0,Nyquist)
        
        maxshift = max(max(d2));
        %rng('shuffle')
        s(d,:) = 1./abs(maxshift*randn(1,Q));
        
        fitQ = 20; % only fit limited numbers.        
        if Q > 20
            [obj] = fitGMM(fitQ, freq_range, the_log_density_d);
            if d==big_var
                w(1, 1:20) = obj.ComponentProportion;
            end
            
            m(d, 1:20) = obj.mu;
            %     m = repmat(obj.mu, [1, D])';
            s(d, 1:20) = sqrt(reshape(obj.Sigma, 1, fitQ));     %/N%(x(end)-x(1));    % Check this line!
            %     s = repmat(sqrt(reshape(obj.Sigma,1,Q))', [1, D])';
        end
        
    elseif log_flag == 2
        %% fitting the whole log empirical spectral density using mixture of mode
        disp('Fitting the whole log empirical spectral density')
        [obj] = fitGMM(Q, freq_range, the_log_density_d);
        
        if d==big_var
            w(1,:) = obj.ComponentProportion;
        end
        
        m(d,:) = obj.mu;
        %     m = repmat(obj.mu, [1, D])';
        s(d,:) = sqrt(reshape(obj.Sigma, 1, Q));     %/N%(x(end)-x(1));    % Check this line!
        %     s = repmat(sqrt(reshape(obj.Sigma,1,Q))', [1, D])';
    end
    
    if D>1 && fft_mode==0
        break
    elseif D>1 && fft_mode==1
        disp('Copy empirical spectral density to each dimension')
        m = repmat(m(d, :), D, 1);
        s = repmat(s(d, :), D, 1);
        sk = repmat(sk(d, :), D, 1);
        break
    end
end

hypinit(1:Q) = log(std(fftdata_d) .* w(:));
hypinit(Q+(1:Q*D)) = log(m(:));

if lsm_mode==3
    hypinit(Q+Q*D+(1:Q*D)) = log(sk(:));
    hypinit(Q+2*Q*D+(1:Q*D)) = log(s(:));
else
    hypinit(Q+Q*D+(1:Q*D)) = log(s(:));
end

hypinit(isinf(hypinit))=1e-8;
hypinit(isnan(hypinit))=1e-8;

spec_fit.spec_range = freq_range;
spec_fit.temp_log_spec = the_log_density;
spec_fit.temp_win_spec = the_win_density;
end


function [obj] = fitGMM(Q, freq_range, the_log_density_d)
% Fit a Gaussian Mixture Model

total_area = trapz(freq_range, the_log_density_d);
spec_cdf = cumtrapz(freq_range, the_log_density_d);
spec_cdf = spec_cdf ./total_area;

nsamp = 1e4; a = rand(nsamp, 1);
invsamps = zeros(numel(a),1);
for i=1:numel(a)
    invsamps(i) = inv_spec_cdf(a(i), freq_range, spec_cdf);
end

if exist('OCTAVE_VERSION', 'builtin') ~= 0
    statset = @struct;
end

options = statset('Display','final','MaxIter', 1000);
obj = gmdistribution.fit(invsamps, Q, 'Start', 'plus', 'CovarianceType', ...
    'diagonal', 'Options', options, 'Replicates', Q);
end
