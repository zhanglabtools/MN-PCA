function metrics = helpEval(metric_names, true, pred)
% Help function to evaluate the output of matrix.
% metrics = HELPEVAL(metric_names, true, pred)
%   metric_names: cells of metrics names
%   true: structure contanis true signal
%   pred: structure contains predicted signal
metrics = zeros(length(metric_names), 1);
for i = 1:length(metric_names)
    if strcmp(metric_names{i}, 'frob')
        metric= norm(true.signal - pred.signal, 'fro');
    elseif strcmp(metric_names{i}, 'rmse')
        se = (true.signal - pred.signal).^2; 
        metric = sqrt(mean(se(:)));
    elseif strcmp(metric_names{i}, 'psnr')
        se = (true.signal - pred.signal).^2;
        mse = mean(se(:));
        metric = 10 * log10(max(true.signal(:))^2 /mse);
    elseif strcmp(metric_names{i}, 'nmi')
        metric = mean(kmeansEval(true.nclust, pred.score, true.labels, 20));
    % Evluate precion matrix estimation of iA
    elseif strcmp(metric_names{i}, 'tpr1')
        if ~exist('out_prec1', 'var')
            out_prec1 = evalPREC(true.iA, pred.iA);
        end
        metric = out_prec1.tpr;
    elseif strcmp(metric_names{i}, 'tnr1')
        if ~exist('out_prec1', 'var')
            out_prec1 = evalPREC(true.iA, pred.iA);
        end
        metric = out_prec1.tnr;
    elseif strcmp(metric_names{i}, 'prec1')
        if ~exist('out_prec1', 'var')
            out_prec1 = evalPREC(true.iA, pred.iA);
        end
        metric = out_prec1.precision;           
    % Evlautuation precision matrix of iB
    elseif strcmp(metric_names{i}, 'tpr2')
        if ~exist('out_prec2', 'var')
            out_prec2 = evalPREC(true.iB, pred.iB);
        end        
        metric = out_prec2.tpr;
    elseif strcmp(metric_names{i}, 'tnr2')
        if ~exist('out_prec2', 'var')
            out_prec2 = evalPREC(true.iB, pred.iB);
        end
        metric = out_prec2.tnr;
    elseif strcmp(metric_names{i}, 'prec2')
        if ~exist('out_prec2', 'var')
            out_prec2 = evalPREC(true.iB, pred.iB);
        end
        metric = out_prec2.precision;
    elseif strcmp(metric_names{i}, 'norm1')
        metric = norm(true.iA - pred.iA);
    elseif strcmp(metric_names{i}, 'norm2')
        metric = norm(true.iB - pred.iB);    
    else
        error('Metric %s not implemented\n', metric_names{i});
    end
    metrics(i) = metric;
end


end

