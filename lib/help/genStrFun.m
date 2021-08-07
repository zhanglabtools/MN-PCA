function f = genStrFun(rep, varargin)
% Generate a function that return a structure.
    res.metrics_names = varargin;
    res.evaluation = zeros(rep, length(varargin));
    f = @() res;
end