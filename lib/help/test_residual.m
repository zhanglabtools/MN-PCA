function Out = test_residual(R)
%UNTITLED5 Summary of this function goes here
[Out.T1, Out.p1] = hypotest_cov(corr(R), size(R, 1));
[Out.T2, Out.p2] = hypotest_cov(corr(R'), size(R, 2));
end

