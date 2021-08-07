function [Y_, iA, iB] = MnPCAq1_wrapper(Y, dim, lam1, lam2, sig, true_Y, n_iter)
%Wrapper function for MN-PCA-w2
%   [Y_, iA, iB] = MnPCAw2_wrapper(Y, dim, lam1, lam2, sig)
%   X             --- the data matrix
%   dim           --- dimension
%   lam1 and lam2 --- l1 regularization parameters.
%   sigma         --- predefined

%for test
% Y = td.Y;
% dim = 2;
% lam1 = 5;
% lam2 = 5;
% sig = 1;
% save data  to .mat
if nargin < 6  
    save('temp/in.mat', 'Y');
else
    save('temp/in.mat', 'Y', 'true_Y');
end
if nargin < 7
    n_iter = 400; 
end
if isunix
    system(sprintf('unset MKL_NUM_THREADS;python MnPCAw2/MnPCAq1.py temp/in.mat temp/out.mat %d %.4f %.4f  -s %.2f -n %d', ... 
           dim , lam1, lam2, sig, n_iter));    
elseif ispc
    cmd_str = sprintf('python MN-PCA-w2/MnPCAq1.py temp/in.mat temp/out.mat %d %.4f %.4f %.2f %d\n', ... 
           dim , lam1, lam2, sig, n_iter);
    fprintf(cmd_str)
    system(sprintf('python MN-PCA-w2/MnPCAq1.py temp/in.mat temp/out.mat %d %.4f %.4f  -s %.2f -n %d', ... 
           dim , lam1, lam2, sig, n_iter));
end
temp = load('temp/out.mat');
Y_ = temp.Y_';
iA = temp.iA; % iA' * iA 
iB = temp.iB;
iA = iA' * iA;
iB = iB' * iB;
% diagonalize
end

function A_ = diagonalize(A)
    u = diag(A);
    u = sqrt(1 ./ u);
    A_ = diag(u) * A * diag(u);
end