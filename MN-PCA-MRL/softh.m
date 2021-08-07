function y = softh(x, lam)
%Softhreholding function 

y = max(abs(x) - lam, 0) .* sign(x);


end

