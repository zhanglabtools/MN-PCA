function metrics = evalLR(true, pred)
%EVALLR evaluate the recovry of the low rank.
%   metrics = evalLR(truelr, predlr)
se = (true - pred).^2;
metrics.mse = mean(se(:));
metrics.frob = norm(true - pred, 'fro');
metrics.psnr = 10 * log10(max(true(:))^2 / metrics.mse);
end

