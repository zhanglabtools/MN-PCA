function z = cat_oddeven(x, y)
assert(length(x) == length(y));
z = zeros(length(x) * 2, 1);
z(1:2:end) = x;
z(2:2:end) = y;
end