function  markers = gen_data_marker(dnames)
%Generate data markser

n = length(dnames);
sqrt_n = ceil(sqrt(n));
marker_types = {'o', '*', 'd', 'p', 'h', '+',  'v', '>'}; 
marker_colors = cbrewer('qual', 'Set1', sqrt_n); 
markers.types = cell(n, 1);
markers.colors = cell(n, 1);
for idx_mk = 1:sqrt_n
    for idx_c = 1:sqrt_n
        idx_data = (idx_mk - 1) * sqrt_n + idx_c;
        if idx_data > n
            return;
        end
        markers.types{idx_data} = marker_types{idx_mk};
        markers.colors{idx_data} = marker_colors(idx_c, :);
    end
end

