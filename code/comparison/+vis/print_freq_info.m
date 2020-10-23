function print_freq_info(Xt)
% Print info. about level frequencies of matrix Xt

freqs = full(tabulate(Xt(:)));
fprintf('# of levels = %d\n', size(freqs,1))
fprintf('------------------\n')
max_levels = min(size(freqs,1), 8);
[~ , fidx] = sort(freqs(:, 3), 1, 'descend');
freqs = freqs(fidx, :);
fprintf('Level    Freq.\n')
fprintf('------------------\n')
for i = 1:max_levels
    fprintf('%5g    %5g%%\n', freqs(i, 1), round(freqs(i, 3),2))
end
if (size(freqs,1) > max_levels), fprintf('   (truncated)  \n'), end
