function lcs = logsumexp(X, varargin)
% LOGSUMEXP(X, dim) computes log(sum(exp(X), dim)) robustly. Care lightspeed users!
% MODIFIED BY DAVID INOUYE (7-5-16) to omit NaN values
%
%     lse = logsumexp(X[, dim]);
%
% This routine works with general ND-arrays and matches Matlab's default
% behavior for sum: if dim is omitted it sums over the first non-singleton
% dimension.
%
% Note: Tom Minka's lightspeed has a logsumexp function, which:
%     1) sets dim=1 if dim is missing
%     2) returns Inf for sums containing Infs and NaNs;
%
% This routine is fairly fast and accurate for many uses, including when all the
% values of X are large in magnitude. There is a corner case where the relative
% error is avoidably bad (although the absolute error is small), when the
% largest argument is very close to zero and the next largest is moderately
% negative. For example:
%     logsumexp([0 -40])
% Cases like this rarely come up in my work. My LOGPLUSEXP and LOGCUMSUM
% functions do cover this case.
%
% SEE ALSO: LOGCUMSUMEXP LOGPLUSEXP

% Iain Murray, September 2010

% History: IM wrote a bad logsumexp in ~2002, then used Tom Minka's version for
% years until eventually wanting something slightly different.
%
% Copyright license from Ian Murray:
% Permission is hereby granted, free of charge, to any person obtaining
% a copy of this software and associated documentation files (the
% "Software"), to deal in the Software without restriction, including
% without limitation the rights to use, copy, modify, merge, publish,
% distribute, and/or sell copies of the Software, and to permit persons
% to whom the Software is furnished to do so, provided that the above
% copyright notice(s) and this permission notice appear in all copies of
% the Software and that both the above copyright notice(s) and this
% permission notice appear in supporting documentation.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
% EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
% MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
% OF THIRD PARTY RIGHTS. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
% HOLDERS INCLUDED IN THIS NOTICE BE LIABLE FOR ANY CLAIM, OR ANY
% SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER
% RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
% CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
% CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
% 
% Except as contained in this notice, the name of a copyright holder
% shall not be used in advertising or otherwise to promote the sale, use
% or other dealings in this Software without prior written authorization
% of the copyright holder.

if (numel(varargin) > 1)
    error('Too many arguments')
end

if isempty(X)
    % Easiest way to get this trivial but annoying case right!
    lcs = log(sum(exp(X),varargin{:},'omitnan'));
    return;
end

if isempty(varargin)
    mx = max(X);
else
    mx = max(X, [], varargin{:});
end
Xshift = bsxfun(@minus, X, mx);
lcs = bsxfun(@plus, log(sum(exp(Xshift),varargin{:},'omitnan')), mx);

idx = isinf(mx);
lcs(idx) = mx(idx);