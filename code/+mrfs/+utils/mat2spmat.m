function spMat = mat2spmat( A )
%MAT2SPMAT Summary of this function goes here
%   Detailed explanation goes here
[I,J,S] = find(A);
%spMat.I = I;
%spMat.J = J;
%spMat.S = S;
spMat.IJSt = [I,J,S]';
spMat.nnz = length(S);
temp = find(diff(J))+1;
spMat.colStarts = [1; temp; repmat( length(J)+1, size(A,2)-length(temp), 1)];

end

