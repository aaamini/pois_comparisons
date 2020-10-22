function rootsMatT = findRoots( ddgCoeffMat, JNz )
n = size(ddgCoeffMat,1);
rootsMatT = NaN(length(JNz), n); % Roots plus endpoints
if(max(JNz) == 2) % Simple SQR case
    rootsMatT = NaN(length(JNz), n); % Roots plus endpoints
    temp = -ddgCoeffMat(:,2)./ddgCoeffMat(:,1);
    rootsMatT(temp > 1) = temp(temp > 1).^2;
    return;
end

% Get recursvie LCM and polynomial so y = x^(1/base)
base = 1; for jj = 1:length(JNz); base = lcm(base, JNz(jj)); end
polyCoeffI = [base./JNz,0]; % Polynomial coefficient for each JNz y^?
polyDeg = max(polyCoeffI);

% Setup variables for loop
ddgCoeffMatT = ddgCoeffMat';
C = zeros(polyDeg+1,1);
Cidx = polyDeg+1-polyCoeffI;

% Loop through each and compute the roots
for i = 1:size(ddgCoeffMat,1)
    C(Cidx) = ddgCoeffMatT(:,i);
    assert(all( ~isnan(C) | ~isinf(C) ), 'Inf or NaN in C');
    ry = roots(C);

    % Transform roots back into space of x = y^base (only keep real)
    temp = real(ry(imag(ry)==0 & ry > 1)).^base;
    rootsMatT(1:length(temp),i) = temp;
end

%assert(all(rootsMatT(:)==rootsMatT2(:) | (isnan(rootsMatT(:)) & isnan(rootsMatT2(:)))), 'checking');

end