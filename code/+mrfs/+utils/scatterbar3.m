function scatterbar3(X,Y,Z,width,ops)
%SCATTERBAR3   3-D scatter bar graph.
%
%   REVISION: Slightly modified by David Inouye (2015) to include 2D bar charts in 3D.
%
%   SCATTERBAR3(X,Y,Z,WIDTH) draws 3-D bars of height Z at locations X and Y with width WIDTH.
%
%   X, Y and Z must be of equal size.  If they are vectors, than bars are placed
%   in the same fashion as the SCATTER3 or PLOT3 functions.
%
%   If they are matrices, then bars are placed in the same fashion as the SURF
%   and MESH functions.
%
%   The colors of each bar read from the figure's colormap according to the bar's height.
%
%   NOTE:  For best results, you should use the 'zbuffer' renderer.  To set the current
%   figure renderer to 'zbuffer' use the following command:
%
%       set(gcf,'renderer','zbuffer')
%
%    % EXAMPLE 1:
%    y=[1 2 3 1 2 3 1 2 3];
%    x=[1 1 1 2 2 2 3 3 3];
%    z=[1 2 3 6 5 4 7 8 9];
%    scatterbar3(x,y,z,1)
%    colorbar
%
%    % EXAMPLE 2:
%    [X,Y]=meshgrid(-1:0.25:1);
%    Z=2-(X.^2+Y.^2);
%    scatterbar3(X,Y,Z,0.2)
%    colormap(hsv)
%
%    % EXAMPLE 3:
%    t=0:0.1:(2*pi);
%    x=cos(t);
%    y=sin(t);
%    z=sin(t);
%    scatterbar3(x,y,z,0.07)

% By Mickey Stahl - 2/25/02
% Engineering Development Group
% Aspiring Developer

if(nargin < 5); ops = []; end
if(~isempty(ops) && isnumeric(ops)) % For backwards compatability
    mod = ops;
    ops = [];
    ops.mod = mod;
end

%{
if(ops.mod == 1)
    Y = Y-1;
elseif(ops.mod == 2)
    X = X-1;
end
%}

if(~isfield(ops,'mod')); ops.mod = 0; end
if(~isfield(ops,'color')); ops.color = 'flat'; end
if(~isfield(ops,'edgeColor')); ops.edgeColor = [0,0,0]; end
if(~isfield(ops,'colorMap')); ops.colorMap = bsxfun(@times, ((1-gray())*0.7+0.3), [0.4 0.4 0.6]/0.6); end
if(~isfield(ops,'percWhite')); ops.percWhite = 0; end
if(~isfield(ops,'freezeXY')); ops.freezeXY = false; end
ops.percWhite = max(min(ops.percWhite, 1),0); % Cap between 0 and 1

ops.maxZ = max(Z(:)); % Needed for color translation
ops.minZ = min(Z(:));
%ops.maxZ = 1;
%ops.minZ = 0;

[r,c]=size(Z);
for j=1:r,
    for k=1:c,
        if ~isnan(Z(j,k))
            drawbar(X(j,k),Y(j,k),Z(j,k),width/2, ops)
        end
    end
end

zl=[min(Z(:)) max(Z(:))];
if zl(1)>0,zl(1)=0;end
if zl(2)<0,zl(2)=0;end
if(~ops.freezeXY)
    xlim([min(X(:))-width max(X(:))+width]);
    ylim([min(Y(:))-width max(Y(:))+width]);
end
if(~all(zl == [0, 0])); zlim(zl); end
%caxis([min(Z(:)) max(Z(:))]);

function drawbar(x,y,z,width, ops)
if(ops.mod == 1)
    h(1)=patch(width.*[-1 -1 1 1]+x,y.*[1 1 1 1],z.*[0 1 1 0],'b');
elseif(ops.mod == 2)
    h(1)=patch(x.*[1 1 1 1],width.*[-1 -1 1 1]+y,z.*[0 1 1 0],'b');
elseif(ops.mod == 3)
    h(1)=patch(width.*[-1 -1 1 1]+x,width.*[-1 -1 1 1]+y,z.*[0 1 1 0],'b');
elseif(ops.mod == 4)
    h(1)=patch(width.*[-1 -1 1 1]+x,width.*[1 1 -1 -1]+y,z.*[0 1 1 0],'b');
elseif(ops.mod == 5)
    h(1)=patch([-width -width width width]+x,[-width width width -width]+y, -0.0001*ones(1,4),'b');
else
    h(1)=patch([-width -width width width]+x,[-width width width -width]+y,[0 0 0 0],'b');
    h(2)=patch(width.*[-1 -1 1 1]+x,width.*[-1 -1 -1 -1]+y,z.*[0 1 1 0],'b');
    h(3)=patch(width.*[-1 -1 -1 -1]+x,width.*[-1 -1 1 1]+y,z.*[0 1 1 0],'b');
    h(4)=patch([-width -width width width]+x,[-width width width -width]+y,[z z z z],'b');
    h(5)=patch(width.*[-1 -1 1 1]+x,width.*[1 1 1 1]+y,z.*[0 1 1 0],'b');
    h(6)=patch(width.*[1 1 1 1]+x,width.*[-1 -1 1 1]+y,z.*[0 1 1 0],'b');
end
if(strcmp(ops.color, 'flat'))
    nColors = size(ops.colorMap,1);
    colorIdx = round((z-ops.minZ)/(ops.maxZ-ops.minZ)*(nColors-1))+1;
    colorIdx = max(min(colorIdx, nColors),0); % Cap at max and min
    color = ops.colorMap(colorIdx,:);
else
    color = ops.color;
end
color = ops.percWhite*[1,1,1] + (1-ops.percWhite)*color;
set(h,'FaceColor', color,'EdgeColor', ops.edgeColor)
