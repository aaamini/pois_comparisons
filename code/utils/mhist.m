function mhist(x, labels, xLabelStr, titleStr, labelPos)
if nargin < 5, labelPos = 'on'; end

switch labelPos
case 'on'
    % these go well with a 6 x 8in figure
    margin_left = 0.01;
    label_xpos = 0.02;
    label_ypos = 0.9;
    margin_bottom = 0.07;
    margin_top = 0.04;
case 'left'
    margin_left = 0.12;
    label_xpos = -0.12;
    label_ypos = 0.5;
    margin_bottom = 0.08;
    margin_top = 0.05;
end
% plot multiple histograms
x_nrow = size(x,1)
nbins = floor(x_nrow/5);
nbins = min(nbins, 3);
nbins = max(nbins, round(30*log10(x_nrow)))

hist_args = {nbins,'Normalization','pdf','EdgeColor','none','FaceColor',0.1*ones(1,3)};
%  if(size(x,1) > 10^2)
%      hist_args(1) = []; % Auto choose bin size
% %     % hist_args(1) = 200; % Auto choose bin size
%  end

% labelFontSize = 10;
labelFontSize = 14;
maxXlim = quantile(x(:),1);
nrows = size(x,2);
% figure(1), clf
ax = NaN(nrows,1);
for k = 1:nrows
    xCur = x(:,k);
    ax(k) = subaxis(nrows,1,k,'SpacingVert',0.01,'MR',0.02,'ML',margin_left,'MB', margin_bottom,'MT', margin_top); % subplot(nrows,1,k);
    hh(k) = histogram(xCur,hist_args{:},'parent',ax(k) );
    
    temp = get(hh(k),'Values');
    maxYlim = max(temp(2:end))*1.1;
    set(ax(k),'xlim',[0,maxXlim], 'ylim',[0,maxYlim]);

    hold on;
    LW = 2;
    plot(nanmean(xCur)*ones(2,1), [0,maxYlim],'-r',...
        'LineWidth', LW,...
        'parent', ax(k),...
        'color','r');
    plot(quantile(xCur,0)*ones(2,1), [0,maxYlim],':b',...
        'LineWidth', LW,...
        'parent', ax(k),...
        'color','b');
    plot(quantile(xCur,0.5)*ones(2,1), [0,maxYlim],'--b',...
        'LineWidth', LW,...
        'parent', ax(k),...
        'color','b');
    plot(quantile(xCur,1)*ones(2,1), [0,maxYlim],':b',...
        'LineWidth', LW,...
        'parent', ax(k),...
        'color','b');
    hold off;
    
   
    set(ax(k), 'FontSize',labelFontSize-2);
    text(label_xpos, label_ypos, labels{k}, 'Units','Normalized','FontSize',labelFontSize+1, 'Color', 'black')
    if (k == nrows), xlabel(xLabelStr,'FontSize',labelFontSize+1); end
    if (k == 1), title(titleStr, 'FontSize',labelFontSize+2), end
end
set(ax(1:nrows-1,:),'xticklabel','')
set(ax(:),'yticklabel','','ytick',[])

