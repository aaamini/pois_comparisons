function marginal3(X,Y,Z,ops)
hold on;
if(isvector(X))
    scatter3(X,Y,Z,[],Z,'.');
    line(X,Y,Z,'Color',[0.5,0.5,0.5]);
    
else
    H = surf(X,Y,Z);
    colormap(ops.colorMap);
    shading interp;
    set(H,'LineStyle','-','EdgeAlpha',0.2,'EdgeColor','k');
end
hold off;

end

