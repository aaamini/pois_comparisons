classdef RegionMat
    % Fairly simple override of behavior for indexing notation
    
    properties
        eta1
        concavity
        
        logU
        logL
        
        minX
        maxX
        evalMinX
        evalMaxX
        fgMin
        fgMax
        gMin
        gMax
        dgMin
        dgMax
        
        bu
        cu
        bl
        cl
    end
    
    methods
        function obj = RegionMat(eta1)
            if(nargin < 1); mrfs.grm.univariate.Poisson(); return; end
            obj.eta1 = eta1;
            obj.logU = NaN(size(eta1));
            obj.logL = NaN(size(eta1));
            obj.concavity = NaN(size(eta1));
            obj.minX = NaN(size(eta1));
            obj.maxX = NaN(size(eta1));
            obj.evalMinX = NaN(size(eta1));
            obj.evalMaxX = NaN(size(eta1));
            obj.fgMin = NaN(size(eta1));
            obj.fgMax = NaN(size(eta1));
            obj.gMin = NaN(size(eta1));
            obj.gMax = NaN(size(eta1));
            obj.dgMin = NaN(size(eta1));
            obj.dgMax = NaN(size(eta1));
            obj.bu = NaN(size(eta1));
            obj.cu = NaN(size(eta1));
            obj.bl = NaN(size(eta1));
            obj.cl = NaN(size(eta1));
        end
        
        function varargout = size(obj,varargin)
            varargout = cell(max(nargout,1),1);
            [varargout{:}] = size(obj.eta1,varargin{:});
        end
        
        function obj = transpose(obj)
            for prop = fieldnames(obj)'
                obj.(prop{1}) = obj.(prop{1}).';
            end
        end
        
        function obj = ctranspose(obj)
            for prop = fieldnames(obj)'
                obj.(prop{1}) = obj.(prop{1})';
            end
        end

        function TF = isempty(obj)
            if(isempty(obj.eta1))
                TF = true;
            else
                TF = false;
            end
        end
    
        function visualize( o, a, eta )
            if(size(o,1) > 1)
                warning('Cannot visualize more than 1 at this time (aborting visualization)');
                return;
            end
            
            % Visualize true line
            [M, fCell, xCell] = mrfs.grm.univariate.Poisson.exactM( a, eta );
            minF = max(fCell{1})-20;
            maxIdx = find(fCell{1}>=minF,1,'last');
            x = xCell{1}(1:maxIdx);
            fCell{1} = fCell{1}(1:maxIdx);
            shift = x*eta(1)-gammaln(x+1);
            
            % Compute region
            k = size(eta,2);
            f = @(x) x.^(1./(1:k))*eta' - gammaln(x+1) + log(x.^a);
            %sel = false(size(x));
            %sel(x==0) = true;
            logU = NaN(size(x)); logL = NaN(size(x));
            logU(x==0) = log(0^a); logL(x==0) = log(0^a);
            nRegions = size(o.minX,2);
            for ri = 1:nRegions
                if isnan(o.minX(ri)); continue; end
                % Determine domain and update sel
                leftX = ceil(o.minX(ri));
                rightX = min(ceil(o.maxX(ri))-1,max(x));
                %sel(x >= leftX & x <= rightX) = true;
                
                % Add exact at left endpoint
                logU(x==leftX) = f(leftX);
                logL(x==leftX) = f(leftX);
                
                % Add linear approximation till right endpoint
                etaHatU = o.eta1(ri) + o.bu(ri);
                etaHatL = o.eta1(ri) + o.bl(ri);
                for q = (leftX+1):rightX
                    logU(x==q) = o.cu(ri) + etaHatU*q - gammaln(q+1);%log(poisspdf(q,AhatU));
                    logL(x==q) = o.cl(ri) + etaHatL*q - gammaln(q+1);%log(poisspdf(q,AhatL));
                end
                % Special case for num = 2 or 3
                if(rightX-leftX <= 2)
                    for curX = leftX:rightX
                        logU(x==curX) = f(curX);
                        logL(x==curX) = f(curX);
                    end
                end
            end
            
            if(any(isnan(logU)))
                Mu = NaN; Ml = NaN;
            else
                nQueryCur = max(sum(~isnan(o.minX)) + 1, size(eta,2)+1);
                [Mu,Ml] = mrfs.grm.univariate.Poisson.approxM( a, eta, nQueryCur );
            end
            
            % Display approximations (with calculations)
            subplot(2,1,1);
            otherOps = {'MarkerSize',8};
            plot(x, exp(logU), '-s', otherOps{:});
            hold on;
            plot(x, exp(fCell{1}), '-x', otherOps{:});
            plot(x, exp(logL), '-o', otherOps{:});
            yl = ylim();
            for ri = 1:nRegions
                if isnan(o.minX(ri)); continue; end
                minX = min(ceil(o.minX(ri)), max(x)+1)-0.5;
                plot(minX*ones(2,1),yl,':k');
            end
            ylim(yl);
            hold off;
            
            % Display title and approximations
%             title( sprintf('M = %g(=%g), Mu=%g(=%g), Ml=%g(=%g)', ...
%                 M, mrfs.utils.logsumexp(fCell{1}),...
%                 Mu, mrfs.utils.logsumexp(logU),...
%                 Ml, mrfs.utils.logsumexp(logL)...
%                 ) );
            %title( sprintf('M=%g, Mu=%g, Ml=%g, (Mu-M)/M=%g, (Mu-Ml)/Ml=%g', ...
            %    M, Mu, Ml, (Mu-M)/abs(M), (Mu-Ml)/abs(Ml) ) );
            title('Node Conditional Density Bounds exp(f(x)+g(x))');
            legend({'Upper','Exact','Lower'},'Location','Best');
            
            % Plot approximation of root functions
            subplot(2,1,2);
            % Correct for -Inf at 0 (thus shift doesn't recover original approximation)
            logUshift = logU-shift;
            logLshift = logL-shift;
            fShift = fCell{1}-shift;
            plot(x, logUshift, '-', otherOps{:});
            hold on;
            plot(x, fShift, '-', otherOps{:});
            plot(x, logLshift, '-', otherOps{:});
            yl = ylim();
            for ri = 1:nRegions
                if isnan(o.minX(ri)); continue; end
                minX = min(ceil(o.minX(ri)), max(x)+1)-0.5;
                maxX = min(ceil(o.maxX(ri))-1, max(x)+1)-0.5;
                plot(minX*ones(2,1),yl,':k');
            end
            hold off;
            legend({'Upper','Exact','Lower'},'Location','Best');
            title('Linear Upper and Lower Bounds of g(x)');
            xl = xlim();
            
            %{
            subplot(3,1,3);
            cla;
            hold on;
            plot(0,[0,0,0]); % To shift colors
            for ri = 1:nRegions
                if isnan(o.minX(ri)); continue; end
                minX = min(ceil(o.minX(ri)), max(x)+1)-0.5;
                maxX = min(ceil(o.maxX(ri))-1, max(x)+2)-0.5;
                bar(mean([minX,maxX]),exp(o.logU(ri))-exp(o.logL(ri)));
            end
            yl = ylim();
            for ri = 1:nRegions
                if isnan(o.minX(ri)); continue; end
                minX = min(ceil(o.minX(ri)), max(x)+1)-0.5;
                plot(minX*ones(2,1),yl,':k');
            end
            hold off;
            xlim(xl);
            title('Diff between bounds');
            %}
            %fprintf('min inclusive; max inclusive; logU; logL\n');
            %disp([ceil(o.minX);ceil(o.maxX)-1;o.logU;o.logL]);
            %fprintf('x; fCell{1}\n');
            %disp([x';fCell{1}']);
            pause(0.3);
        end
    end
end