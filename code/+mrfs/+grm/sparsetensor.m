classdef sparsetensor < handle
    % Simple implementation of sparse tensor
    properties (GetAccess = public, SetAccess = protected)
        sz;
        spVec;
        isScalar = false;
    end
    
    methods
        %% Constructor
        function obj = sparsetensor( sz )
            if(isempty(sz));
                sz = 1;
            end
            if(isscalar(sz) && sz == 1)
                obj.isScalar = true;
            end
            
            obj.sz = sz(:)';
            obj.spVec = sparse(prod(sz),1);
        end
        
        %% Add/update values in tensor
        function update(o, subs, vals)
            assert(size(subs,1) == length(vals), 'Incorrect number of subscripts or values');
            o.spVec(o.getI(subs)) = vals;
        end
        
        %% Get values from tensor
        function vals = get(o, subs)
            vals = o.spVec(o.getI(subs));
        end
        
        %% Get subtensor
        function subTensor = getSubTensor(o, s)
            subTensor = mrfs.grm.sparsetensor( o.getSubTensorSize(s) );
                        
            % Get the indices and values
            [I,~,V] = find(o.spVec);

            if(~isempty(I))
                % Change to subscripts
                subs = o.getSubs(I);
                sel = subs(:,end) == s;

                % Update subtensor
                if(length(o.sz) == 1) % Is vector
                    % Scalar value subtensor
                    subTensor.update(subs(sel), V(sel));
                else
                    % Non-scalar value subtensor
                    subTensor.update(subs(sel, 1:(end-1)), V(sel));
                end
            end
        end
        
        function subSz = getSubTensorSize( o, s )
            if(s ~= 1 && o.isScalar)
                subSz = 0;
            elseif(length(o.sz) == 1)
                subSz = 1;
            else
                subSz = o.sz(1:(end-1));
            end
        end
        
        function setSubTensor(o, s, subTensor)
            % Set all in slice s to 0
            [~,V,subs] = o.getAll();
            if(~isempty(V))
                sel = (subs(:,end) == s);
                o.update(subs(sel,:), zeros(size(V(sel))));
            end
            
            % Add index s to subTensor subscripts
            [~,V,subs] = subTensor.getAll();
            if(subTensor.isScalar)
                subs = s*ones(size(V));
            else
                subs = [subs, s*ones(size(V))];
            end
            
            % Update tensor
            if(~isempty(V))
                o.update(subs, V);
            end
        end
        
        %% Indexing methods
        function [I, V, subs] = getAll(o)
            if(o.isScalar)
                I = 1; V = o.spVec(1);
            else
                [I,~,V] = find(o.spVec);
            end
            if(nargout >= 3)
                subs = o.getSubs(I);
            end
        end
        
        function [Iuniq, Vuniq, subsUniq] = getAllUniq(o)
            % Get unique indices values (adding together) and subs
            [I, V, subs] = o.getAll();
            subs = sort(subs,2);
            [subsUniq,ia,~] = unique(subs,'rows');
            Iuniq = I(ia);

            % Naive way to do combine V values since multiple values
            Vuniq = zeros(size(subsUniq,1),1);
            for ii = 1:size(subsUniq,1)
                Vuniq(ii) = sum(V(  all( bsxfun(@eq,subs,subsUniq(ii,:)) ,2)  ));
            end
        end
        
        
        function I = getI(o, subs)
            if(length(o.sz) == 1)
                I = subs;
            else
                subsCell = mat2cell(subs, size(subs,1), ones(size(subs,2),1));
                I = sub2ind(o.sz, subsCell{:});
            end
        end
        
        function bool = getBool(o,subs)
            I = o.getI(subs);
            bool = sparse(I,ones(size(I)),true(size(I)),prod(o.sz),1);
        end
        
        function subs = getSubs(o, I)
            subsCell = cell(1,length(o.sz));
            [ subsCell{:} ] = ind2sub(o.sz, I);
            subs = cell2mat(subsCell);
        end
        
        function sz = size(o)
            sz = o.sz;
        end
        
        %% VERY INEFFICENT, JUST FOR DEBUGGING
        function A = full(o)
            if(length(o.sz) == 1)
                A = zeros(o.sz,1);
            else
                A = zeros(o.sz);
            end
            [I,~,S] = find(o.spVec);
            subsT = o.getSubs(I)';
            for i = 1:length(S)
                indCell = num2cell(subsT(:,i));
                A(indCell{:}) = S(i);
            end
        end
    end
    
end

%Simple test code
%clear A3; A3 = mrfs.grm.sparsetensor(3*ones(3,1)); subs = [1 1 1; 2 2 2; 1 1 2; 3 1 2; 2 1 3]; vals = [111, 222, 112, 312, 213]'; A3.update(subs,vals); disp(A3.full()); B = A3.getSubTensor(2); fprintf('B\n'); disp(B.full()); A3.setSubTensor(1,B); A3.setSubTensor(2,B); A3.setSubTensor(3,B); fprintf('A with copies of B\n'); disp(A3.full());