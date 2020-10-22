function [model, thetaNode, thetaEdge] = fit( Xt, method, lam, nWorkers, R, R0 )
% Fits Truncated PGM using XMRF package in R
% formerly "xmrfwrapper"

    if(nargin < 4); nWorkers = 4; end % Default from XMRF package
    if(nargin < 5); R = full(max(Xt(:))); end % Default from XMRF package
    if(nargin < 6); R0 = 10; end % Default from XMRF package
    
    curr_path = fileparts(mfilename('fullpath')); % retreive the path the function file lives in
    
    %% Save files
    argsFile = tempname;
    dataFile = tempname;
    thetaFile = tempname;
    
    fid = fopen(argsFile,'w+');
    fprintf( fid, '%s,%e,%d,%d,%d\n', method, lam, nWorkers, R, R0 );
    fclose(fid);
    
    fid = fopen(dataFile,'w+');
    fprintf( fid, ['%d', repmat(',%d', 1, full(size(Xt,1))-1), '\n'], full(Xt(:)));
    fclose(fid);
    
    %% Call R script
    % [status,result] = system('hostname -f');
    % if(status == 0 && ~isempty(strfind(result,'cs.utexas.edu')))
    %     Rcmd = '/lusr/opt/R-3.2.2/bin/Rscript';
    % else
    %     Rcmd = 'Rscript';
    % end
    if ~system(sprintf('cd %s && Rscript fit.R %s %s %s', curr_path, argsFile, dataFile, thetaFile))
        fprintf('  R script for XRMF ran successfully.\n')
    end

    % cmd = sprintf('cd comparison && %s --vanilla xmrf.wrapper.R %s %s %s', ...
    %     Rcmd, argsFile, dataFile, thetaFile);
    % % fprintf('XMRFWrapper.m: Executing the following system command:\n%s\n\n', cmd);
    % for attempt = 1:10
    %     status = system(cmd);
    %     if(status == 0)
    %         fprintf('Correctly executed command on attempt = %d\n',attempt);
    %         break;
    %     end
    % end
    
    %% Extract theta
    theta = csvread(thetaFile);
    thetaNode = diag(theta);
    thetaEdge = theta - diag(thetaNode);
    thetaEdge = (thetaEdge + thetaEdge')/2; % Make symmetric
    
    model = struct('thetaEdge', thetaEdge, 'thetaNode', thetaNode);
    
end
    