function grm = fitstagewise( Xt, k, stageFuncs, opts )

%% Loop through each stage
[~,p] = size(Xt);
initType = 0;
opts.grm = mrfs.grm.GRMParameters( k, p, initType ); 
for kCur = 1:k
    %% Execute update on parameters before fitting
    nAdded = stageFuncs{kCur}( opts.grm ); % Modify grm parameters
    
    %% Use GRM fit with current GRM parameters
    if(nAdded > 0)
        fprintf('\n\n<<<< Starting stage %d/%d: Added %d new parameters (%d-tuples) >>>>\n\n', kCur, k, nAdded, kCur );
        opts.grm = mrfs.grm.fitgrm( Xt, k, opts );
    else
        fprintf('\n\n<<<< SKIPPING stage %d/%d because 0 new parameters (%d-tuples) >>>>\n\n', kCur, k, kCur );
    end
end

grm = opts.grm; % Extract final stage grm

end
