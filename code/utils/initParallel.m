function initParallel( nWorkers )
%INITPARALLEL Attempt to start a parallel pool of size nWorkers if no pool already exists
%
% initParallel(nWorkers)
% Attempt to start parallel workers
if isempty(nWorkers)
    try
        [~,nStr] = system('grep --count "^processor" /proc/cpuinfo');
        nWorkers = str2double(nStr)-1;
        if(isnan(nWorkers)); nWorkers = 4; end
    catch
        nWorkers = 4;
    end
end

pool_handle = gcp('nocreate');
if isempty(pool_handle)
    
    myCluster = parcluster('local');
    myCluster.NumWorkers = nWorkers; 
    saveProfile(myCluster);    

    parpool(nWorkers)
    fprintf('\nCreated a pool with %d workers.\n', nWorkers)
else
    fprintf('\nUsing current pool with %d workers\n.', pool_handle.NumWorkers)
end