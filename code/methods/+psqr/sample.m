function XtSample = sample( PsiExt, nSamples, nGibbs, nInner, nBatches )
    XtSample = cell(nBatches,1);
    nSamplesVec = floor(nSamples/nBatches)*ones(1,nBatches);
    oneMore = nSamples - sum(nSamplesVec);
    nSamplesVec(1:oneMore) = nSamplesVec(1:oneMore) + 1;
    assert(sum(nSamplesVec) == nSamples, 'Batches not split properly');
    assert(all(nSamplesVec == floor(nSamples/nBatches) | nSamplesVec == floor(nSamples/nBatches)+1), 'Batches not split properly');

    parfor bi = 1:nBatches;
        nSamplesCur = nSamplesVec(bi);
        XtSample{bi} = mrfs.grm.univariate.Poisson.sampleSQR_Gibbs( PsiExt, nSamplesCur, nGibbs, nInner );
    end
    XtSample = cell2mat(XtSample);
end