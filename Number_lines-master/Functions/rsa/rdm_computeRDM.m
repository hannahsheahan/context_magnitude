function [rdmSet,whiteEEG] = rdm_computeRDM(results,disttype)
%% RDM_COMPUTEMAHALRDM
%
% computes distance RDMset
%
% results.betas  = electrode-time-condition matrix
% results.resids = electrode-time-trial matrix [for Mahalanobis]
% disttype = 'euclidean' / 'correlation' / 'mahalanobis'
%
% (c) Timo Flesch, 2016

% we want a time-condition-condition rdm-set:
n = size(results.betas,3);
rdmSet = zeros(size(results.betas,2),(n^2-n)/2);

switch disttype
    case 'correlation'
        whiteEEG = 0*permute(results.betas,[2,3,1]);
end

% iterate through all time points
for timePoint = 1:size(results.betas,2)
    
    respMat     = squeeze(mean(results.betas(:,timePoint,:),2));
    
    switch disttype
        case 'mahalanobis'
            residMat                = squeeze(mean(results.resids(:,timePoint,:),2));
            rdmSet(timePoint,:)     = pdist(respMat','mahalanobis',covdiag(residMat'));
        case 'correlation'
            residMat                = squeeze(mean(results.resids(:,timePoint,:),2));
            whiteEEG(timePoint,:,:) = respMat'*covdiag(residMat')^(-.5);
            rdmSet(timePoint,:)     = pdist(squeeze(whiteEEG(timePoint,:,:)),'correlation');
        case 'euclidean'
            rdmSet(timePoint,:)     = pdist(respMat');            
    end
        
end

end