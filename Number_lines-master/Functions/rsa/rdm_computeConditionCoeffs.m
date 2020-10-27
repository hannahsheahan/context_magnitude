function [results, covmat] = rdm_computeConditionCoeffs(eegMat,behavVect)
%% RESULTS = RDM_COMPUTECONDITIONCOEFFS(EEGMAT,BEHAVVECT)
%
% regresses the eeg signal on the trial-by-condition design matrix for all electrodes and points in time
%
% eegMat = electrode-by-time-by-trial matrix of raw voltages
% behavVect = trial-vector (indicating the condition on each trial)
%
% (c) Timo Flesch, 2016 + Fabrice Luyckx 2019

%% set up design matrix

uniqueconds = unique(behavVect)';

% ignore nans
conditions = uniqueconds(~isnan(uniqueconds));

% dmat = zeros(size(behavVect,1),length(conditions));
% for c = 1:length(conditions)
%     dmat(behavVect==conditions(c),c) = 1;   
% end

dmat = dummyvar(behavVect);
 
results = struct(); 
results.betas   = zeros(size(eegMat,1),size(eegMat,2),length(conditions));
results.resids  = 0*eegMat;

% covmat  = struct();
% covmat.betas    = zeros(size(eegMat,1),size(eegMat,1),size(eegMat,2));
% covmat.resids   = covmat.betas;

%% for all electrodes and time points: run regression

for t = 1:size(eegMat,2)
    for el = 1:size(eegMat,1)
        
        [betas,~,resid]          = regress(zscore(squeeze(eegMat(el,t,:))),dmat);
        results.betas(el,t,:)    = betas; 
        results.resids(el,t,:)   = resid;
       
       %eeg_recon(:,el) = dmat*betas;       
    end       
    
%     covmat.betas(:,:,t)     = covdiag(eeg_recon);
%     covmat.resids(:,:,t)    = covdiag(squeeze(results.resids(:,t,:))');
%     
end

%% Regression with mtimesx and pseudo-inverse

% nbchan      = size(eegMat,1);
% ntimepoints = size(eegMat,2);
% ntrials     = size(eegMat,3);
% nregr       = size(dmat,2);
% 
% allLoops    = allcomb(1:nbchan,1:ntimepoints);
% pseudoInv   = nan(nregr,ntrials,nbchan,ntimepoints);
% 
% % Get pseudoinverse
% for r = 1:size(allLoops,1)
%     
%     curr_chan   = allLoops(r,1);
%     curr_time   = allLoops(r,2);
%     
%     pseudoInv(:,:,curr_chan,curr_time) = pinv(dmat);
% end
% 
% % Reshape EEG data
% ydat = eegMat;
% ydat = permute(ydat,[3,4,1,2]);
% 
% % Regression
% Betas = mtimesx(pseudoInv,ydat);
% Betas = squeeze(permute(Betas,[3,4,1,2])); % reshape betas
% 
% % Residuals
% yhat = 0*eegMat;
% for e = 1:nbchan   
%     yhat(e,:,:) = [dmat * squeeze(permute(Betas(e,:,:),[3,1,2]))]';
% end
% 
% resids = eegMat - yhat;
% 
% % Store results for output
% results.betas   = Betas;
% results.resids  = resids;

end