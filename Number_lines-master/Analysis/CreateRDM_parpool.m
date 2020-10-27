function CreateRDM_parpool(combo,numvec,data,outputstr,do,params,paths)
%function CreateRDM(s,orgidx,rsacond,inputfile,outputfile,do,params,paths)
%   Detailed explanation goes here

%% Prep stuff

sz = length(combo);

s = combo(1);

if sz == 1
    outputfile  = sprintf(outputstr,params.submat(s));
elseif sz == 2
    c = combo(2); % condition
    outputfile  = sprintf(outputstr,params.submat(s),c);
end

fprintf('\nCreating RDMs sub %d.\n',params.submat(s));

inputfile   = sprintf('Con_numbers_sub%03d',params.submat(s));

%% Index

idx     = data.sub == params.submat(s);

if sz == 1 % fillers/primary targets
    orgidx  = data.target(idx) == params.whichTarget & data.ctg(idx) ~= 1;

    tmpnum  = numvec(idx);
    tmpnum(orgidx & data.mod(idx) == 2) = tmpnum(orgidx & data.mod(idx) == 2)+20;
    tmpnum(orgidx & data.mod(idx) == 3) = tmpnum(orgidx & data.mod(idx) == 3)+50;
    tmpnum(~orgidx) = nan;

    condnumz = unique(tmpnum(~isnan(tmpnum)));
    j = 1;

    % Needs to recompute to numbers in single sequence (for binning)
    for i = 1:length(condnumz)
        tmpnum(tmpnum == condnumz(i)) = j;
        j = j+1;
    end

elseif sz == 2
    orgidx  = data.target(idx) == params.whichTarget & data.mod(idx) == c; % per condition
end

rsacond = tmpnum;

%% Load data

[erpdata,bindx,eindx] = PreprocERP(orgidx,inputfile,paths);

%% Resize data

% Reshape eeg data
eegdat      = erpdata(:,:,eindx);

% Vector with different conditions
condvec = rsacond(bindx)';

% Extra numbers
nsamp       = min(histcounts(condvec,'BinMethod','integers')); % Number of samples from each condition
conds       = unique(condvec(~isnan(condvec)));
ncond       = length(conds); % number of conditions
ntimepoints = size(eegdat,2);

%% Calculate distances (subsampled or not)

if do.subsamp

    % Initialise rdm
    allRDM = zeros((ncond^2-ncond)/2,ntimepoints);

    % RSA (subsampled nit times)

    fprintf('\nRSA subject %d.\n',params.submat(s));

    for n = 1:params.nit

        fprintf('%d ... ',n);

        condvec2    = nan(length(condvec),1);

        for m = 1:ncond
            I = find(condvec == conds(m));
            J = Shuffle(I);
            condvec2(J(1:nsamp)) = conds(m);
        end

        % Run regression with subset
        new_idx = ~isnan(condvec2);
        results = rdm_computeConditionCoeffs(eegdat(:,:,new_idx),condvec2(new_idx));

        % Compute RDM
        rdmSet      = rdm_computeRDM(results,params.disttype);
        allRDM      = allRDM + rdmSet';

    end

    % Average over iterations
    RDM = allRDM./params.nit;

    fprintf('\n')

else
    %% Run regression

    fprintf('\nCalculating betas subject %d.\n',params.submat(s));

    results = rdm_computeConditionCoeffs(eegdat,condvec);

    %% Compute RDM

    fprintf('\nCalculating RDM for subject %d, %s distance.\n',params.submat(s),params.disttype);

    rdmSet  = rdm_computeRDM(results,params.disttype);
    RDM     = rdmSet';

end

%% Save RDM

if do.saveRDM

    rdm.data        = RDM;
    rdm.conds       = conds;

    save(fullfile(paths.data.saveEEG,outputfile),'rdm');
    fprintf('\nRDMs saved for subject %d.\n',params.submat(s));
end

end
