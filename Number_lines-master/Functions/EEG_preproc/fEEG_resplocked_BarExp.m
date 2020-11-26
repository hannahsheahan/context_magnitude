function fEEG_resplocked_BarExp(params,do,s,inputfile,outputfile,mfile,newmfile)
%function fEEG_resplocked_BarExp(params,do,s,inputfile,outputfile)
% Function to extract response locked (sample) epochs. Also deletes 
% duplicate triggers.
% 
% Fabrice Luyckx, 10/10/2018

if do.resplocked
       
    fprintf('\nResponse locked epoching subject %d.\n', params.submat(s))
    
    % Restart eeglab (because of memory issues)
    close all
    clear ALLEEG EEG CURRENTSET ALLCOM
    [ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;
    
    % Load dataset
    EEG = pop_loadset(inputfile, params.data.saveEEG);
    [ALLEEG, EEG] = eeg_store(ALLEEG, EEG, 1);
    
    % Load rejected trials file
    load(fullfile(params.data.saveEEG,mfile));    
    rejectedTrialz = rejectedTrialz(params.trialz); % omit extra trials
        
    % Delete duplicate triggers
    a = [2,10,11,12,13,14,15,16,17,18,70];
    b = [2,10,11,12,13,14,15,16,17,18,71];
    c = [2,10,11,12,13,14,15,16,17,18];
    
    leftInd     = strfind([EEG.event.type],a);
    rightInd    = strfind([EEG.event.type],b);
    lateInd     = strfind([EEG.event.type],c);
    
    leftInd2    = repmat(leftInd,length(a),1) + repmat(0:length(a)-1,size(leftInd,2),1)';
    leftInd2    = leftInd2(:);    
    rightInd2    = repmat(rightInd,length(a),1) + repmat(0:length(a)-1,size(rightInd,2),1)';
    rightInd2    = rightInd2(:);
    if ~isempty(lateInd)
        lateInd2    = repmat(lateInd,length(a),1) + repmat(0:length(a)-1,size(lateInd,2),1)';
        lateInd2    = lateInd2(:);
    end
    
    keepTrigz = 0*[EEG.event.type];
    keepTrigz(leftInd2)     = 1;
    keepTrigz(rightInd2)    = 1;
    if ~isempty(lateInd) 
        keepTrigz(lateInd2) = 1;
    end

    % Check for triggers that accidentally occur before fixation (=2)
    allTrigz    = [EEG.event.type];
    allEpochs   = [EEG.event.epoch];
    for e = unique(allEpochs)
        trigz = allTrigz(allEpochs == e);
        if trigz(1) ~= 2
            idx = find(allEpochs == e);
            keepTrigz(idx(1)) = 0;
        end
    end
    
    % Remove excess triggers
    EEG.event(~keepTrigz)   = []; 
    
    % Find trials where there was no response included
    nRespEpochs = length(find([EEG.event.type]==70))+length(find([EEG.event.type]==71));
    orignTrials = EEG.trials;
    noResp      = [];
    
    if nRespEpochs < orignTrials
        allTrigz    = [EEG.event.type];
        allEpochs   = [EEG.event.epoch];
        for e = unique(allEpochs)
            trigz = allTrigz(allEpochs == e);
            if ~any(trigz == 70) && ~any(trigz == 71)
                noResp = [noResp,e];
            end
        end
        noResp      = sort(noResp);
    end
    
    % Epoching
    fprintf('\n%d original epochs\n',EEG.trials);
    newname             = outputfile;
    [EEG , keepEpochs]  = pop_epoch(EEG, params.triggerz.resplock, params.epoch.resplock, 'newname', newname, 'epochinfo', 'yes');
    [ALLEEG, EEG]       = eeg_store(ALLEEG, EEG, 1);
    
    % Get real indices of left out trials    
    badTrials = setdiff(1:nRespEpochs,keepEpochs); % find bad epochs
    
    if length(keepEpochs) < orignTrials
        % Adjust indices for epochs that had no responses
        for j = 1:length(noResp)
            badTrials(badTrials > noResp(j)) = badTrials(badTrials > noResp(j))+1;
        end   
        badTrials = sort([badTrials,noResp]);   
    end
    
    % Find these bad trials in original rejected trials
    idx                     = find(rejectedTrialz == 0);
    newIdx                  = idx(badTrials);
    rejectedTrialz(newIdx)  = 1;
    
    % Print out some stats
    disp(' ');
    disp(['Trials idx out of bound: ' num2str(setdiff(1:nRespEpochs,keepEpochs))]);
    if ~isempty(noResp), disp(['Trials without responses: ' num2str(noResp)]);
    else disp(['Trials without responses: ' num2str(0)]); end
    disp(['New n rejected trials: ' num2str(sum(rejectedTrialz))]);
    
    % Save new rejected trials in mat file
    save(fullfile(params.data.finalEEG,newmfile),'rejectedTrialz');
    
    % Save the dataset
    EEG.setname     = outputfile;
    EEG             = pop_saveset(EEG, outputfile, params.data.saveEEG);
    
    fprintf('\nResponse locked epoching subject %d finished.\n', params.submat(s));
    
end

end
