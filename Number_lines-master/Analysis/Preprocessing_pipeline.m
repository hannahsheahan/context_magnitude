%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Context_EEG: EEG preprocessing runner 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% LOAD DATA

clc
clear

% Set path
savefolder          = 'EEG_ERP'; % location of final preprocessed data

params.main         = fullfile('/Volumes','Data_drive','Minime_local'); % your main folder
params.toolbox.main = fullfile('~','Documents','MATLAB');
eeglabversion       = 'eeglab14_1_2b'; % your version of eeglab

params.analysis         = fullfile(params.main,'Analysis');
params.data.main        = fullfile(params.main,'Data');
params.data.behav       = fullfile(params.data.main,'Behavioural');
params.data.eeg         = fullfile(params.data.main,'EEG_raw');
params.data.saveEEG     = fullfile(params.data.main,'EEG_preproc');
params.data.finalEEG    = fullfile(params.data.main,savefolder);
params.toolbox.eeglab   = fullfile(params.toolbox.main, eeglabversion);
params.functions.main   = fullfile(params.main,'Functions');
params.functions.preproc = fullfile(params.functions.main,'EEG_preproc');

cd(params.analysis);
addpath(params.data.behav);
addpath(params.data.eeg);
addpath(params.data.saveEEG);
addpath(params.toolbox.eeglab);
addpath(params.functions.preproc);

%% Steps

do.parpooling       = true; % parallelise preprocessing step

do.import           = false;    % downsample/remove extra channels
do.filtering        = false;    % filter data
do.badchan          = false;     % remove bad channels and interpolate
do.averageref       = false;     % average referencing
do.longepoching     = false;     % extract long epochs and baseline correct
do.artefact         = false;     % then we need to visually inspect the data and reject epochs for each subject
do.rejectagain      = false;    % when you changed stuff in the previous steps, but don't want to reject trials manually again
do.ica              = false;    % use automatic ICA (can be quite long)
do.componentrej     = false;    % component rejection of ICA
do.extract          = false;    % make mat-file for data analysis

%% Settings

params.exclude_ppt      = 111;
params.submat           = setdiff(sort(101:139),params.exclude_ppt); % valid subject indices
params.trialz           = 1:2880; % which trials to use?
params.triggerz.all     = {100 1 2 11 12 21 22 23 24 31 32 33 41 42};
params.triggerz.long    = {11 21 23};

params.resamp           = 250; % resampling rate
params.epoch.long       = [-.25 1]; % time window of long epoch (check what your first trigger will be!) (in seconds)
params.baseline.long  	= [-250 0]; % baseline trial epoch (in ms)
params.filter.highpass 	= .05; % high-pass filter
params.filter.lowpass 	= 40; % low-pass filter

%Bad channels
params.badchan = {
    {'FT8','T8','T7','C1'},... % Subject 101
    {'T8','T7','AF8','PO7'},... % Subject 102 -> technical failure (1835 = random 23 trigger)
    {'T7','T8','FT8','F5','FT7','FC6'},... % Subject 103
    {},... % Subject 104
    {'FC2','PO7','O2','P6','P3','FCZ'},... % Subject 105
    {'F7','AF7','F5','FT7'},... % Subject 106
    {'T8','F7','FC6','F8','FC5'},... % Subject 107
    {'T7','T8','FP1','FPZ','FP2'},... % Subject 108
    {'AF7','AF8','F5','F8','FT8'},... % Subject 109
    {'T7','T8','FT8'},... % Subject 110
    {'FC6','FT8','T8','AF8','FC4'},... % Subject 112
    {'AF8','F6','F8','PO7'},... % Subject 113
    {},... % Subject 114
    {'FP1','AF8','F6','F8'},... % Subject 115
    {'CP5','TP8','PO4','PO7','PO8','CP2','P4','CPZ','CP3','P7','O2'},... % Subject 116
    {},... % Subject 117
    {'FT8'},... % Subject 118
    {'FC2'},... % Subject 119
    {'C6','FT7','O1'},... % Subject 120
    {'T8','F8','F6','PO3','PO7','AF8','CP6','C3','FP2'},... % Subject 121
    {'FP2','AF4','TP8','C2','AF7','P1','F1'},... % Subject 122
    {'T8','T7','FT7','CP1','AF7'},... % Subject 123
    {'T8','TP7'},... % Subject 124
    {'C4','FC2','FT7','FC6','CP5','T7','CP3','C3'},... % Subject 125
    {'PO8','FC4','FC6','T8'},... % Subject 126
    {'FP2','AF8','F6','FT8','CP4','CP2'},... % Subject 127
    {'T7','AF8','PO7','O1','F6','T8','FT8','P8','FP1','F8'},... % Subject 128
    {'T7','AF7','AF8','F3','FC4','F5','F6','FP2'},... % Subject 129
    {'POZ','CP6',},... % Subject 130
    {'CP2','FCZ','O1','PO7'},... % Subject 131
    {'F7','T7','AF8','FP1','FP2','F8'},... % Subject 132
    {'O1','POZ','T8'},... % Subject 133
    {'TP8'},... % Subject 134
    {'T8','F5','F7','F8','F6'},... % Subject 135
    {'T8','FT8','FC6','T7','F8'},... % Subject 136
    {'T7','P7','PO7','T8'},... % Subject 137
    {'T7','T8','TP7','AF8','FP1','FP2','AF8','TP8'},... % Subject 138
    {'T7','T8','O2'},... % Subject 139
    };

% Name prefix for all files
prefix = 'Con_numbers';

%% Parpooling

if do.parpooling
    numWorkers = length(params.submat);
    parpool(length(params.submat));
else
    numWorkers = 0;
end

try
    %% 1. DOWNSAMPLE IMPORTED DATA & REMOVE EXTRA CHANNELS
    
    parfor (s = 1:length(params.submat),numWorkers)
        
        inputfile   = sprintf('Acquisition_%d.set',params.submat(s));
        outputfile  = sprintf('%s_sub%d_downsampled.set',prefix,params.submat(s));
        
        fEEG_import(params,do,s,inputfile,outputfile);
    end
    
    %% 2. FILTER DATA
    
    parfor (s = 1:length(params.submat),numWorkers)
        
        inputfile   = sprintf('%s_sub%d_downsampled.set',prefix,params.submat(s));
        outputfile 	= sprintf('%s_sub%d_filtered.set',prefix,params.submat(s));
        
        fEEG_filter(params,do,s,inputfile,outputfile);
    end
    
    
    %% 3. BAD CHANNEL DETECTION AND INTERPOLATION
    
    parfor (s = 1:length(params.submat),numWorkers)
        
        inputfile 	= sprintf('%s_sub%d_filtered.set',prefix,params.submat(s));
        outputfile 	= sprintf('%s_sub%d_interpolated.set',prefix,params.submat(s));
        
        fEEG_badchan(params,do,s,inputfile,outputfile);
    end
    
    %% 4. AVERAGE REFERENCING
    
    parfor (s = 1:length(params.submat),numWorkers)
        
        inputfile   = sprintf('%s_sub%d_interpolated.set',prefix,params.submat(s));
        outputfile 	= sprintf('%s_sub%d_avref.set',prefix,params.submat(s));
        
        fEEG_averageref(params,do,s,inputfile,outputfile);
    end
    
    %% 5. LONG EPOCHING
    
    parfor (s = 1:length(params.submat),numWorkers)
        
        inputfile   = sprintf('%s_sub%d_avref.set',prefix,params.submat(s));
        outputfile 	= sprintf('%s_sub%d_longepoched.set',prefix,params.submat(s));
        
        fEEG_longepoch(params,do,s,inputfile,outputfile);
    end
    
    %% 6. REJECT ARTEFACTS (partially manually)
    
    if do.artefact
        
        % partially manual
        % click on bad trials + UPDATE MARKS before closing plot window!!
        % then reject the epochs USING EEG LAB - tools -> reject data epochs -> reject marked epochs
        
        subject = [];
        while isempty(subject)
            subject = input('Which subject? ');
            if ~any(subject == params.submat)
                error('Subject index not found.');
            end
        end
        
        inputfile   = sprintf('%s_sub%d_longepoched.set',prefix,subject);
        outputfile	= sprintf('%s_sub%d_rejtrials.set',prefix,subject);
        mfile       = sprintf('%s_sub%d_rejectedTrials.mat',prefix,subject);
        
        if ~isempty(subject)
            disp(' ');
            disp(['Rejecting trials of subject ' num2str(subject)])
            
            % restart eeglab (because of memory issues)
            close all
            clear ALLEEG EEG CURRENTSET ALLCOM
            [ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;
            
            % load dataset
            EEG = pop_loadset(inputfile, params.data.saveEEG);
            [ALLEEG, EEG] = eeg_store(ALLEEG, EEG, 1);
            
            % Open EEG plot
            pop_eegplot(EEG,1,1,0);
            pause
            EEG = ALLEEG(2);
            
            % Reject marked epochs
            marksUpdated = questdlg('Done with updating marks?');
            if strmatch(marksUpdated, 'Yes')
                
                % Save rejected trials in m-file
                clear rejectedTrialz
                rejectedTrialz = [EEG.reject.rejmanual(params.trialz)];
                save(fullfile(params.data.saveEEG,mfile),'rejectedTrialz');
                fprintf('\nRejected trials m-file subject %d saved.\n',subject);
                
                % Reject artefacts
                EEG = pop_rejepoch(EEG,rejectedTrialz);
                
                % Save file
                EEG.setname     = outputfile;
                EEG             = pop_saveset(EEG, outputfile, params.data.saveEEG);
                
                fprintf('\nRejected trials subject %d finished and saved.\n',subject);
                
            end
        end
    end
    
    %% 6a EXTRA: reject trials again
    % After you've done something in the previous steps, but you don't want to
    % manually start rejecting all the trials you've rejected before.
    
    for s = 1:length(params.submat)
        
        inputfile   = sprintf('%s_sub%d_longepoched.set',prefix,params.submat(s));
        outputfile	= sprintf('%s_sub%d_rejtrials.set',prefix,params.submat(s));
        mfile       = sprintf('%s_sub%d_rejectedTrials.mat',prefix,params.submat(s));
        
        fEEG_rejectagain(params,do,s,inputfile,outputfile,mfile);
    end
    
    %% 7. ICA
    
    tic
    
    parfor (s = 1:length(params.submat),numWorkers)
        
        inputfile   = sprintf('%s_sub%d_rejtrials.set',prefix,params.submat(s));
        outputfile	= sprintf('%s_sub%d_ica.set',prefix,params.submat(s));
        
        fEEG_ica(params,do,s,inputfile,outputfile);
    end
    
    disp('Finished all ICA.');
    elapsedtime = toc;
    fprintf('\nTime elapsed is %.2f minutes or %.2f hours.\n',elapsedtime/60,elapsedtime/60/60);
    
    %% 8. REMOVE ICA COMPONENTS
    
    if do.componentrej
        
        close all
        clear ALLEEG EEG CURRENTSET ALLCOM
        
        subject = [];
        while isempty(subject)
            subject = input('Which subject? ');
            if ~any(subject == params.submat)
                error('Subject index not found.');
            end
        end
        
        inputfile   = sprintf('%s_sub%d_ica.set',prefix,subject);
        outputfile	= sprintf('%s_sub%d_prunedica.set',prefix,subject);
        mfile       = sprintf('%s_sub%d_rejectedComponents.mat',prefix,subject);
        
        if ~isempty(subject)
            disp(' ');
            disp(['Removing components of subject ' num2str(subject)])
            
            % restart eeglab (because of memory issues)
            [ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;
            
            % load dataset
            EEG = pop_loadset(inputfile, params.data.saveEEG);
            [ALLEEG, EEG] = eeg_store(ALLEEG, EEG, 1);
            
            % Open EEG plot
            pop_selectcomps(EEG,1:35); % plot the component map for selection
            %pop_selectcomps(EEG,1:61); % plot the component map for selection
            %pop_eegplot(EEG, 1, 1, 0); % plot channel data timecourse % change scale to 50 and set full screen
            pop_eegplot(EEG, 0, 1, 0); % plot ICA compoments timecourse % change scale to 10 and set full screen
            pause
            EEG = ALLEEG(2);
            
            % Reject marked epochs
            compsUpdated = questdlg('Done with marking components?');
            if strmatch(compsUpdated, 'Yes')
                
                % Save removed components in m-file
                clear rejectedCompz
                rejectedCompz = find(EEG.reject.gcompreject == 1);
                save(fullfile(params.data.saveEEG,mfile),'rejectedCompz');
                fprintf('\nRemoved components m-file subject %d saved\n', subject);
                
                % Remove components
                EEG = pop_subcomp(EEG);
                
                % Save file
                EEG.setname     = outputfile;
                EEG             = pop_saveset(EEG, outputfile, params.data.saveEEG);
                
                fprintf('\nRemoving %d components from subject %d finished and saved.\n',length(rejectedCompz),subject);
                
            end
        end
    end
    
    %% 10. EXTRACT RELEVANT DATA IN MAT-FILE
    
    parfor (s = 1:length(params.submat),numWorkers)        

        inputfile   = sprintf('%s_sub%d_prunedica.set',prefix,params.submat(s));
        outputfile 	= sprintf('%s_sub%d_samples.mat',prefix,params.submat(s));
        
        fEEG_extractmat(params,do,s,inputfile,outputfile);        
    end

    %% End parpool session
    
    if do.parpooling
        delete(gcp());
    end
    
catch ME
    
    if do.parpooling
        delete(gcp());
    end
    
    rethrow(ME)
    fprintf('\nTry loop failed.\n');
    
end