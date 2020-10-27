function fEEG_wavelet(sbt,params,paths)
%function fEEG_wavelet(sbt,params,paths)
%
%   Make wavelet transformation of EEG data.
%   Saves per sample separately.
%
%   Fabrice Luyckx, 12/1/2017

try
%% Get properties
s   	= sbt(1);
sess    = sbt(2);

%% Display

fprintf('\nWavelet transformation subject %d, session %d.\n',params.submat(s),sess);

%% Load data

% Load EEG data
load(fullfile(paths.data_EEG,['Subject' num2str(params.submat(s)) '_Session' num2str(sess) '_BarExp_EEG_samples.mat']));

% Load bad trials
load(fullfile(paths.data_EEG,['Subject' num2str(params.submat(s)) '_Session' num2str(sess) '_BarExp_rejectedTrials.mat']));

%% Define variables

% Variables
goodtrials  = 1-rejectedTrialz(1:params.ntrials);
ntimepoints = length(eeg.timepoints); % number of time points in epoch
rtrials     = length(find(goodtrials == 1)); % number of preserved trials

% Reshape
erpdata     = reshape(eeg.data,[eeg.nbchan,ntimepoints,params.nsamp,rtrials]);

%% Wavelet

dt              = (eeg.timepoints(2)-eeg.timepoints(1))/1000; % time between two datapoints (sampling rate) in sec
time            = 1:ntimepoints; % number of datapoints per trial to look at

% Loop through each electrode (save separate for each sample)
for f = 1:params.nsamp
    
    % Preallocate
    waves           = zeros(eeg.nbchan,length(params.period_ind),size(erpdata,4),length(time));
    
    for e = 1:eeg.nbchan
        [wave,period,scale,coi] = waveletmat(squeeze(erpdata(e,time,f,:))',dt,-1,-1,-1,-1,-1,params.period_ind);
        waves(e,:,:,:)     	= wave;
    end
    
    % Save data
    savename = ['BarExp_wavelet_sub' num2str(params.submat(s)) '_sess' num2str(sess) '_samp' num2str(f) '.mat'];
    save(fullfile(paths.data_wav,savename),'waves','period','scale','coi','-v7.3');
    
end

fprintf('\nWavelet transformation subject %d, session %d finished.\n',params.submat(s),sess);

catch ME
    disp(ME);
    return
end
end

