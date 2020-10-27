function [erpdata, bindx, eindx] = PreprocERP(orgidx,inputfile,paths)
% [erpdata, bindx, eindx] = PreprocERP(orgidx,inputfile,paths)
% 
% Function to load subject data, concatenate and get indices of
% valid trials.
%
% Fabrice Luyckx, 17/10/2018

%% Initialise

erpdata     = [];

% Index for exluding trials
goodtrials  = [];
bindx       = []; % behavioural index
eindx       = []; % eeg index

%% Load data

fprintf('\nloading ...\n');

%% Load data

% Load EEG data
load(fullfile(paths.data.EEG,sprintf('%s_samples.mat',inputfile)));

% Load bad trials
load(fullfile(paths.data.EEG,sprintf('%s_rejectedTrials.mat',inputfile)));

erpdata     = eeg.data;
goodtrials  = 1-rejectedTrialz;

%% Indices for excluding trials

bindx   = orgidx & goodtrials'; % behavioural index
eindx   = orgidx(logical(goodtrials')) == 1; % eeg index

end

