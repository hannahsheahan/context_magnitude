%% Load path and data

% Set path
paths.main              = fullfile('/Volumes','Data_drive','Number_lines'); % your main folder
paths.toolbox.main      = fullfile('~','Documents','MATLAB');
eeglabversion           = 'eeglab14_1_2b'; % your eeglab version
supercomputer           = false;

paths.analysis          = fullfile(paths.main,'Analysis');
paths.data.main         = fullfile(paths.main,'Data');
paths.data.behav        = fullfile(paths.data.main,'Behavioural');
paths.data.EEG          = fullfile(paths.data.main,eegfolder); % baselined data
paths.data.saveEEG      = fullfile(paths.data.main,savefolder);
paths.functions.main    = fullfile(paths.main,'Functions');
paths.toolbox.eeglab    = fullfile(paths.toolbox.main, eeglabversion);
paths.toolbox.spm       = fullfile(paths.toolbox.main,'spm12');
paths.toolbox.fieldtrip = fullfile(paths.toolbox.main,'fieldtrip');
paths.figures.main      = fullfile(paths.main,'Figures');
paths.figures.current   = fullfile(paths.figures.main);

% Add paths
cd(paths.analysis);
addpath(paths.data.behav);
addpath(paths.data.EEG);
addpath(paths.toolbox.eeglab);
addpath(paths.toolbox.spm);
addpath(paths.toolbox.fieldtrip);
addpath(genpath(paths.functions.main));
addpath(paths.figures.current);

% Load behavioural data
load(fullfile(paths.data.behav,'Con_fulldata_behav.mat'));

% Load channel data
if exist(fullfile(paths.data.EEG,'chanlocs_file.mat')) > 0
    load(fullfile(paths.data.EEG,'chanlocs_file'));
end

% Load timepoints for data
if exist(fullfile(paths.data.EEG,'timepoints_file.mat')) > 0
    load(fullfile(paths.data.EEG,'timepoints_file'));
end

% Open eeglab to reach eggheadplot plugin
%eeglab;
%close

%% Variables

params.submat       = unique(data.sub)';
params.nsubj        = length(params.submat);
params.ttrials      = length(data.sub);
params.ntrials      = length(data.sub(data.sub == params.submat(1)));
params.nblocks      = length(unique(data.block));
params.btrials      = params.ntrials/params.nblocks; 
params.nconds       = 3;
params.numz         = unique(data.num);
params.nnumz        = length(params.numz);

for c = 1:params.nconds
    params.condnumz{c} = unique(data.num(data.mod == c & data.ctg > 0 & data.ctg < 2))';
end