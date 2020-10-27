%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Context_main: runner
%
% All functions written by Fabrice Luyckx and Clemens Teupe
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

execute = questdlg('Clear all and start experiment?');
% execute = 'Yes';

if strmatch(execute, 'Yes')
    
    clc
    clear 
    close all
    
    % Using EEG?
    EEGquest   = questdlg('Using EEG system?');
    switch EEGquest
        case 'Yes'
            inEEG = true;
        case 'No'
            inEEG = false;
        case 'Cancel'
            return;
    end
    
    %% Set path
    datapath = cd;
    datafolder = fullfile(datapath,'/Con_data/');
    
    addpath(genpath(fullfile(datapath,'Con_functions/'))); % add function folder with subfolder
    addpath(datafolder); % add datafolder
    
%% Create participant structure

    argindlg = inputdlg({'Participant number   ','Gender (M/F)','Age','Hand (L/R)'},'',1,{'000','','','R'});
    if isempty(argindlg)
        return;
    else
        participant                 = struct;
        participant.name            = upper(argindlg{1});
        participant.gender          = argindlg{2};
        participant.age             = argindlg{3};
        participant.handedness      = argindlg{4};
    end
    
    %% Get OS
    os = computer;
    
    %% Set keys
    KbName('UnifyKeyNames');
    
    %% What I need to know
    ppnr        = str2num(participant.name);    % participant number
    
    nTri        = 2880;                         % number of trials (bTri
                                                % must be divisible by
                                                % twelve)
    nBlo        = 24;                           % number of blocks (must be
                                                % divisible by three)

    low         = [25 35];                      % range of low numbers
    hig         = [30 40];                      % range of high numbers
    
    bPriTar     = 30;                           % number of primary targets
                                                % per block (must be equal
                                                % to 3*(nTri/nBlo)/12)
    pSecTar     = 0.12;                         % probability with which to
                                                % replace grey filler
                                                % numbers with secondary
                                                % target numbers
    
    randomise   = 1;                            
    if randomise == 1 % variable to activate randomisation
        % Seed random number generator
        rng('shuffle');
    end   
    aborted = 0;
    
    colAss = [1 2 3]; % colour assignment
    if randomise == 1
        colAss = Shuffle(colAss); % first number indicates colour of low
                                  % range blocks, second number indicates
                                  % colour of high range blocks, and third
                                  % number indicates colour of combined
                                  % range blocks (1 = blue, 2 = orange, 3 =
                                  % purple)
    end
    
    expstart = tic;
    
     %% Demo
%     practice = 2;
%     demoTrialz = 30;
%     demoBlockz = 3;
%     demoBPriTar = 3;
%     
%     disp ('Demo initiated.')
%     [~, ~, ~, aborted] = Exp_trial(inEEG,randomise, practice, ppnr, demoTrialz,demoBlockz,low,hig,demoBPriTar,pSecTar,colAss);
%     
    %% Practice trials
    
%     if aborted == 0
%         practice        = 1;
%         practTrialz     = 144;
%         practBlockz     = 3;
%         practBPriTar    = 12;
%         
%         disp('Practice initiated.')
%         [~, ~, ~, aborted] = Exp_trial(inEEG,randomise,practice,ppnr,practTrialz,practBlockz,low,hig,practBPriTar,pSecTar,colAss);
%     end   
    
    %% Run experiment
     if aborted == 0
        practice = 0;
        disp('Experiment initiated.')
        [data, time, triggers, aborted] = Exp_trial(inEEG,randomise, practice, ppnr, nTri, nBlo, low, hig, bPriTar, pSecTar, colAss);
     end
    
    if aborted == 0
        %% Save data
        participant.filename = sprintf('Con_ppt_%s_%3s.mat',participant.name,datestr(now,'yyyymmddHHMMSS'));
        datafile = fullfile(datapath,'/Con_data/',participant.filename);
        save(datafile,'participant', 'colAss', 'data','time','triggers');
        
        %% Send e-mail with data
        %Exp_send_email;
    end
    
    telapsed = toc(expstart);
    fprintf('\nExperiment took %.2f minutes\n',telapsed/60);
    
end

%% EXPLANATION DIFFERENT FUNCTIONS
%
% Exp_initialise = define all variables for data and stim(ulus) structures,
% very handy to check when something went wrong in the randomisation or
% trial files.
%
% Exp_setup = give all variables the real values and option to randomise
% everything. Use this to check whether randomisation does what it's
% supposed to do and check underlying distributions.
%
% Exp_trial = run the real experiment without all the fuzz around it
%
% Exp_main = file that executes the whole experiment. Here you define the
% actual values of the number of trials etc. in your experiment.