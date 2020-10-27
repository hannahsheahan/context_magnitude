%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Cross-validated time windows
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% LOAD DATA

clc
clear

% Logicals
eegfolder           = 'EEG_RDM';
savefolder          = 'EEG_RDM';

% Load stuff
Load_paths_and_data; % load data and path
Config_plot; % load plot variables

% Logicals
do.parpooling  	= false; % parpool?

do.subsamp      = true; % subsample data?

do.timewdw      = true; % find averaging time window
do.save_timez   = true; % save time windows
do.avRDM        = true; % average RDMs
do.save_rdm     = true; % save averaged rdms

do.smooth       = true; % smooth results?
do.plotting     = true; % plot?
do.signif       = false; % test significance?
do.save_plot    = false; % save plot?

%% Extra variables

data.target = round(data.ctg);

data.dist2targ      = 0.*data.target;
data.dist2targ(1)   = data.num(1);
primtg              = data.num(1);

for t = 2:params.ttrials
    
    if data.block(t) ~= data.block(t-1) % if first trial
        primtg = 0;
        data.dist2targ(t) = 0;
    else
        data.dist2targ(t) = data.num(t) - primtg;
    end
    
    if data.target(t) == 1
        primtg = data.num(t);
    end
end

data.dist2targ(data.ctg == 1) = 0;

% Number of timepoints
ntimepoints = length(timepoints);

%% Some settings

params.whichTarget  = 1; % for creating RDMs
targetname          = {'fill','prim','sec'};
params.cvname       = 'allcond';
params.disttype     = 'correlation';

params.nit          = 100; % number of iterations for subsampling

%% Load all RDMs (sub x cond x time)

for s = 1:params.nsubj % subject correlations
    
    fprintf('\nLoading EEG RDM subject %d.\n',params.submat(s));
    
    inputfile  = sprintf('Con_numbers_sub%03d_RDM_%s_%s_%s',params.submat(s),targetname{params.whichTarget+1},params.cvname,params.disttype);
    
    if do.subsamp
        inputfile = sprintf('%s_subsamp%d',inputfile,params.nit);
    end
    
    % Load RDM
    load(fullfile(paths.data.saveEEG,inputfile));
    
    % Smooth data
    if do.smooth
        wdwsz    = 60/4; % size convolution kernel (ms/downsampling rate)
        rdm.data = smoothRDM(rdm.data,wdwsz);
    end
    
    if s == 1
        nconds  = length(rdm.conds);
        allrdm  = zeros(params.nsubj,(nconds^2-nconds)/2,ntimepoints);
    end
    
    for t = 1:ntimepoints
        allrdm(s,:,t) = rdm.data(:,t);
    end
end

%% Find time window for averaging (leave-one-out)

if do.timewdw
    
    % Leave-one-out correlations
    timez.magn  = nan(params.nsubj,2);
    timez.cond  = nan(params.nsubj,2);
    timez.vis   = nan(params.nsubj,2);
    
    for s = 1:params.nsubj
        
        fprintf('\nSub %d\n',s);
        
        submat2 = setdiff(1:params.nsubj,s);
        
        for z = 1:params.nsubj-1 % leave-one-out

            % Data of single subject still included
            subdata = squeeze(allrdm(submat2(z),:,:));
            
            % Model RDMs
            mods = ModelRDM_3cond_prim(z,data.condcol(:,:,submat2(z)),paths);
            
            models  = fieldnames(mods);
            for m = 1:length(models)
                allmod      = mods.(models{m});
                actmod(:,m) = nanzscore(allmod);
            end
            
            % Regression (Pearson)
            for t = 1:ntimepoints
                betas = regress(zscore(subdata(:,t)),[actmod(:,1)*0+1 actmod]);
                rsa_results(z,t,:) = betas(2:end);
            end
        end
        
        % Get significant cluster
        nit         = 5000;
        p_thresh    = 0.005;
        
        p(1,:)  = ClusterCorrection2(rsa_results(:,:,3),nit,p_thresh); % magn
        p(2,:)  = ClusterCorrection2(rsa_results(:,:,2),nit,p_thresh); % cond
        p(3,:)  = ClusterCorrection2(rsa_results(:,:,1),nit,p_thresh); % visual
        p(p > p_thresh) = nan;
        
        % Select cluster
        tmpM    = find(p(1,:) == min(p(1,:))); % all timepoints in strongest cluster
        tmpC    = find(p(2,:) == min(p(2,:))); % all timepoints in strongest cluster
        tmpV    = setdiff(find(p(3,:) == min(p(3,:))),[tmpC,tmpM]); % exclude later timepoints
        
        % Save time windows
        timez.magn(s,:) = [tmpM(1) tmpM(end)];
        timez.cond(s,:) = [tmpC(1) tmpC(end)];
        timez.vis(s,:)  = [tmpV(1) tmpV(end)];
        
    end
    
    if do.save_timez
        save(fullfile(paths.data.saveEEG,'Crossvalidated_timewindows_asIndex'),'timez');
        fprintf('\nTime windows saved.\n');
    end
else
    load(fullfile(paths.data.saveEEG,'Crossvalidated_timewindows_asIndex'));
end

%% Save RDMs

if do.avRDM
    
    magndat         = zeros(nconds,nconds,params.nsubj);
    conddat         = magndat;
    visdat          = magndat;
    
    for s = 1:params.nsubj
        
        subdat  = squeeze(allrdm(s,:,:));
        
        % Magnitude
        magndat(:,:,s)      = squareform(mean(subdat(:,timez.magn(s,1):timez.magn(s,2)),2));
        
        % Condition
        conddat(:,:,s)      = squareform(mean(subdat(:,timez.cond(s,1):timez.cond(s,2)),2));
        
        % Visual
        visdat(:,:,s)       = squareform(mean(subdat(:,timez.vis(s,1):timez.vis(s,2)),2));
    end
    
    % Save files
    if do.save_rdm
        save(fullfile(paths.data.saveEEG,sprintf('avRDM_magn_%s',targetname{params.whichTarget+1})),'magndat');
        save(fullfile(paths.data.saveEEG,sprintf('avRDM_cond_%s',targetname{params.whichTarget+1})),'conddat');
        save(fullfile(paths.data.saveEEG,sprintf('avRDM_vis_%s',targetname{params.whichTarget+1})),'visdat');
        fprintf('\nData files saved.\n');
    end
    
end
