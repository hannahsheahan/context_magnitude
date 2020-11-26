%%%%%%%%
%% RSA
%%%%%%%%

%% LOAD DATA

clc
clear

% Paths
eegfolder           = 'EEG_ERP';
savefolder          = 'EEG_RDM';

% Load stuff
Load_paths_and_data; % load data and path
Config_plot; % load plot variables

% Logicals
do.parpooling  	= false; % parpool?

do.rdm          = true; % convert to RDM
do.subsamp      = true; % subsampled data? 
do.saveRDM      = true;

do.rsa          = true;

do.smooth       = true; % smooth results?
do.plotting     = true; % plot?
do.signif       = true; % test significance?
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

% Adjacent primary targets
data.comppair   = 0*[data.sub data.sub];

for t = 1:params.ttrials
    if data.ctg(t) == 1
        data.comppair(t,1)  = 0;
        data.comppair(t,2)  = 0;
        last_prim           = data.num(t);
    elseif data.ctg(t) > 1 && data.ctg(t) < 2
        data.comppair(t,1)  = last_prim;
        data.comppair(t,2)  = data.num(t);
        last_prim           = data.comppair(t,2);
    else
        data.comppair(t,1)  = last_prim;
        data.comppair(t,2)  = data.num(t);
    end
end

% Number of timepoints
ntimepoints         = length(timepoints);

%% Some settings

params.whichTarget  = 1; % for creating RDMs (0 = filler, 1 = primary, 2 = secondary)
targetname          = {'fill','prim','sec'};
params.cvname       = 'allcond';
params.disttype     = 'correlation';
params.wdwsz        = round(60/4); % window size to smooth (ms/samprate)

params.nit          = 100; % number of iterations for subsampling

%% Create cross-validation RDM

if do.rdm
    
    try
        tic % get current time
        
        % Parpooling
        numWorkers = params.nsubj;
        if do.parpooling
            c = parcluster();
            c.NumWorkers = numWorkers;
            parpool(c,numWorkers);
        else
            numWorkers = 0;
        end
        
        % Vector that will define numbers per conditions
        numvec  = data.comppair(:,2); % 1 for previous primary
        
        % All combinations (subjects)
        allcombinations = combvec(1:params.nsubj)';
        
        % File names
        outputstr  = sprintf('Con_numbers_sub%%03d_RDM_%s_%s_%s_block',targetname{params.whichTarget+1},params.cvname,params.disttype);
        
        if do.subsamp
            outputstr = sprintf('%s_subsamp%d',outputstr,params.nit);
        end
        
        % Create RDMs
        parfor (combo = 1:length(allcombinations),numWorkers) %for s = 1:params.nsubj
            CreateRDM_parpool(allcombinations(combo,:),numvec,data,outputstr,do,params,paths);
        end
        
        % Wrap up
        disp('Finished creating RDMs.');
        elapsedtime = toc;
        fprintf('\nTime elapsed is %.2f minutes or %.2f hours.\n',elapsedtime/60,elapsedtime/60/60);
        
        % End parpool session
        if do.parpooling
            delete(gcp);
        end
        
    catch ME
        
        if do.parpooling
            delete(gcp);
        end
        
        rethrow(ME)
        
        elapsedtime = toc;
        fprintf('\nTime elapsed is %.2f minutes or %.2f hours.\n',elapsedtime/60,elapsedtime/60/60);
        
        return
    end
end

%% RSA

if do.rsa
    
    clear mods actmod allmod rsa_results
    
    for s = 1:params.nsubj % subject correlations
        
        fprintf('\nCorrelating model - EEG RDM subject %d.\n',params.submat(s));
        
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
        
        % Model RDMs
        if params.whichTarget == 1 % primary
            mods = ModelRDM_prim(s,data.condcol(:,:,s),paths);
        end
        
        models  = fieldnames(mods);
        for m = 1:length(models)
            allmod      = mods.(models{m});
            actmod(:,m) = nanzscore(allmod);
        end
        
        % Regression
        for t = 1:ntimepoints
            betas = regress(nanzscore(rdm.data(:,t)),[actmod(:,1)*0+1 actmod]);
            rsa_results(s,t,:) = betas(2:end);
        end
        
    end
    
    %% Plot correlation time series
    
    if do.plotting
        
        plotmat = [1:3]; % regressors to plot
        
        % Settings
        xlims       = [min(timepoints) 1000];
        ylims       = [-.05 .18];
        masstest    = 'cluster';
        p_crit      = 0.05;
        
        if params.whichTarget == 1
            legendnamez = {'Visual','Condition','Magnitude'};
            %legendnamez = {'Visual','Condition','Magnitude','Color','RT'};
        end
        
        for g = 1:length(plotmat)
            
            figS = figure; hold on;
            
            plot([500 500],ylims,'k-','LineWidth',1.5); % stimulus offset
            
            [steh,sigp] = fEEG_steplot(rsa_results(:,:,plotmat(g)),timepoints,ylims,do.signif,masstest,colz(plotmat(g),:),p_crit,'bottom');
            
            sigp(1).YData = sigp(1).YData*0-0.03;
            
            % Additional settings and labels
            xlim(xlims);
            ax = gca;
            set(ax,'FontSize',16,'LineWidth',1.5);
            xlabel('Time from sample onset (ms)');
            ylabel('\beta coefficients');
            
            ax.XLabel.FontSize = axfntsz;
            ax.YLabel.FontSize = axfntsz;
            
            if do.save_plot
                save2eps(figS,paths.figures.current,sprintf('RSA_%s_%s',targetname{params.whichTarget+1},legendnamez{plotmat(g)}),[.1 .1 500 300]);
            end
        end
        
    end
    
end
