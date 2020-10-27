%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Context_numbers: regressions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Runs several regression models on the behavioural data with either
% accuracy or reaction time as dependent variable
% Compares relative vs absolute models and expands the regression models 
% distance to the middle of the context range as an extra regressors

% Requires VBA toolbox for model comparison: https://nl.mathworks.com/matlabcentral/fileexchange/56720-mbb-team-vba-toolbox

%% LOAD DATA

clc
clear

eegfolder       = []; % unused here
savefolder      = []; % unused here

% Load stuff
Load_paths_and_data; % load data and path
Config_plot; % load plot variables

% Logicals
do.save_fit     = false; % save fitted betas and LL
do.modcomp      = true; % compare models
do.signif       = true; % do significance testing?

do.plotting     = true; % plot?
do.save_plot    = false; % save plot?

%% Extra variables

% Target
data.target = round(data.ctg);
targs       = (data.ctg>0 & data.ctg<2);

% Find outlying RT trials
boundval    = 2.5;
outRT       = zeros(params.nsubj,2);
inv_rt      = 1./data.RT;
for s = 1:params.nsubj
    mRT         = mean(inv_rt(inv_rt > 0 & data.sub == params.submat(s)));
    sdRT        = std(inv_rt(inv_rt > 0 & data.sub == params.submat(s)));
    outRT(s,1)  = mRT-boundval*sdRT; % lower bound outliers RT
    outRT(s,2)  = mRT+boundval*sdRT; % upper bound outliers RT
end

%% Regression accuracy

normnamez   = {'Relative','Absolute'};
dvnamez     = {'Accuracy','RT'};
regnamez    = {'Absdiff','Absdiff + dist2mid'};

for r = 1:2 % 'abs diff' or 'abs diff + dist2mid'
    
    clear bb_m bb_a dev
    
    for n = 1:2 % relative or absolute
        
        %Normalize numbers per condition  (relative)
        if n == 1
            num_cond = 0.*data.num;
            for c = 1:params.nconds
                
                cidx = data.mod == c & data.target == 1;
                
                %Condition bounds
                cmin = min(data.num(cidx));
                cmax = max(data.num(cidx));
                
                %Normalise numbers within condition
                num_cond(cidx) = (data.num(cidx)-cmin)./(cmax-cmin);
            end
            
            midval_local    = [.5 .5 .5];
            midval_global   = [.5 .5 .5];
            
        elseif n == 2 % Normalize within experiment (absolute)
            num_cond        = (data.num - min(data.num))./(max(data.num)-min(data.num));
            
            midval_local    = [1/3 2/3 .5];
            midval_global   = [.5 .5 .5];
        end
        
        % Middle value within range
        data.meanctx                = data.mod.*0;
        data.allctx                 = data.mod.*0;
        
        for c = 1:3
            cidx = data.mod == c & data.target == 1;
            data.meanctx(cidx)	= midval_local(c);
            data.allctx(cidx)   = midval_global(c);
        end
        
        % Difference to mean within range
        data.meandiff                   = (num_cond-data.meanctx).*targs; % current
        t                               = find(data.target==1);
        data.lastmeandiff               = data.meandiff.*0;
        data.lastmeandiff(t(2:end))     = data.meandiff(t(1:end-1)); % previous
        
        % Difference to mean over all ranges
        data.alldiff                    = (num_cond-data.allctx).*targs; % current
        t                               = find(data.target==1);
        data.lastalldiff                = data.alldiff.*0;
        data.lastalldiff(t(2:end))      = data.alldiff(t(1:end-1)); % previous
        
        % Difference primary targets
        data.comppair   = 0*[data.sub data.sub];
        
        for t = 1:params.ttrials
            if data.ctg(t) == 1
                data.comppair(t,1)  = 0;
                data.comppair(t,2)  = 0;
                last_prim           = num_cond(t);
            elseif data.ctg(t) > 1 && data.ctg(t) < 2
                data.comppair(t,1)  = last_prim;
                data.comppair(t,2)  = num_cond(t);
                last_prim           = data.comppair(t,2);
            end
        end
        
        for d = 1:2 % dependent variable: accuracy or RT
            fprintf('Running %s / %s / %s\n',normnamez{n},dvnamez{d},regnamez{r});
            
            for s = 1:params.nsubj
                
                idx = data.sub == params.submat(s) & (data.ctg > 1 & data.ctg < 2) & data.RT > 0 & inv_rt >= outRT(s,1) & inv_rt <= outRT(s,2);
                
                mean_diffdiff   = abs(data.meandiff(idx))-abs(data.lastmeandiff(idx));
                all_diffdiff    = abs(data.alldiff(idx))-abs(data.lastalldiff(idx));
                
                % Regressors
                if r == 1
                    XX_loc  = abs(diff(data.comppair(idx,:),[],2));
                    XX_glob = abs(diff(data.comppair(idx,:),[],2));
                elseif r == 2
                    XX_loc  = [abs(diff(data.comppair(idx,:),[],2)) mean_diffdiff];
                    XX_glob = [abs(diff(data.comppair(idx,:),[],2)) all_diffdiff];
                end
                
                % Regression
                if d == 1 % accuracy
                    YY  = data.cor(idx);
                    [bb_m(s,:,n,d), dev(s,1,n,d)] = glmfit(zscore(XX_loc),YY,'binomial','link','probit');
                    [bb_a(s,:,n,d), dev(s,2,n,d)] = glmfit(zscore(XX_glob),YY,'binomial','link','probit');
                elseif d == 2 % reaction time
                    YY  = inv_rt(idx);
                    [bb_m(s,:,n,d), dev(s,1,n,d)] = glmfit(zscore(XX_loc),YY);
                    [bb_a(s,:,n,d), dev(s,2,n,d)] = glmfit(zscore(XX_glob),YY);
                end
                
            end
        end        
    end
    
    % Save results per regression
    regresults{r,1} = bb_m; % sub x beta x norm x dv
    regresults{r,2} = bb_a;
    regresults{r,3} = dev; % sub x context x norm x dv
end

%% Test models

remsub = 4; % remove subjects

options.verbose     = 0; % no output printed
options.DisplayWin  = 0; % no figure window

% Testing local v global (only makes sense for absolute case!)
for r = 1:2
    
    getbetas_m  = regresults{r,1};
    getbetas_a  = regresults{r,2};
    getdev      = regresults{r,3};
    
    for n = 1:2
        for d = 1:2
            
            testdat = squeeze(getdev(:,:,n,d));
            betas   = cat(3,getbetas_m(:,:,n,d),getbetas_a(:,:,n,d));
            
            if d == 1 % remove for accuracy
                testdat(4,:) = [];
                betas(4,:,:) = [];   
            end
            
            % T-test betas
            [h,p,~,stats] = ttest(squeeze(betas(:,2:end,1)));
            
            % BMS
            [posterior,out] = VBA_groupBMC(-testdat',options);
            
            fprintf('\nTesting local vs global: %s ~ %s, %s',dvnamez{d},regnamez{r},normnamez{n});
            fprintf('\nModel frequencies: %.02f %.02f',out.Ef);
            fprintf('\nProtected exceedance probabilities: %.02f %.02f', out.pxp);
            p
            
        end
    end
end

% Testing relative v absolute
for r = 1
    
    getbetas_m  = regresults{r,1};
    getbetas_a  = regresults{r,2};
    getdev      = regresults{r,3};
    
    for d = 1:2
        
        testdat = [squeeze(getdev(:,1,1,d)),squeeze(getdev(:,1,2,d))];
        betas   = cat(3,getbetas_m(:,:,1,d),getbetas_a(:,:,2,d));
        
        if d == 1
            testdat(4,:) = [];
            betas(4,:,:) = [];
        end
        
        % T-test betas
        [h,p] = ttest(squeeze(betas(:,2:end,1)));
        
        % BMS
        [posterior,out] = VBA_groupBMC(-testdat',options);
        
        fprintf('\nTesting relative vs absolute: %s ~ %s',dvnamez{d},regnamez{r});
        fprintf('\nModel frequencies: %.02f %.02f',out.Ef);
        fprintf('\nProtected exceedance probabilities: %.02f %.02f', out.pxp);
        p
        
    end
end

% - Combining relative with context mean is meaningless (mean of context is equal for all and context then)
% - When comparing absolute vs relative, just use absolute difference regression

%% Plot results

if do.plotting
    
    r           = 2;
    n           = 2;
    d           = 1;
    
    getbetas_m  = regresults{r,1};
    getbetas_a  = regresults{r,2};
    
    betas = cat(3,getbetas_m(:,:,n,d),getbetas_a(:,:,n,d));
            
    if d == 1
        betas(4,:,:) = [];
    end
    
    x           = betas(:,:,1);
    M           = squeeze(mean(x(:,2:end)));
    CI          = 1.96.*squeeze(std(x(:,2:end))./sqrt(size(x,1)));
    nregr       = size(x,2)-1;
    regrnamez   = {'|\Delta|_{DV}','|\Delta|_{ctxt}'};
    
    figR = figure('color',[1 1 1],'position',[100 100 260 350]); hold on;
    
    for r = 1:nregr
        hB = bar(r,M(r),'BarWidth',.5,'FaceColor',[1 1 1].*.6,'EdgeColor',[1 1 1].*.5,'LineWidth',1.5);
        cloudplot(r,x(:,r+1),.35,15,[1 1 1].*.35,[1 1 1].*.35);
        errorbar(r,M(r),CI(r),'Color',[1 1 1].*.0,'LineWidth',2);
    end
    
    ax = gca;
    set(ax,'XTick',[1:nregr],'XTickLabel',regrnamez,'FontSize',axlabelfntsz,'LineWidth',1.5);
    ax.YAxis.FontSize = axfntsz;
    ylabel('\beta coefficients','FontSize',axlabelfntsz);
    
    xlim([.5 nregr+.5]);
    ylim([-.2 1]);
    
    %title('Acc (primary): middle of context','FontSize',axfntsz);
    
    if do.save_plot
        save2eps(figR,paths.figures.current,'Regr_CS_acc_diffmid_context',[.1 .1 260 350]);
    end
    
end