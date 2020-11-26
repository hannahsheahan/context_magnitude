%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% RDM RT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% LOAD DATA

clc
clear
%close all

eegfolder       = []; % unused here
savefolder      = []; % unused here

% Load stuff
Load_paths_and_data; % load data and path
Config_plot; % load plot variables

% Logicals
do.save_rt      = true; % save RT rdm?
do.plotting     = true;

%% EXTRA VARIABLES

data.target = data.ctg;
data.target(data.target > 0 & data.target < 2) = 1;

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

% Difference primary targets
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
    end
end

%% RDM RT (distances of average RT per number)

orgidx  = data.target == 1;

tmpnum  = data.num;
tmpnum(orgidx & data.mod == 2) = tmpnum(orgidx & data.mod == 2)+20;
tmpnum(orgidx & data.mod == 3) = tmpnum(orgidx & data.mod == 3)+50;
tmpnum(~orgidx) = nan;

unum    = unique(tmpnum(~isnan(tmpnum)));
totnum  = length(unum);
numRT   = nan(params.nsubj,totnum);
rdmRT   = nan(totnum,totnum,params.nsubj);

% Loop through subjects to get average RT per number
for s = 1:params.nsubj
    
    fprintf('\nSub %d\n',s);
    
    idx = data.sub == params.submat(s) & (data.r == 1 | data.r == 2) & data.ctg > 0 & data.ctg < 2 & ...
        inv_rt >= outRT(s,1) & inv_rt <= outRT(s,2);
    
    subRT = zscore(data.RT(idx));
    
    for n = 1:totnum
        nidx        = tmpnum(idx) == unum(n);
        numRT(s,n)  = mean(subRT(nidx));
    end
    
    % Create RDM
    rdmRT(:,:,s)    = dist(numRT(s,:));
end

if do.save_rt
    save(fullfile(paths.data.behav,'RDM_RT'),'rdmRT');
    fprintf('\nRDM RT saved.\n');
end

%% Plot RT RDM (group average)

if do.plotting
    
    plotdat = mean(rdmRT,3);
    maplims = [min(squareform(plotdat)) max(squareform(plotdat))];
    
    numz = {'25'  '26'  '27'  '28'  '29'  '30'  '31'  '32'  '33'  '34'  '35' ...
        '30'  '31'  '32'  '33'  '34'  '35' '36'  '37'  '38'  '39'  '40' ...
        '25'  '26'  '27'  '28'  '29'  '30'  '31'  '32'  '33'  '34'  '35' '36'  '37'  '38'  '39'  '40'};
    
    
    nx = length(numz);
    ny = nx;
    xtickz = linspace(0.5,nx+0.5,nx+1);
    ytickz = linspace(0.5,ny+0.5,ny+1);
    linecolz = [.6 .6 .6];
    
    figR = figure;
    ct = cbrewer('seq','YlGn',100);
    colormap(ct);
    
    hi = imagesc(plotdat,maplims); hold on;
    
    cm = viridis();
    colormap(cm);
        
    axis square
    ax  = gca;
    set(ax,'TickLength',[0 0]);
    
    set(ax,'xtick', xtickz, 'ytick', ytickz);
    set(ax,'xgrid', 'on', 'ygrid', 'on', 'gridlinestyle', '-','GridAlpha',1,'GridColor',linecolz);
    set(ax,'LineWidth',.5);
    
    plot([xtickz(1) xtickz(end)],[ytickz(12) ytickz(12)],'Color',linecolz,'LineWidth',3);
    plot([xtickz(1) xtickz(end)],[ytickz(23) ytickz(23)],'Color',linecolz,'LineWidth',3);
    plot([xtickz(12) xtickz(12)],[ytickz(1) ytickz(end)],'Color',linecolz,'LineWidth',3);
    plot([xtickz(23) xtickz(23)],[ytickz(1) ytickz(end)],'Color',linecolz,'LineWidth',3);
    
    set(ax,'XTickLabel',[],'YTickLabel',[]);
   
end
