function [data, time] = Exp_setup(randomise, varargin)
% [data, time] = Exp_setup(randomise, [practice], [ppnr], [nTri], [nBlo], [low], [hig], [bPriTar], [pSecTar], [colAss])

%% DEFAULT VALUES

optargs = {0 99 36 3 [25 35] [30 40] 3 0.12 [1 2 3]};

% Now put these defaults into the valuesToUse cell array,
% and overwrite the ones specified in varargin.
specif = find(~cellfun(@isempty,varargin)); % find position of specified arguments
[optargs{specif}] = varargin{specif};

% Place optional args in memorable variable names
[practice, ppnr, nTri, nBlo, low, hig, bPriTar, pSecTar, colAss] = optargs{:};

%% Initialise variables

[data] = Exp_initialise(ppnr,nTri,nBlo,low,hig,bPriTar,pSecTar);

bTri = nTri/nBlo;      % number of trials per block

makeLong = @(x) x(:); % function to squelch data

%% Get screen variables

nScreens        = Screen('Screens');
screenNumber    = max(nScreens);
screen          = Screen('Resolution',screenNumber);

%% Data variables

% Variables for data
data.sub    = ppnr*ones(nTri,1);                     % subject index
data.blo  = makeLong(repmat(1:nBlo,bTri,1));    % block index

com = [low(1) hig(2)];

modOpt = {low, hig, com};                                       % different options for the mode of a block

modOrd = [ones(1, nBlo/3) 2*ones(1, nBlo/3) 3*ones(1, nBlo/3)]; % order of modes (1 = low, 2 = high, 3 = combined)

if randomise == 1 && practice == 0                              % randomising order of modes
    modOrd = Shuffle(modOrd);
end

data.mod = makeLong(repmat(modOrd,bTri,1));                     % vector coding the mode of the block for each trial

fil2 = [1 0 2];
fil3 = [1 0 2 2];
fil4 = [1 0 2 2 2];

ctgOpt = {fil2 fil3 fil4};

ctgOrd = [ones(1, bTri/12) 2*ones(1, bTri/12) 3*ones(1, bTri/12)];

data.num = []; % numbers that will be presented in RSVP
data.ctg = []; % categories of that numbers (0 = filler number, 1 = first
% primary target, 1.1 = primary target lower than previous
% one, 1.2 = primary target higher than previous one)
data.col = []; % colours

% Colours

% Colour assignment

% colAss = [1 2 3];
% if randomise == 1
%     colAss = Shuffle(colAss); % first number indicates colour of low range
%                               % blocks, second number indicates colour of
%                               % high range blocks, and third number
%                               % indicates colour of combined range blocks
%                               % (1 = blue, 2 = orange, 3 = purple)
% end

screens         = Screen('Screens');
screenNumber    = max(screens);

col.white           = WhiteIndex(screenNumber); % white

% Transform rgb
rgb = @(x) x/255;     % Transform RGB to 0 - 1 values

col.category(1,:)  	= rgb([86 180 233]).*col.white; % blue
col.category(2,:)  	= rgb([230 159 0]).*col.white; % orange
col.category(3,:)  	= rgb([230,120,220]).*col.white; % purple

for i = 1:nBlo
    temMod = modOpt{modOrd(i)};
    
    if randomise == 1
        ctgOrd = Shuffle(ctgOrd);
    end
    
    temCtg = [ctgOpt{ctgOrd(:)}]';
    
    temNum = randi(com, bTri, 1);
    temNum(temCtg == 1) = randi(temMod, bPriTar, 1);
    
    temCol = repmat([1 1 1]*col.white,bTri,1);
    for j = 1:bTri
        if temCtg(j) == 1
            temCol(j, :) = col.category(colAss(modOrd(i)),:);
        end
    end
    
    ind = find(temCtg == 1);
    k = 1;
    for l = 2:bTri
        if ~(ind(k) == ind(end)) & l == ind(k+1)
            while temNum(l) == temNum(ind(k))
                temNum(l) = randi(temMod);
            end
            if temNum(l) < temNum(ind(k))
                temCtg(l) = 1.1;
            elseif temNum(l) > temNum(ind(k))
                temCtg(l) = 1.2;
            end
            k = k+1;
        elseif temNum(l) == temNum(ind(k))
            while temCtg(l) == 0 && temNum(l) == temNum(ind(k))
                temNum(l) = randi(com);
            end
        elseif temCtg(l) == 2 && rand <= pSecTar
            temNum(l) = temNum(ind(k));
        else
            temCtg(l) = 0;
        end
    end
    
    data.num = [data.num; temNum];
    data.ctg = [data.ctg; temCtg];
    data.col = [data.col; temCol];
end

%% Calculate rest

data.xr = zeros(nTri, 1);                       % no respose expected
data.xr(data.ctg == 1.1) = 1;                   % lower
data.xr(data.ctg == 1.2) = 2;                   % higher
data.xr(data.ctg == 2) = 3;                     % same

%% Time variables

time.framedur       = 1/screen.hz;                  % approximate frame duration
if time.framedur == Inf
    time.framedur = .016666;
    warning('No frame duration registered.');
end

time.fixDur     = 1;                    % fixation duration
time.stimDur    = 0.5;                 	% duration of each sample
time.respBuffer = 0.2;                  % buffer before response collection

if practice == 0
    time.ISI        = 0.75 + 0.5*rand(nTri,1);      % interstimulusinterval exp
elseif practice == 1
    time.ISI        = 0.75 + 0.5*rand(nTri,1);      % interstimulusinterval pract
end

time.ISI(data.ctg == 1.1 | data.ctg == 1.2) = 1; % extra ISI time for primary targets

time.rISI       = time.stimDur + time.ISI; % response cue interval
time.fbISI      = .1;                   % before feedback
time.fbTime     = .15;            % feedback time
time.fbDeadline = time.fbTime;      % deadline for pressing before next stimulus
time.EBI        = 1;                   	% end block interval

time.expStart   = 0;                    % start of experiment
time.expEnd     = 0;                    % end of experiment
time.expDur     = 0;                    % duration of experiment
time.trialStart = zeros(nTri,1);                    % start of trial
time.trialEnd   = zeros(nTri,1);                    % end of trial
time.trialDur   = zeros(nTri,1);     % duration of trial
time.presEnd    = 0;                    % end of sample presentation
time.deadline   = 2;                    % response deadline

end