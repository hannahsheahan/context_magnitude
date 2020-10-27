function [data] = Exp_initialise(varargin)
% initialise all variables
% function [data, stim] = Exp_initialise([ppnr],[nTri], [nBlo], [low], [hig], [bPriTar], [pSecTar])

%% DEFAULT VALUES

optargs = {99 30 3 [25 35] [30 40] 3 0.12};

% Now put these defaults into the valuesToUse cell array, 
% and overwrite the ones specified in varargin.
specif = find(~cellfun(@isempty,varargin)); % find position of specified arguments
[optargs{specif}] = varargin{specif};

% Place optional args in memorable variable names
[ppnr, nTri, nBlo, low, hig, bPriTar, pSecTar] = optargs{:};

bTri = nTri/nBlo;      % number of trials per block

%% Initialise all necessary variables

% Variables for data
data.sub        = -99*ones(nTri,1);

data.trial      = [1:nTri]';
data.blo        = zeros(nTri,1);
data.mod        = zeros(nTri,1);

data.num        = zeros(nTri,1);
data.ctg        = zeros(nTri,1);

data.xr         = -99*ones(nTri,1);  % expected response
data.r          = -99*ones(nTri,1); % ppt response
data.keycode    = -99*ones(nTri,1); % code of key pressed
data.cor        = -1*ones(nTri,1); % correct
data.RT         = -99*ones(nTri,1); % reaction time

end