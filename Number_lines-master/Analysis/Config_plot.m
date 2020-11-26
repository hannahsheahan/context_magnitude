%%%%%%%%%%%%%%%%%%%%%%%%
%% DEFINE plot variables
%%%%%%%%%%%%%%%%%%%%%%%%

%% Functions

% Function to make matrix one long column
makeLong = @(x) x(:);

% Function to calculate color matrices
rgb     = @(x) round(x./255,2);

% Standard error
sem     = @(x,n) nanstd(x)./sqrt(n);

%% Plot variables

axfntsz         = 20;
axlabelfntsz    = 24;
titlefntsz      = 24;
lgndfntsz       = 20;
lnwid           = 2;
mksz            = 10;
barwid          = .9;

set(0,'DefaultAxesFontName', 'Helvetica');
set(0,'DefaultTextFontname', 'Helvetica');
set(0,'DefaultLineLineWidth',2);
set(0,'DefaultAxesLineWidth',1.5);

modmark     = {'d','x'};
markers 	= {'o','s','d','x','h','p','+','*'};

%% Colours

colz(1,:)       = rgb([115,180,102]); % green
colz(2,:)       = rgb([245,106,100]); % red
colz(3,:)       = rgb([50,137,184]); % dark blue
colz(4,:)       = rgb([255,190,0]); % orange
colz(5,:)       = rgb([38,38,38]); % dark grey
colz(6,:)       = rgb([255 0 120]); % dark grey
colzedge        = colz*.8;

% Colours for three contexts
condcolz(1,:)   = rgb([0,68,136]);
condcolz(2,:)   = rgb([34,85,85]);
condcolz(3,:)   = rgb([221,170,51]);
condcolzedge 	= condcolz.*.8;

%%

condnamez   = {'Low','High','Full'};
numnamez    = {{'25','26','27','28','29','30','31','32','33','34','35'},...
                {'30','31','32','33','34','35','36','37','38','39','40'},...
                {'25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40'}};
