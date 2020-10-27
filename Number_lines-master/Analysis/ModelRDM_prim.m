function mods = ModelRDM_prim(s,col,paths)

%% Visual similarity

load(fullfile(paths.data.behav,'Vis_sim_numbers'));

tmp = squareform(visSim);

c1      = tmp(1:11,1:11);
c2      = tmp(6:16,6:16);
c3      = tmp;
c1c2    = tmp(1:11,6:16);
c1c3    = tmp(1:11,1:16);
c2c3    = tmp(6:16,1:16);

pc = [c1 c1c2 c1c3; c1c2' c2 c2c3; c1c3' c2c3' c3];

mods.vissim = squareform(pc);

%% Conditions

c1 = zeros(11,11);
c2 = zeros(11,11);
c3 = zeros(16,16);
c1c2 = ones(11,11);
c1c3 = ones(11,16);
c2c3 = ones(11,16);

pc = [c1 c1c2 c1c3; c1c2 c2 c2c3; c1c3' c2c3' c3];

mods.cond = squareform(pc);

%% Relative magnitude
 
c1 = linspace(0,1,11);
c2 = linspace(0,1,11);
c3 = linspace(0,1,16);

pc = dist([c1 c2 c3]);

mods.rel = squareform(pc);

%% Absolute magnitude

numz = linspace(0,1,16);

pc  = dist([numz(1:11) numz(6:16) numz(1:16)]);

%mods.abs = squareform(pc);

%% Uncomment the following block to include control RDMs

% %% Colour
%
% % CIElab (CIE76)
% lab     = rgb2lab(col);
% colcomb = nchoosek(1:size(col,1),2);
% 
% for c = 1:size(colcomb,1)
%     l1          = lab(colcomb(c,1),:);
%     l2          = lab(colcomb(c,2),:);
%     sumz        = (l1(1)-l2(1)).^2+(l1(2)-l2(2)).^2+(l1(3)-l2(3)).^2;
%     deltaE(c)   = sqrt(sumz);
% end
% 
% c1 = zeros(11,11);
% c2 = zeros(11,11);
% c3 = zeros(16,16);
% c1c2 = ones(11,11).*deltaE(1);
% c1c3 = ones(11,16).*deltaE(2);
% c2c3 = ones(11,16).*deltaE(3);
% 
% pc = [c1 c1c2 c1c3; c1c2 c2 c2c3; c1c3' c2c3' c3];
% 
% mods.col = squareform(pc)./max(squareform(pc));
% 
% %% Reaction time
% 
% load(fullfile(paths.data.behav,'RDM_RT'));
% 
% mods.rt = squareform(rdmRT(:,:,s));

end
