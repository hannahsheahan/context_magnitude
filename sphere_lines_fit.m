


function [SSE,pred,numline,coord] = sphere_lines_fit(p, Y)
%close all
% SN added:
% p = parameters
% Y = neural RDMs (703 x 703)
% Orthogonal = (0,0) - (pi/2,pi/2) - (pi/2,0)

theta1      = 0; % low range
phi1        = 0; % low range
theta2      = 0; % high range
phi2        = 0; % high range
theta3      = 0; % full range
phi3        = 0; % full range
cdist       = p(1); % distance from center
normline    = p(2);
%high_normline    = p(3);

% normalize low and high line ranges
% ex: 0.5 is half the length of the long line, 1 = all same size
%full_offset = p(5); 
offset = p(3); %chaning this changes the low high offset.
hoffset = p(4); %chaning this changes the low high offset.
% note: offset isnt consistent with 5/15, its below this
% also note, scaling of this depends on cdist (i.e. distance between
% contexts vs distances within context)

%% Coordinates based on angles
lineLen = 15;
% Normalize lines (= change length)
linelength = (lineLen*normline)/lineLen;

% Coordinates low range
x1 = (linelength/2)*sin(theta1)*cos(phi1);
y1 = (linelength/2)*sin(theta1)*sin(phi1);
z1 = (linelength/2)*cos(theta1);

lowline(:,1) = linspace(-x1,x1,lineLen-5);
lowline(:,2) = linspace(-y1,y1,lineLen-5) + cdist/sqrt(3);
lowline(:,3) = linspace(-z1,z1,lineLen-5) + offset;

linelength = (lineLen*normline)/lineLen;

% Coordinates high range
x2 = (linelength/2)*sin(theta2)*cos(phi2);
y2 = (linelength/2)*sin(theta2)*sin(phi2);
z2 = (linelength/2)*cos(theta2);

highline(:,1) = linspace(-x2,x2,lineLen-5) - cdist/2;
highline(:,2) = linspace(-y2,y2,lineLen-5) - cdist*tand(30)/2;
highline(:,3) = linspace(-z2,z2,lineLen-5) + hoffset;

% Coordinates full range (always same length)
x3 = .5*sin(theta3)*cos(phi3);
y3 = .5*sin(theta3)*sin(phi3);
z3 = .5*cos(theta3);

fullline(:,1) = linspace(-x3,x3,lineLen) + cdist/2;
fullline(:,2) = linspace(-y3,y3,lineLen) - cdist*tand(30)/2;
fullline(:,3) = linspace(-z3,z3,lineLen); % + full_offset;


%% Distance matrix

numline = [lowline; highline; fullline];
pred  = pdist(numline); pred = pred./max(abs(pred(:)));

%% SSE for fit

%SSE = sum((zscore(Y)' - zscore(pred)').^2);
%tY = tril(squareform(Y)); tY(tY==0) = [];
%tPred = tril(squareform(pred)); tPred(tPred==0)=[];
tY = squareform(Y); tY(1:15,1:15) =nan; tY(16:25,16:25) = nan; tY(26:35,26:35) = nan;
tY(isnan(tY)) = []; %tY

tP = squareform(pred); tP(1:15,1:15) =nan; tP(16:25,16:25) = nan; tP(26:35,26:35) = nan;
tP(isnan(tP)) = []; %tY

SSE = sum((tY' - tP').^2);

end
%%

 