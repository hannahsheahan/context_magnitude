% Author: Stephanie Nelli
% Date: Jan? 2020

%% note: feel free to clean things up or do some better optimisation scheme. this
%is definitely a first draft :)
clear
close all

% load RDMS
load('matlab_constantcontextlabel_recurrentnet_meanactivations.mat')
activations = arr
rdm{1} = pdist(activations);

%load('trueConlab_activations.mat')
%rdm{2} = pdist(f);
descriptData = {['retainNet']};
% max and min parameter values based on a priori hypotheses
% its the context seperataion, the normalisation, the low offset and high
% offset (param 5 is unused as of now)
minp    = [0 9/15 -6/15 -5/15 -1]; % -10/15];
maxp    = [2 15/15 6/15  5/15  1]; % 10/15];
% context, norm, offset
rdmIdx = 1
%for rdmIdx = 1:2
    Y = rdm{rdmIdx}./max(max(abs(rdm{rdmIdx}))); %zeros(1,703); % it only matters for fitting
    
    clear tparam
    for init = 1:10 %fit it several times with random initialisataions
        initp   = [rand/5 rand/5 rand/5 rand/5 rand/5]; %small random starts
        
        fun = @(p)sphere_lines_fit(p, Y);
        temp_param(init,:)  = fmincon(fun, initp, [], [], [], [], minp, maxp);
    end
    % you can either take the average of the above if convergence is good
    % or take one or two examples.
    params(rdmIdx,:) = mean(temp_param); 
    
    params_plot = [0, 0, 0 0, 0 0, params(rdmIdx,1), params(rdmIdx,2), params(rdmIdx,3), params(rdmIdx,4), params(rdmIdx,5)]; % parallel configuration with relative scaling
    
    lineLen = 15;
    theta1      = params_plot(1); % low range
    phi1        = params_plot(2); % low range
    theta2      = params_plot(3); % high range
    phi2        = params_plot(4); % high range
    theta3      = params_plot(5); % full range
    phi3        = params_plot(6); % full range
    cdist       = params_plot(7); % distance from center
    normline    = params_plot(8);
    % normalize low and high line ranges
    % ex: 0.5 is half the length of the long line, 1 = all same size
    %2d0 change back so norm is same for high and low
    low_offset = params_plot(9); %chaning this changes the low high offset.
    high_offset = params_plot(10); %chaning this changes the low high offset.
    %           full_offset = params_plot(11);% note: offset isnt consistent with 5/15, its below this
    % also note, scaling of this depends on cdist (i.e. distance between
    % contexts vs distances within context)
    
    %% Coordinates based on angles. dont really need to change anything below 
    
    % Normalize lines (= change length)
    linelength = (lineLen*normline)/lineLen;
    
    % Coordinates low range
    x1 = (linelength/2)*sin(theta1)*cos(phi1);
    y1 = (linelength/2)*sin(theta1)*sin(phi1);
    z1 = (linelength/2)*cos(theta1);
    
    lowline(:,1) = linspace(-x1,x1,lineLen-5);
    lowline(:,2) = linspace(-y1,y1,lineLen-5) + cdist/sqrt(3);
    lowline(:,3) = linspace(-z1,z1,lineLen-5) + low_offset;
    
    linelength = (lineLen*normline)/lineLen;
    
    % Coordinates high range
    x2 = (linelength/2)*sin(theta2)*cos(phi2);
    y2 = (linelength/2)*sin(theta2)*sin(phi2);
    z2 = (linelength/2)*cos(theta2);
    
    highline(:,1) = linspace(-x2,x2,lineLen-5) - cdist/2;
    highline(:,2) = linspace(-y2,y2,lineLen-5) - cdist*tand(30)/2;
    highline(:,3) = linspace(-z2,z2,lineLen-5) + high_offset;
    
    % Coordinates full range (always same length)
    x3 = .5*sin(theta3)*cos(phi3);
    y3 = .5*sin(theta3)*sin(phi3);
    z3 = .5*cos(theta3);
    
    fullline(:,1) = linspace(-x3,x3,lineLen) + cdist/2;
    fullline(:,2) = linspace(-y3,y3,lineLen) - cdist*tand(30)/2;
    fullline(:,3) = linspace(-z3,z3,lineLen); % + full_offset;
    
    
    %% Distance matrix
    
    numline = [lowline; highline; fullline];
    
    
    %% Plot lines
    %%
    figure(1), hold on,
    subplot(3,2,rdmIdx),
    hold on;
    plot3(lowline(:,1),  lowline(:,2), lowline(:, 3),'ro', 'LineWidth', 2);
    plot3(highline(:,1), highline(:,2),highline(:,3),'go', 'LineWidth', 2);
    plot3(fullline(:,1), fullline(:,2),fullline(:,3),'bo', 'LineWidth', 2);
    hold on, title(['fit for ', descriptData{rdmIdx}])
    set(gca, 'FontSize', 14)
    legend('low', 'high', 'full')
    axis square
    view([-50 15])
    
    % figure(1), hold on, subplot(2,1,rdmIdx),
    % hold on;
    % plot3(numline(1:lineLen-5,1),                      numline(1:size(lowline,1),2),numline(1:size(lowline,1),3),'ro');
    % plot3(numline(lineLen-5+1:lineLen-5+lineLen-5,1),  numline(size(lowline,1)+1:size(lowline,1)+size(highline,1),2),numline(size(lowline,1)+1:size(highline,1)+size(lowline,1),3),'go');
    % plot3(numline(end-lineLen+1:end,1),                numline(end-size(fullline,1)+1:end,2),numline(end-size(fullline,1)+1:end,3),'bo');
    % axis square
    
    pred_RDM = dist(numline.');
    
    figure(1),
    subplot(3,2,rdmIdx+2),
    imagesc(pred_RDM)
    hold on, title(['fit for ', descriptData{rdmIdx}])
    set(gca, 'FontSize', 14)
    axis square
    
    figure(1),
    subplot(3,2,rdmIdx+4),
    imagesc(fliplr(flipud(squareform(rdm{rdmIdx}))))
    hold on, title(['actual RDM ', descriptData{rdmIdx}])
    set(gca, 'FontSize', 14)
    axis square
%end
