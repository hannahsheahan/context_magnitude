%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Con_numbers: visual similarity of numbers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Requires Psychtoolbox to create stimuli

clear

% Paths
eegfolder   = []; % unused here
savefolder  = []; % unused here
Load_paths_and_data;

% Logicals
do.plotting         = true; % plot number arrays?
do.save_plot        = true; % save plot of numbers?
do.save_data        = true; % save similarity data?

%% Create stimuli

num = 25:40;

try
    
    % Skip sync tests
    Screen('Preference', 'SkipSyncTests', 1);
    
    % N screens
    screens         = Screen('Screens');
    screenNumber    = max(screens);
    
    % Basic colours
    col.white           = WhiteIndex(screenNumber); % Define black and white (white will be 1 and black 0). This is because
    col.black           = BlackIndex(screenNumber); % luminace values are (in general) defined between 0 and 1.
    
    % Background colours
    col.background      = col.white / 2;
    col.fix             = [.4 .4 .4].*col.white;
    
    % //SCREEN variables
    
    % Open window and window data
    [w, windowRect]                 = PsychImaging('OpenWindow', screenNumber, col.white); % Open an on screen window and color it grey
    [screenXpixels, screenYpixels]  = Screen('WindowSize', w);     % Get the size of the on screen window in pixels
    [scr.xCenter, scr.yCenter]      = RectCenter(windowRect); % Get the centre coordinate of the window in pixels
    
    % Retrieve the maximum priority number
    topPriorityLevel = MaxPriority(w);
    Priority(topPriorityLevel);
    
    % Frame for stimulus presentation
    stim.frameSide      = 300;
    stimRect            = [0 0 stim.frameSide stim.frameSide]; % Frame for stimulus presentation
    rectXpos            = screenXpixels * .5;
    rectYpos            = screenYpixels * .5;
    scr.rectCoord       = CenterRectOnPointd(stimRect, rectXpos, rectYpos);
    
    % //TEXT VARIABLES (for instructions or other on-screen text)
    
    leftMargin      = 50;
    rightMargin     = leftMargin;
    topMargin       = 50;
    
    stimSize        = 80;
    standardFont    = 'Calibri';
    
    Screen('TextFont', w, standardFont); % Font
    Screen('TextStyle',w, 0); % bold
    Screen('TextSize', w , stimSize);
    
    %% Loop through numbers
    
    rct = [920 500 1000 553]; % box in which number is printed
    
    for n = 1:length(num)
        line1   = sprintf('%d',num(n));
        Screen('DrawText',w,line1,rct(1),rct(2),col.black);
        
        Screen(w,'Flip');
        
        WaitSecs(.1);
        
        imArray{n} = Screen('GetImage',w,rct);
    end
    
    %% Close window
    
    sca
    
catch ME
    sca
    rethrow(ME)
end

%% Print numbers

if do.plotting
    figure;
    for n = 1:length(num)
        subplot(4,4,n);
        imshow(imArray{n});
    end
    
    if do.save_plot
        save2png(gcf,paths.figures.current,'Visual_similarity_numbers',[.1 .1 600 500]);
    end
end

%% Calculate correlation distance between two numbers

visSim = zeros(length(num),length(num));

for r = 1:length(num)
    for s = 1:length(num)
        visSim(r,s) = 1-corr2(mean(imArray{r},3),mean(imArray{s},3));
    end
end

% Vectorize dissimilarity matrix
visSim = squareform(visSim);

%% Save data

if do.save_data
    save(fullfile(paths.data.behav,'Vis_sim_numbers'),'visSim');
end


