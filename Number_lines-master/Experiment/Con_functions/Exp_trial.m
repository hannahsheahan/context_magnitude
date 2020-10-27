function [data, time, triggers, aborted] = Exp_trial(inEEG,randomise, practice, varargin)
% function [data, time, triggers, aborted] = Exp_trial(inEEG,randomise, practice, [ppnr], [nTri], [nBlo], [low], [hig], [bPriTar], [pSecTar], [colAss])

try
    %% DEFAULT VALUES

    optargs = {99 30 3 [25 35] [30 40] 3 0.12 [1 2 3]};

    % Now put these defaults into the valuesToUse cell array,
    % and overwrite the ones specified in varargin.
    specif = find(~cellfun(@isempty,varargin)); % find position of specified arguments
    [optargs{specif}] = varargin{specif};

    % Place optional args in memorable variable names
    [ppnr, nTri, nBlo, low, hig, bPriTar, pSecTar, colAss] = optargs{:};

    %% Initialise EEG system

    [portobject, portaddress,triggerlength,holdvalue,triggers] = connectEEG(inEEG,practice);

    %% Settings

    % Psychtoolbox defaults
    PsychDefaultSetup(2);

    % Initialize for PsychPortAudio
    InitializePsychSound(1);
    Rate            = 44100;
    deviceid        = [];
    d               = PsychPortAudio('GetDevices');
    mode            = 1; % playback only
    reqlatencyclass = 2; % Request latency mode 2, which used to be the best one in our measurement:
    buffersize      = 0;     % Pointless to set this. Auto-selected to be optimal.
    suggestedLatencySecs = [];
    channels        = 2;

    % Skip sync tests
    %Screen('Preference', 'SkipSyncTests', 1);
    %Screen('Preference', 'VisualDebugLevel', 1);
    HideCursor;	% Hide the mouse cursor
    ListenChar(2); % makes it so characters typed don't show up in the command window
    commandwindow;

    % Keys
    KbName('UnifyKeyNames');
%    esc  	= KbName('NonUS\|'); % for touch pad MBP
%   esc      = KbName('q'); % for German MBP
     esc  	= KbName('Escape');
    space   = KbName('Space');
    higher 	= KbName('UpArrow');
    lower 	= KbName('DownArrow');

    alwaysKeys = ([esc, space, higher, lower]);

    RestrictKeysForKbCheck([alwaysKeys]); % restrict ppt to only use these keys

    %% Data path to save temporary variables

    datapath = cd;
    datafolder = fullfile(datapath,'/Con_data/');

    %% Initialise data and setup

    [data, time] = Exp_setup(randomise, practice, ppnr, nTri, nBlo, low, hig, bPriTar, pSecTar, colAss);

    bTri = nTri/nBlo; % number of trials per block

    time.expStart = GetSecs;

    %% Functions

%     % Accuracy to money
%     moneyBonus = @(maxBonus,cor,nTri) length(find(cor == 1))/nTri * maxBonus;

    % Transform rgb
    rgb = @(x) x/255;     % Transform RGB to 0 - 1 values

     %% Trial variables and open window

    screens         = Screen('Screens');
    screenNumber    = max(screens);

    % COLOUR variables

    % Basic colours
    col.white           = WhiteIndex(screenNumber); % Define black and white (white will be 1 and black 0). This is because
    col.black           = BlackIndex(screenNumber); % luminace values are (in general) defined between 0 and 1.

    % Background colours
    col.background      = col.white / 2;
    col.fix             = [.4 .4 .4].*col.white;

    % //SCREEN variables

    % Open window and window data
    [w, windowRect]                 = PsychImaging('OpenWindow', screenNumber, col.background); % Open an on screen window and color it grey
    [screenXpixels, screenYpixels]  = Screen('WindowSize', w);     % Get the size of the on screen window in pixels
    [scr.xCenter, scr.yCenter]      = RectCenter(windowRect); % Get the centre coordinate of the window in pixels

    % Retrieve the maximum priority number
    topPriorityLevel = MaxPriority(w);
    Priority(topPriorityLevel);
    %Screen('BlendFunction', w, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

    % //TEXT VARIABLES (for instructions or other on-screen text)

    leftMargin      = 50;
    rightMargin     = leftMargin;
    topMargin       = 50;

    titleSize       = 60;
    textSize        = 26;
    stimSize        = 80;
    respSize        = 40;
    standardFont    = 'Calibri';

    Screen('TextFont', w, standardFont); % Font

    % //STIMULUS variables

    % Fixation circle
    scr.fixRad          = 10; % radius of the fixation circle

    % Frame for stimulus presentation
    stim.frameSide      = 300;
    stimRect            = [0 0 stim.frameSide stim.frameSide]; % Frame for stimulus presentation
    rectXpos            = screenXpixels * .5;
    rectYpos            = screenYpixels * .5;
    scr.rectCoord       = CenterRectOnPointd(stimRect, rectXpos, rectYpos);

    % Position of response cues
    dev                 = stim.frameSide/4; % deviation from center
    scr.penForcedResp   = 5;
    boxMargin           = 30;
    boxHeight           = respSize + boxMargin;
    boxWidth            = boxHeight;
    respFrame           = [0 0 boxWidth boxHeight]; % Frame for stimulus presentation
    scr.opt1Xpos        = scr.xCenter - boxWidth/2 - dev;
    scr.opt2Xpos        = scr.xCenter + boxWidth/2 + dev;
    scr.optYpos         = scr.yCenter;
    scr.lRespFrame  	= CenterRectOnPointd(respFrame, scr.opt1Xpos, scr.optYpos); % frame box
    scr.rRespFrame  	= CenterRectOnPointd(respFrame, scr.opt2Xpos, scr.optYpos); % frame box

    % Load feedback sounds
    positive            = psychwavread('Positive.wav');
    negative            = psychwavread('Negative.wav');
    miss                = psychwavread('Miss.wav');
    goodTone            = [positive positive]';
    badTone             = [negative negative]';
    missTone            = [miss miss]';
    %pahandle            = PsychPortAudio('Open', [], [], 0, [], nrchannels,[]);
    pahandle            = PsychPortAudio('Open', deviceid, mode, ...
                            reqlatencyclass, Rate, channels, buffersize, suggestedLatencySecs);

    %% Instructions before experiment

    if practice == 1
        % Instructions before practice
        instructions    = 'beforepractice'; Exp_instructions;
    elseif practice == 0
        % Instructions after practice
        instructions    = 'afterpractice'; Exp_instructions;
    end

    %% Actual trial

    % Display blank (grey) screen
    Screen('TextFont', w, 'Calibri');
    Screen(w,'FillRect',col.background);    % blank screen
    Screen(w,'Flip');                   % write to screen
    WaitSecs(1);

    for t = 1:nTri

        %% Start of a block

        if mod(t,bTri) == 1
            % Instructions before each new block
            instructions = 'startblock'; Exp_instructions;
            [triggers.time.blockStart(t)] = sendTriggers(inEEG,practice,triggers.blockStart+data.blo(t),portobject,portaddress,triggerlength,holdvalue);            

            % Fixation cross
            Screen('DrawDots', w, [scr.xCenter, scr.yCenter] , scr.fixRad, col.white, [], 2);
            Screen(w,'Flip');
            [triggers.time.fixOnset(t)] = sendTriggers(inEEG,practice,triggers.fixOnset,portobject,portaddress,triggerlength,holdvalue);
            WaitSecs(time.fixDur);

            % Adjust text
            Screen('TextStyle',w, 0); % bold
            Screen('TextSize', w , stimSize);

            % Mask
            line1 = '##';
            rct = CenterRectOnPoint(Screen('TextBounds',w,line1),rectXpos,rectYpos);
            Screen('DrawText',w,line1,rct(1),rct(2),col.white);

            Screen(w,'Flip');
            [triggers.time.fixOffset(t)] = sendTriggers(inEEG,practice,triggers.fixOffset,portobject,portaddress,triggerlength,holdvalue);
            WaitSecs(time.stimDur);

            % ISI
            Screen(w,'Flip');
            WaitSecs(1);
        end

        %% Start of a trial

        if practice == 0
            disp(num2str(t));   % print trial number
        end

        %% Stimulus presentation and response recording

        % Stimulus presentation
        line1 = sprintf('%s',num2str(data.num(t)));
        rct = CenterRectOnPoint(Screen('TextBounds',w,line1),rectXpos,rectYpos);
        Screen('DrawText',w,line1,rct(1),rct(2),data.col(t,:));

        [time.trialStart] = Screen(w,'Flip');
        
        if data.ctg(t) >= 1 && data.ctg(t) < 2
            [triggers.time.primOnset(t)] = sendTriggers(inEEG,practice,triggers.primOnset,portobject,portaddress,triggerlength,holdvalue);
        elseif data.ctg(t) == 0
            [triggers.time.fillOnset(t)] = sendTriggers(inEEG,practice,triggers.fillOnset,portobject,portaddress,triggerlength,holdvalue);
        elseif data.ctg(t) == 2
            [triggers.time.secOnset(t)] = sendTriggers(inEEG,practice,triggers.secOnset,portobject,portaddress,triggerlength,holdvalue);  
        end
        
        % Response recording
        press           = 0;
        keycode         = 0;
        elapsed         = 0;
        played          = 0; % fb played?
        flipped         = 0;
        totalTrialTime  = time.stimDur + time.ISI(t);
        
        WaitSecs(time.respBuffer);

        while elapsed < totalTrialTime
            elapsed = GetSecs - time.trialStart;
            [kdown, ~, codes] = KbCheck;  % check for key press

            % interstimulus interval
            if elapsed >= time.stimDur && flipped == 0
                Screen(w,'Flip');
                % Triggers offset
                if data.ctg(t) >= 1 && data.ctg(t) < 2
                    [triggers.time.primOffset(t)] = sendTriggers(inEEG,practice,triggers.primOffset,portobject,portaddress,triggerlength,holdvalue);
                elseif data.ctg(t) == 0
                    [triggers.time.fillOffset(t)] = sendTriggers(inEEG,practice,triggers.fillOffset,portobject,portaddress,triggerlength,holdvalue);
                elseif data.ctg(t) == 2
                    [triggers.time.secOffset(t)] = sendTriggers(inEEG,practice,triggers.secOffset,portobject,portaddress,triggerlength,holdvalue);
                end
                flipped = 1;
            end

            % check escape key
            if kdown==1 && elapsed < (totalTrialTime - time.fbDeadline)
                if codes(esc)

                    % Save data
                    if practice == 0
                        tmpname = sprintf('datatmp_%d.mat',ppnr);
                        save(fullfile(datafolder,tmpname),'data','stim','time');
                    end

                    aborted = 1;
                    sca;
                    closeStuff();

                    if inEEG
                        CloseIOPort;
                    end

                    return;
                end

                press = press+1;

                if press == 1

                    if codes(lower) == 1 || codes(higher) == 1 || codes(space) == 1
                        data.RT(t,1)        = GetSecs - time.trialStart;    % log RT
                        keycode             = find(codes==1);               % which button
                        data.keycode(t,1)   = keycode(1);                   % take only first in case of simultaneous press
                    end

                    % Feedback

                    % Response
                    if data.keycode(t,1) == lower
                        data.r(t,1) = 1;
                        [triggers.time.downResp(t)] = sendTriggers(inEEG,practice,triggers.downResp,portobject,portaddress,triggerlength,holdvalue);
                    elseif data.keycode(t,1) == higher
                        data.r(t,1) = 2;
                        [triggers.time.upResp(t)] = sendTriggers(inEEG,practice,triggers.upResp,portobject,portaddress,triggerlength,holdvalue);
                    elseif data.keycode(t,1) == space
                        data.r(t,1) = 3;
                        [triggers.time.spaceResp(t)] = sendTriggers(inEEG,practice,triggers.spaceResp,portobject,portaddress,triggerlength,holdvalue);
                    end

                    % Correct?
                    if data.xr(t,1) == data.r(t,1)
                        data.cor(t,1) = 1;
                        PsychPortAudio('FillBuffer', pahandle, goodTone); % audio
                        PsychPortAudio('Start', pahandle);
                        [triggers.time.fbOnset(t)] = sendTriggers(inEEG,practice,triggers.fbOnset,portobject,portaddress,triggerlength,holdvalue);
                        WaitSecs(time.fbTime);
                        PsychPortAudio('Stop', pahandle); % Stop playback
                        [triggers.time.fbOffset(t)] = sendTriggers(inEEG,practice,triggers.fbOffset,portobject,portaddress,triggerlength,holdvalue);
                        played = 1;
                    else
                        data.cor(t,1) = 0;
                        PsychPortAudio('FillBuffer', pahandle, badTone); % audio
                        PsychPortAudio('Start', pahandle);
                        [triggers.time.fbOnset(t)] = sendTriggers(inEEG,practice,triggers.fbOnset,portobject,portaddress,triggerlength,holdvalue);
                        WaitSecs(time.fbTime);
                        PsychPortAudio('Stop', pahandle); % Stop playback
                        [triggers.time.fbOffset(t)] = sendTriggers(inEEG,practice,triggers.fbOffset,portobject,portaddress,triggerlength,holdvalue);
                        played = 1;
                    end
                end
            end

            % Feedback too late
            if elapsed > (totalTrialTime - time.fbDeadline) && played == 0 && data.xr(t) ~= 0
                PsychPortAudio('FillBuffer', pahandle, missTone); % audio
                PsychPortAudio('Start', pahandle);
                WaitSecs(time.fbTime);
                [triggers.time.fbOnset(t)] = sendTriggers(inEEG,practice,triggers.fbOnset,portobject,portaddress,triggerlength,holdvalue);
                PsychPortAudio('Stop', pahandle); % Stop playback
                [triggers.time.fbOffset(t)] = sendTriggers(inEEG,practice,triggers.fbOffset,portobject,portaddress,triggerlength,holdvalue);
                played = 1;
            end

        end

        % No response
        if data.r(t,1) == -99
            data.r(t,1) = 0;
        end

        % Incorrect?
        if data.r(t,1) == 0 && data.xr(t,1) ~= data.r(t,1)
            data.cor(t,1) = 0;
        elseif data.r(t,1) == 0 && data.xr(t,1) == data.r(t,1)
            data.cor(t,1) = 1;
        end
        
        time.trialEnd = GetSecs;
        time.trialDur(t) = time.trialEnd - time.trialStart;

        %% End of a block

        if mod(t,bTri) == 0

            WaitSecs(time.EBI);

            accPriTar = (sum(data.blo == data.blo(t) & data.ctg ~= 0 & data.ctg ~= 2 & data.cor == 1)-1)/(sum(data.blo == data.blo(t) & data.ctg ~= 0 & data.ctg ~= 2)-1);
            accSecTar = 0.5*sum(data.blo == data.blo(t) & data.ctg == 2 & data.cor == 1)/sum(data.blo == data.blo(t) & data.ctg == 2) + 0.5*sum(data.blo == data.blo(t) & data.ctg == 0 & data.cor == 1)/sum(data.blo == data.blo(t) & data.ctg == 0);
            % accSecTar = 50% * responses tp secondary target +50% reponses
            % to fillers
            if accSecTar < 0
                accSecTar = 0;
            end

            % Text on screen
            instructions = 'endblock'; Exp_instructions;

            if practice == 0
                % Save data from block
                tmpname = sprintf('datatmp_%d.mat',ppnr);
                save(fullfile(datafolder,tmpname),'data','stim','time');
            end
            
        end

    end

    %% End experiment

    if practice == 0

        % Translate points into money
%         money = roundn(moneyBonus(2.5,data.cor,ntrials),-1);
        
        absAccPriTar = (sum((data.ctg == 1.1 | data.ctg == 1.2) & data.cor == 1)-1)/(sum(data.ctg == 1.1 | data.ctg == 1.2)-1);

        absAccSecTar = 0.5*(sum(data.ctg == 2 & data.cor == 1)/sum(data.ctg == 2)) + 0.5*(sum(data.ctg == 0 & data.cor == 1)/sum(data.ctg == 0));
        
        absAcc = min([absAccPriTar absAccSecTar]);
        
        money = roundn(5*absAcc,-1)

        % Instructions at end of experiment
        instructions = 'endexp'; Exp_instructions;

        % Get length of experiment
        time.expEnd     = GetSecs;
        time.expDur     = time.expEnd - time.expStart;

        % Close screen
        sca
    end

    % Close all
    aborted = 0;
    closeStuff();

    % Close EEG port
    if inEEG
        CloseIOPort;
    end

catch ME

    warning('Something went wrong in the experiment.');

    tmpname = sprintf('datatmp_%d.mat',ppnr);
    save(fullfile(datafolder,tmpname),'data','time');

    aborted = 1;
    sca;
    closeStuff();

    if inEEG
        CloseIOPort;
    end

    % Get length of experiment.
    if time.expStart > 0
        time.expEnd     = GetSecs;
        time.expDur     = time.expEnd - time.expStart;
    end

    rethrow(ME)

end

end

%% Commands for closure

function closeStuff()

ShowCursor;
ListenChar(0);
Priority(0);
PsychPortAudio('Close'); % Close the audio device

end

%% Connect EEG

function [portobject, portaddress,triggerlength,holdvalue,triggers] = connectEEG(inEEG,practice)

if inEEG && practice == 0

    IOPortfolder = 'C:\Users\csummerfield\Documents\MATLAB\IOPort';
    addpath(IOPortfolder);

    [portobject, portaddress] = OpenIOPort;
    triggerlength = 0.005; %send trigger for 5
    holdvalue     = 0;

    disp('EEG system initialised.');

else
    portobject      = [];
    portaddress     = [];
    triggerlength   = [];
    holdvalue       = [];
end

% Define triggers
triggers = struct();
triggers.blockStart   	= 100;  % block onset
triggers.fixOnset   	= 1;    % fixation onset
triggers.fixOffset      = 2;    % fixation offset
triggers.primOnset  	= 11;   % primary target onset
triggers.primOffset     = 12;   % primary target offset
triggers.fillOnset  	= 21;   % filler onset
triggers.fillOffset     = 22;   % filler offset
triggers.secOnset       = 23;   % secondary target onset
triggers.secOffset      = 24;   % secondary target offset
triggers.downResp       = 31;   % response down
triggers.upResp         = 32;   % response up
triggers.spaceResp      = 33;   % response space
triggers.fbOnset        = 41;   % feedback onset
triggers.fbOffset       = 42;   % feedback offset

triggers.time.blockStart   	= [];  % block onset
triggers.time.fixOnset   	= [];    % fixation onset
triggers.time.fixOffset      = [];    % fixation offset
triggers.time.primOnset  	= [];   % primary target onset
triggers.time.primOffset     = [];   % primary target offset
triggers.time.fillOnset  	= [];   % filler onset
triggers.time.fillOffset     = [];   % filler offset
triggers.time.secOnset       = [];   % secondary target onset
triggers.time.secOffset      = [];   % secondary target offset
triggers.time.downResp       = [];   % response down
triggers.time.upResp         = [];   % response up
triggers.time.spaceResp      = [];   % response space
triggers.time.fbOnset        = [];   % feedback onset
triggers.time.fbOffset       = [];   % feedback offset

end

%% Send triggers

function [trigpoint] = sendTriggers(inEEG,practice,trig,portobject,portaddress,triggerlength,holdvalue)

if inEEG && practice == 0
     io64( portobject, portaddress, trig); %this sends the trigger
     trigpoint = GetSecs;
     WaitSecs(triggerlength);
     io64( portobject, portaddress, holdvalue ); %this sets the trigger channel back to its hold value (0)
else
    trigpoint = 0;
end

end