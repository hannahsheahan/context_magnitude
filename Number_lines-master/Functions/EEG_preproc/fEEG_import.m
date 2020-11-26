
function fEEG_import(params,do,s,inputfile,outputfile)
% function fEEG_import(params,do,s,inputfile,outputfile)
%  Function to import and downsample data.
%
% Fabrice Luyckx, 17/10/2017

if do.import
        
    % Restart eeglab because of memory things
    close all
    clear ALLEEG EEG CURRENTSET ALLCOM
    [ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;
    
    % import the CURRY data in eeglab
    fprintf('\nImporting subject %d.\n',params.submat(s));
    EEG             = loadcurry(fullfile(params.data.eeg,inputfile)); % .dat file
    [ALLEEG, EEG]   = eeg_store(ALLEEG, EEG, 1);
    
    % Remove A1 mastoid and eye channels
    for i=1:size(EEG.chanlocs,2)
        labels{i} = EEG.chanlocs(i).labels;
    end
    
    chan2rem = {'A1','EOGV','EOGH'}; % unnecessary/irrelevant channels
    
    for remmy=1:length(chan2rem)
        remz(remmy)=strmatch(chan2rem{remmy},labels);
    end
    
    EEG.chanlocs(remz) 	= [];
    EEG.data(remz,:)    = []; % removing the corresponding data
    EEG.nbchan      	= size(EEG.data,1);
    
    % Re-sample data
    fprintf('\nDownsampling subject %d.\n',params.submat(s));
    EEG = pop_resample(EEG, params.resamp);
    [ALLEEG, EEG, index] = eeg_store(ALLEEG, EEG);
    
    EEG = eeg_checkset(EEG);
    
    fprintf('\nSaving subject %d.\n',params.submat(s));
    EEG.setname     = outputfile;
    [ALLEEG, EEG]   = eeg_store(ALLEEG, EEG, CURRENTSET);
    EEG             = pop_saveset(EEG, outputfile, params.data.saveEEG);
    
    fprintf('\nImporting and downsampling subject %d finished and saved.\n',params.submat(s));
    
end

end