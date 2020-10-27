function [mytext] = printText(textfile)
% function [mytext] = printText(textfile)
%   
% Function to read out and print text from a textfile with PTB.

% Count number of lines in script
fid = fopen(textfile, 'rt');
if fid==-1
    error('Could not open file!');
end
finishLine = linecount(fid);
fclose(fid);

% Open file again
fid = fopen(textfile, 'rt');
if fid==-1
    error('Could not open file!');
end

% Upload text
mytext = '';
tl = fgets(fid);
lcount = 0;
while lcount < finishLine
    mytext = [mytext tl];
    tl = fgets(fid);
    lcount = lcount + 1;
end
fclose(fid);

% Get rid of '% ' symbols at the start of each line:
mytext = strrep(mytext, '% ', '');
mytext = strrep(mytext, '%', '');

end

%%
function n = linecount(fid)
n = 0;
tline = fgetl(fid);
while ischar(tline)
    tline = fgetl(fid);
    n = n+1;
end
end

