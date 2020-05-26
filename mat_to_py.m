clc;
clear all;
load('targetRDM.mat');
nsubs = size(allrdm,1);
timeslices = 201:300;
for i=1:nsubs
    temp = squeeze(allrdm(i,:,:));
    temp = mean(temp(:,timeslices),2);
    temp = squareform(temp(:));
    data(i,:,:) = temp;
end

save('chris_eeg_data.mat', 'data')