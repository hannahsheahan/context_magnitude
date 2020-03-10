
clear all; close all; clc;
Nsamples = 1000000;
for sample_A = 1:15
    space = [1:15, 1:10, 6:15];
    space(space==sample_A)=[];  % make sure we dont sample same number twice in a row
    samples = randsample(space,Nsamples, true);
    %hist(samples,15);
    
    n = sum(samples<sample_A);
    pAbiggerB(sample_A) = n/Nsamples;
end

pAbiggerB

probA=[]
for i=1:5
    probA(i)=2/35;
end
for i=6:10
    probA(i)=3/35;
end
for i=11:15
    probA(i)=2/35;
end
probA

% policy = use global mean
pcorrect=[]
for i=1:7
   pcorrect(i) = 1-pAbiggerB(i) 
end
for i=8:15
   pcorrect(i) = pAbiggerB(i) 
end

% performance
x=0
for i=1:length(probA)
   x = x + probA(i)*pcorrect(i);
end
x



