function out=lowpasser(in,sampling,lp,varargin)
% function out=lowpasser(in,sampling,lp,[paddy],[plotornot],[verbose]);
%
% high pass filter function
% inputs: in - your signal
%         sampling - sampling frequency (in Hz)
%         lp - your low pass frequency (in Hz)
%         paddy - number of entries with which to pad the signal to avoid
%         edge effects
%         plotornot - optional plotting output
%
% EXAMPLE
% 
% a=0.001:0.001:1;
% t=sin(a*2*pi*40)+sin(a*2*pi*4)+rand(1,1000)*4;
% out=highpasser(t,1000,10,5,1);

if nargin<4
    plotornot=[];
end
paddy=[];
plotornot=[];
verbose=[];
if length(varargin)>0;paddy=varargin{1};,end
if length(varargin)>1;plotornot=varargin{2};,end
if length(varargin)>2;verbose=varargin{3};,end

if ~isempty(verbose)
disp(['data sampled at ',num2str(sampling),' Hz']);;
disp(['low pass filter at ',num2str(lp),' Hz']);;
end


if size(in,1)==1 | size(in,2)==1;

%%%%%%%%%%%%%%%%%%%%%%%%
%%%% if a vector %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%

% PAD
if ~isempty(paddy);
    if ~isempty(verbose);
    disp(['padding with ',num2str(paddy),' scans'])
    end
    pin=pad(in,paddy,1);
    pin(1:paddy)=repmat(in(1),1,paddy);
    pin(end-paddy+1:end)=repmat(in(end),1,paddy);
    in=pin;
end

nscan=length(in);

% FFT
fin=fft(in,nscan);
ffin=fin.*conj(fin);
f = sampling*(0:nscan/2)/nscan;

% work out highpass and lowpass limits in signal space
h=find(f<lp)+1;lowpass=h(end);
if lowpass<1;lowpass=1;,end

% PLOTTING
if ~isempty(plotornot)
figure;
subplot(2,2,1);
plot(in');
title('signal in');

subplot(2,2,3);
plot(f,ffin(1:(nscan/2)+1));
title('fft of signal in');
lx=max(ffin(:))/100:max(ffin(:))/100:max(ffin(:));
ly=ones(1,100)*lp;
hold on; plot(ly,lx,'r','linewidth',3);
end

% get remove spectral components
fin(lowpass:end-lowpass)=0;

% inverse FFT

out=ifft(fin,nscan,2,'symmetric');

% UNpad
if ~isempty(paddy)
out=out(paddy+1:end-paddy);
end
fftout=fft(out).*conj(fft(out));

% PLOTTING
if ~isempty(plotornot)
subplot(2,2,2);plot(f,fftout(1:(nscan/2)+1));
title('fft of signal out');
hold on; 
plot(ly,lx,'r','linewidth',3);
subplot(2,2,4);
plot(out);
title('signal out');
end

else

%%%%%%%%%%%%%%%%%%%%%%%%
%%%% if a matrix %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%


% PAD
if ~isempty(paddy);
    disp(['padding with ',paddy,' scans'])    
    pin=pad(in,paddy,1);
    pin(:,1:paddy)=repmat(in(:,1),1,paddy);
    pin(:,end-paddy+1:end)=repmat(in(:,end),1,paddy);
    in=pin;
end

ndim=ndims(in);
nscan=size(in,ndim);

if ~isempty(verbose);
disp(['found ',num2str(nscan),' scans']);
end

% FFT
fin=fft(in,nscan,ndim);;
ffin=fin.*conj(fin);
f = sampling*(0:(nscan/2))/nscan;

% work out highpass and lowpass limits in signal space
h=find(f<lp);lowpass=h(end);
if lowpass==0;lowpass=1;,end


% PLOT
mffin=squeeze(mean(ffin))';
if ~isempty(plotornot)
figure;

subplot(2,2,1);  % SIGNAL IN
steplot(in,'l',[],{[.8 .8 .8],[0 0 0]},[],{[5] [2]});
title('signal in');

subplot(2,2,2);   % FIN
plot(f(2:end),mffin(2:(nscan/2)+1));
title('fft of signal in');
lx=min(mffin(2:end)):max((mffin(2:end))-min(mffin(2:end)))/9:max(mffin(2:end));
ly=ones(1,10)*lp;
hold on; plot(ly,lx,'r','linewidth',3);
end

%%% remove spectral components
if ndim==2;
fin(:,lowpass:end-lowpass)=0;
elseif ndim==3;
fin(:,:,lowpass:end-lowpass)=0;
elseif ndim==4;
fin(:,:,:,lowpass:end-lowpass)=0;
else
    error(['>',num2str(ndim),' dimensions not supported']);
end


% inverse FFT
out=ifft(fin,nscan,ndim,'symmetric');

 plotout=out;

% UNpad
if ~isempty(paddy);
out=out(:,paddy+1:end-paddy);
end
fftout=fft(out).*conj(fft(out));
mfout=squeeze(mean(fftout))';


% PLOTTING
if ~isempty(plotornot)
subplot(2,2,3);   %ALTERED FIN
plot(f(2:end),mfout(2:(nscan/2)+1));
title('fft of signal out');
hold on; 
lx=min(mfout(2:end)):max((mfout(2:end))-min(mfout(2:end)))/9:max(mfout(2:end));
ly=ones(1,10)*lp;
hold on; plot(ly,lx,'r','linewidth',3);
subplot(2,2,4);
if size(out,1)>30;
ex=shuffle(1:size(out,1));
plotout=plotout(ex(1:30),:);
end
steplot(plotout,'l',[],{[.8 .8 .8],[0.4 0.4 0.4]},[],{[5] [3]});
title('mean & example signal out');
hold on;
end

end




