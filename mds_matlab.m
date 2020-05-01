 clc
 close all
 
 D = mean(magndat,3);
 [Y,eigvals] = cmdscale(D,3);
 %Y = mdscale(D,3);
 numbers = [1:11, 6:16, 1:16]
 
 is = [1, 1, 2];
 js = [2, 3, 3];
 
 for ind=1:3
     i = is(ind);
     j = js(ind);
     subplot(1,3,ind);
     
     plot(Y(1:11,i), Y(1:11, j), '.b'); hold on;
     plot(Y(1:11,i), Y(1:11, j), 'b'); hold on;
     text(Y(1,i), Y(1, j), '1'); hold on;
     text(Y(11,i), Y(11, j), '11'); hold on;
     
     plot(Y(12:11+11,i), Y(12:11+11, j), '.r'); hold on;
     plot(Y(12:11+11,i), Y(12:11+11, j), 'r'); hold on;
     text(Y(12,i), Y(12, j), '6'); hold on;
     text(Y(11+11,i), Y(11+11, j),'16'); hold on;
     
     plot(Y(12+11:end,i), Y(12+11:end, j), '.g'); hold on;
     plot(Y(12+11:end,i), Y(12+11:end, j), 'g'); hold on;
     text(Y(12+11,i), Y(12+11, j), '1'); hold on;
     text(Y(end,i), Y(end, j), '16'); hold on;
     
     axis equal
 end
 
 
 