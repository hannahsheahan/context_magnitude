%% between vs within distance stuff 
% kind of cobbled, but hopefully understandable
clear
close all

%some parameters
n_rand_iter = 10000;
rdm_dx = 1;
lineord = [1:15, 1:10, 6:15];
cvec = [ones(1,15), 2*ones(1,10), 3*ones(1,10)];
cvec_iter = [1,2; 1,3; 2,3];

load('trueConlab_activations.mat')
rdm{rdm_dx} = squareform(pdist(f));
if numel(rdm{1}) ~= 2*numel(unique(rdm{1}(rdm{1}~=0)))+numel(diag(rdm{1}))
    error
end

for cdxx = 1:3 %iterate through context combinations
    cdx = cvec_iter(cdxx,1);
    cdx2 = cvec_iter(cdxx,2);

    temp = rdm{rdm_dx}; temp = temp(cvec == cdx, cvec == cdx2);
    within_pos_dist(rdm_dx, cdxx) = nanmean(diag(temp));
    
    temp = rdm{rdm_dx}; temp = temp(cvec == cdxx, cvec == cdxx);
    clear tmp
    for jj = 1:size(temp,1)-1
        tmp(jj) = temp(jj, jj+1); % take off diagonal
    end
    nbr_dist(rdm_dx, cdxx) = nanmean(tmp);
    clear tmp

    % now compute randomised distributions
    for ridx = 1:n_rand_iter
        cvec2 = Shuffle(cvec);
        
        clear data
        temp = rdm{rdm_dx}; temp = temp(cvec2 == cdx, cvec2 == cdx2);
        r_within_pos_dist(rdm_dx, cdxx,ridx) = nanmean(diag(temp));
        
        clear tmp
        temp = rdm{rdm_dx}; temp = temp(cvec2 == cdxx, cvec2 == cdxx);
        for jj = 1:size(temp,1)-1
            tmp(jj) = temp(jj, jj+1); % take off diagonal
        end
        r_nbr_dist(rdm_dx, cdxx,ridx) = nanmean(tmp);
        clear tmp
        
    end
end
%% plot aand stuff


rand_ratio_pos(:,1,:) = nanmean(r_within_pos_dist(:,[1,2],:),2)./r_nbr_dist(:,[1],:);
rand_ratio_pos(:,2,:) = nanmean(r_within_pos_dist(:,[1,3],:),2)./r_nbr_dist(:,[2],:);
rand_ratio_pos(:,3,:) = nanmean(r_within_pos_dist(:,[2,3],:),2)./r_nbr_dist(:,[3],:);
rand_ratio_pos_all = squeeze(nanmean(rand_ratio_pos,2));

ratio_pos(:,1,:) = nanmean(within_pos_dist(:,[1,2],:),2)./nbr_dist(:,[1],:);
ratio_pos(:,2,:) = nanmean(within_pos_dist(:,[1,3],:),2)./nbr_dist(:,[2],:);
ratio_pos(:,3,:) = nanmean(within_pos_dist(:,[2,3],:),2)./nbr_dist(:,[3],:);
ratio_pos_all = squeeze(nanmean(ratio_pos,2));

% calculate p value
pv = reshape(repmat(ratio_pos, [1,1, size(rand_ratio_pos,3)])>rand_ratio_pos, [3, 3*n_rand_iter]);
pv = 1 - sum(pv.')./(3*n_rand_iter);
%plot
figure(1), hold on,
subplot(3,1,1), histogram(squeeze(rand_ratio_pos(1,:,:)).', 'BinWidth', 0.05, 'EdgeColor','none')
hold on, plot(ratio_pos_all(1)*ones(1,10), linspace(0,n_rand_iter/10,10), 'k', 'LineWidth', 2)
hold on, title(['RNN, pv = ', num2str(round(pv(1),4))])
hold on, plot(ones(1,10), linspace(0,n_rand_iter/10,10), 'k--', 'LineWidth', 0.75)
xlim([0.5 3.6])
legend({'shuffled', 'observed', 'btwn=within'})
set(gca, 'FontSize', 12)
xlabel('btwn/win pos dist ratio')

