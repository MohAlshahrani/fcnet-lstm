% Sample execution for Baseline-conv5 (improved Siam-FC)
% hyper-parameters reported in Supp.material for CVPR'17, Table 2 for arXiv version
tracker_par.join.method = 'xcorr';
tracker_par.net = 'net-epoch-2.mat';
% tracker_par.net_gray = 'baseline-conv5_gray_e100.mat';
% tracker_par.net_gray = 'net-epoch-100.mat';
tracker_par.scaleStep = 1.0470; 
tracker_par.scalePenalty = 0.9825;
tracker_par.scaleLR = 0.68;
tracker_par.wInfluence = 0.175; 
tracker_par.zLR = 0.0102;
img_folder = 'ETH-01-1';
[~,~,dist,overlap, ~,~,~,~] = run_tracker_evaluation(img_folder, tracker_par, false);