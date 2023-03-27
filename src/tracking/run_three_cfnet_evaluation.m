%% Sample execution for CFNet-conv1
% hyper-parameters reported in Supp.material for CVPR'17, Table 2 for arXiv version
tracker_par1.join.method = 'corrfilt';
tracker_par1.net = 'cfnet-conv1_e75.mat';
tracker_par1.net_gray = 'cfnet-conv1_gray_e55.mat';
tracker_par1.scaleStep = 1.0355;
tracker_par1.scalePenalty = 0.9825;
tracker_par1.scaleLR = 0.7;
tracker_par1.wInfluence = 0.2375;
tracker_par1.zLR = 0.0058;

tracker_par2.join.method = 'xcorr';
tracker_par2.net = 'baseline-conv5_e55.mat';
tracker_par2.net_gray = 'baseline-conv5_gray_e100.mat';
tracker_par2.scaleStep = 1.0470;
tracker_par2.scalePenalty = 0.9825;
tracker_par2.scaleLR = 0.68;
tracker_par2.wInfluence = 0.175;
tracker_par2.zLR = 0.0102;

[~,~,dist,overlap,~,~,~,~] = run_tracker_evaluation_new('imgs18', tracker_par2, tracker_par2, tracker_par2, true);