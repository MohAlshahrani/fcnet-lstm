function d = axial_dist_with_GT(estimate, ground_truth)
    % strip off NaNs used to maintain all groundtruth in same matrix
    ground_truth = ground_truth(~isnan(ground_truth));
    if numel(ground_truth)==4
        cygt = ground_truth(2)+ground_truth(4)/2;
    else
        [~, cygt, ~, ~] = get_axis_aligned_BB(ground_truth);
    end
    cye = estimate(2) + estimate(4)/2;
    d = (cygt - cye).^2;
    d = min(d, 50); % saturate
end