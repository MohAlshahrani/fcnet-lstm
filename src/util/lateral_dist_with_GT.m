function d = lateral_dist_with_GT(estimate, ground_truth)
    % strip off NaNs used to maintain all groundtruth in same matrix
    ground_truth = ground_truth(~isnan(ground_truth));
    if numel(ground_truth)==4
        cxgt = ground_truth(1)+ground_truth(3)/2;
    else
        [cxgt, ~, ~, ~] = get_axis_aligned_BB(ground_truth);
    end
    cxe = estimate(1) + estimate(3)/2;
    d = (cxgt - cxe).^2;
    d = min(d, 50); % saturate
end
