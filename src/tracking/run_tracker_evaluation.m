% function [curve_dist, curve_overlap, expected_dist, expected_overlap, all_boxes, all_gt, mean_t, std_t] = run_tracker_evaluation(video, tracker_params, eval_flag)
function all_boxes = run_tracker_evaluation(video, tracker_params, eval_flag)

%RUN_TRACKER_EVALUATION
    % Performs an evaluation similar to the OTB-TRE by initializing the tracker at different starting point in the sequence.
    % Evaluation can be done on a single video or on all videos in the
    % specified run_params.dataset

    global OPE TRE;
    OPE = 1; TRE = 2;

    run_params.subSeq = 1; % number of restarts (20 for official OTB-TRE)
    run_params.n_th_points = 100+1; % 50 for official OTB-TRE
    run_params.log = false;
    run_params.log_prefix = '';
    run_params.stop_on_failure = false; % stop the evaluation after failure and set to zero overlap all remaining frames
    run_params.dataset = 'train';
    %run_params = vl_argparse(run_params, varargin);
    if isfield(tracker_params, 'paths')
        tracker_params.paths = env_paths_tracking(tracker_params.paths);
    else
        tracker_params.paths = env_paths_tracking();
    end

    th_points_d = linspace(0,50,run_params.n_th_points);
    th_points_o = linspace(0,1,run_params.n_th_points);
    applyToGivenRow = @(func, matrix1, matrix2) @(row) func(matrix1(row, :), matrix2(row, :));
    applyToRows = @(func, matrix1, matrix2) arrayfun(applyToGivenRow(func, matrix1, matrix2), 1:size(matrix1,1))';
    IOU_fun = @IOU_with_GT;
    dist_fun = @dist_with_GT;
    lateral_dist_fun = @lateral_dist_with_GT;
    axial_dist_fun = @axial_dist_with_GT;
    
    all_boxes = [];
    all_gt = [];
    all_type = [];
    times = [];
    switch video
        case 'all'
            % all videos, call self with each video name.
            % only keep valid directory names
            dirs = dir(fullfile(tracker_params.paths.eval_set_base, run_params.dataset));
            videos = {dirs.name};
            videos(strcmp('.', videos) | strcmp('..', videos) | ...
                strcmp('anno', videos) | ~[dirs.isdir]) = [];
            nv = numel(videos);
            fprintf('\n||| OTB-TRE evaluation for %d sequences and %d sub-seq each |||\n\n', numel(videos), run_params.subSeq);
            for k = 1:nv
                fprintf('%d/%d - %s\n', k, nv, videos{k});
                if k>1, tracker_params.init_gpu = false; end
                [all_boxes, all_gt, all_type, times] = do_OTB_TRE(videos{k}, all_boxes, all_gt, all_type, times, run_params.subSeq, tracker_params, run_params);
            end

        otherwise
            % run only once
            if ~eval_flag
                run_params.subSeq = 1;
                [all_boxes, times] = do_run_only(video, all_boxes, times, run_params.subSeq, tracker_params, run_params);
            else
                [all_boxes, all_gt, all_type, times] = do_OTB_TRE(video, all_boxes, all_gt, all_type, times, run_params.subSeq, tracker_params, run_params);
            end
    end
    
    % if flag is set then only evaluate it otherwise just display total
    % time taken
    if eval_flag
        distances = applyToRows(dist_fun, all_boxes, all_gt);
        lateral_distances = applyToRows(lateral_dist_fun, all_boxes, all_gt);
        axial_distances = applyToRows(axial_dist_fun, all_boxes, all_gt);        
        lateral_error = rmse(lateral_distances, size(all_gt, 1));
        axial_error = rmse(axial_distances, size(all_gt, 1));
        error = sqrt(lateral_error.^2 + axial_error.^2);
        ious = applyToRows(IOU_fun, all_boxes, all_gt);
        avg_iou = avgIou(ious, size(all_gt, 1));
        [curve_dist, expected_dist] = compute_score(distances, th_points_d);
        [curve_overlap, expected_overlap] = compute_score(ious, th_points_o);
        mean_t = mean(times);
        std_t = std(times);
        fprintf('Overlap: %.2f\tRMSE: %.2f\tLateral RMSE: %.2f\tAxial RMSE: %.2f\tAvg IOU: %.2f\tAvg CTPF: %.2f\n', expected_overlap, error, lateral_error, axial_error, avg_iou, mean_t);
    else
        mean_t = mean(times);
        std_t = std(times);
        fprintf('\nfps: %.1f\n', mean_t);
    end
end

function error = rmse(distances, num)
    dis = 0;
    for d = 1:size(distances, 1)
        dis = dis + distances(d);
    end
    dis = dis/num;
    dis = sqrt(dis);
    error = dis;
end

function iou = avgIou(ious, num)
    iou = 0;
    for d = 1:size(ious, 1)
        iou = iou + ious(d);
    end
    iou = iou/num;
    iou = iou * 100;
end

function [curve, auc] = compute_score(scores, th_points)
    n_scores = numel(scores);
    ge_th = bsxfun(@ge, scores, th_points);
    curve = sum(ge_th,1)/n_scores;
    auc = trapz(curve);
end

function [all_boxes, times] = do_run_only(video, all_boxes, times, subSeq, tpar, rpar)
    %the ground truth will now just contain a single starting frame
    disp('Wasih');
    [tpar.imgFiles, ground_truth] = load_video_info_votformat(fullfile(tpar.paths.eval_set_base,rpar.dataset), video);
    tpar.video = video;
    if rpar.stop_on_failure
        tpar.track_lost = @(frame, box) track_lost_func(frame, box, ground_truth);
    else
        tpar.track_lost = [];
    end
    % decide run_params.subSequences
    n_frames = numel(tpar.imgFiles);
    starts = round(linspace(1, n_frames, subSeq+1));
    starts(end) = [];
    for i = 1:subSeq
        fprintf('\t\tsub-seq %d/%d\n', i, subSeq);
        tpar.startFrame = starts(i);
        n_subframes = n_frames - tpar.startFrame + 1;
        region = ground_truth(tpar.startFrame, :);
        disp(region);
        [cx, cy] = get_axis_aligned_BB(region);
        tpar.targetPosition = [cy cx]; % centre of the bounding box
        tpar.targetSize = [50 50];
        %return;
        %run the tracker
        if i>1, tpar.init_gpu = false; end
        [new_boxes, new_speed] = tracker(tpar, ground_truth);
        all_boxes = cat(1, all_boxes, new_boxes);
        times(end+1) = new_speed;
    end
end

function [all_boxes, all_gt, all_type, times] = do_OTB_TRE(video, all_boxes, all_gt, all_type, times, subSeq, tpar, rpar)
    % we were given the name of a single video to process.
    % get image file names, initial state, and ground truth for evaluation
    global OPE TRE;
    [tpar.imgFiles, ground_truth, st_boxes] = load_video_info_votformat(fullfile(tpar.paths.eval_set_base,rpar.dataset), video);
    tpar.video = video;
    
    if rpar.stop_on_failure
        tpar.track_lost = @(frame, box) track_lost_func(frame, box, ground_truth);
    else
        tpar.track_lost = [];
    end
    % decide run_params.subSequences
    n_frames = numel(tpar.imgFiles);
    starts = round(linspace(1,n_frames, subSeq+1));
    starts(end) = [];
    for i = 1:subSeq
        fprintf('\t\tsub-seq %d/%d\n', i, subSeq);
        tpar.startFrame = starts(i);
        
        n_subframes = n_frames - tpar.startFrame + 1;
        region = ground_truth(tpar.startFrame, :);
        [cx, cy, w, h] = get_axis_aligned_BB(region);
        tpar.targetPosition = [cy cx]; % centre of the bounding box
        tpar.targetSize = [h w];
                
        %run the tracker
        if i>1, tpar.init_gpu = false; end
        [new_boxes, new_speed] = tracker(tpar, ground_truth, st_boxes);
        times(end+1) = new_speed;
        A = new_boxes(1:tpar.startFrame-1, :);
        assert(sum(A(:))==0)
        new_boxes(1:tpar.startFrame-1, :) = [];
        if i==1
            type = OPE*ones(n_subframes ,1);
        else
            type = TRE*ones(n_subframes ,1);
        end
        all_boxes = cat(1, all_boxes, new_boxes);
        this_gt =  ground_truth(tpar.startFrame:end, :);
        if rpar.log
            % log new_boxes and this_gt to file
            log_boxes(rpar.log_prefix, tpar.dataset, rpar.subSeq, video, i, new_boxes, this_gt);
        end
        if size(ground_truth,2)==4
           % Padding with NaNs used to maintain all groundtruth in same matrix
           this_gt = padarray(this_gt,[0 4],NaN,'post');
        end
        all_gt = cat(1, all_gt, this_gt);
        all_type = cat(1, all_type, type);
    end
end

function is_lost = track_lost_func(frame, box, ground_truth)
    %returns true if the track was lost (reads ground truth)
    is_lost = (IOU_with_GT(box, ground_truth(frame,:)) <= 0);
end


