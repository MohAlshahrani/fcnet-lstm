function [bboxes, speed] = tracker(varargin, gt, prev_boxes)
    %% default hyper-params for SiamFC tracker.
    p.join.method = 'corrfilt';
    p.net = 'cfnet-conv2_e80.mat';
    p.net_gray = 'cfnet-conv2_gray_e40.mat';
    p.numScale = 3;
    p.scaleStep = 1.0575;
    p.scalePenalty = 0.9780;
    p.scaleLR = 0.52;
    p.responseUp = 8;
    p.wInfluence = 0.2625; % influence of cosine window for displacement penalty
    p.minSFactor = 0.2;
    p.maxSFactor = 5;
    p.zLR = 0.005; % update rate of the exemplar for the rolling avg (use very low values <0.015)
    p.video = '';
    p.visualization = true;
    p.gpus = [];
    p.track_lost = [];
    p.startFrame = 1;
    p.fout = -1;
    p.imgFiles = [];
    p.targetPosition = [];
    p.targetSize = [];
    p.track_lost = [];
    p.ground_truth = [];

    %% params from the network architecture params (TODO: should be inferred from the saved network)
    % they have to be consistent with the training
    p.scoreSize = 33;
    p.totalStride = 4;
    p.contextAmount = 0.5; % context amount for the exemplar
    p.subMean = false;
    % prefix and ids
    p.prefix_z = 'br1_'; % used to identify the layers of the exemplar
    p.prefix_x = 'br2_'; % used to identify the layers of the instance
    p.id_score = 'score';
    p.trim_z_branch = {'br1_'};
    p.trim_x_branch = {'br2_','join_xcorr','fin_adjust'};
    p.init_gpu = false;
    % Get environment-specific default paths.
    p.paths = struct();
    p.paths = env_paths_tracking(p.paths);
    p = vl_argparse(p, varargin);
    
    % network surgeries depend on the architecture    
    switch p.join.method
        case 'xcorr'
            p.trim_x_branch = {'br2_','join_xcorr','fin_'};
            p.trim_z_branch = {'br1_'};
			p.exemplarSize = 127;
			p.instanceSize = 255;
        case 'corrfilt'
            p.trim_x_branch = {'br2_','join_xcorr','fin_adjust'};
            p.trim_z_branch = {'br1_','join_cf','join_crop_z'};
			p.exemplarSize = 255;
			p.instanceSize = 255;
        otherwise
            error('network type unspecified');
    end

    % Load ImageNet Video statistics
    stats = load(p.paths.stats);
    im = single(p.imgFiles{p.startFrame});
    if(size(im, 3)==1)
        if ~isempty(p.net_gray)
                p.net = p.net_gray;
                p.grayscale = true;
        end
    end

    %% Load pre-trained network
    if ischar(p.net)
        % network has been passed as string
        net_path = [p.paths.net_base p.net];
        net_z = load(net_path,'net');
    else
        % network has been passed as object
        net_z = p.net;
    end
        
    net_x = net_z;
    % Load a second copy of the network for the second branch
    net_z = dagnn.DagNN.loadobj(net_z.net);
    
    % Sanity check
    switch p.join.method
        case 'xcorr'
            assert(~find_layers_from_prefix(net_z, 'join_cf'), 'Check your join.method');
        case 'corrfilt'
            assert(find_layers_from_prefix(net_z, 'join_cf'), 'Check your join.method');
    end
    % create a full copy of the network, not just of the handle
    net_x = dagnn.DagNN.loadobj(net_x.net);

    net_z = init_net(net_z, p.gpus, p.init_gpu);
    net_x = init_net(net_x, p.gpus, false);
    % visualize net before trimming
    % display_net(net_z, {'exemplar', [255 255 3 8], 'instance', [255 255 3 8]}, 'net_full')

    nImgs = numel(p.imgFiles);

    %% Divide the net in 2
    % exemplar branch (only once per video) computes features for the target
    for i=1:numel(p.trim_x_branch)
        remove_layers_from_prefix(net_z, p.trim_x_branch{i});
    end
    % display_net(net_z, {'exemplar', [255 255 3 8]}, 'z_net')
    % display_net(net_z, {'exemplar', [127 127 3 8], 'target', [6 6 1 8]}, 'z_net')
    % instance branch computes features for search region and cross-correlates with z features
    for i=1:numel(p.trim_z_branch)
        remove_layers_from_prefix(net_x, p.trim_z_branch{i});
    end
    % display_net(net_x, {'instance', [255 255 3 8], 'br1_out', [30 30 32 8]}, 'x_net')
    % display_net(net_x, {'instance', [255 255 3 8], 'join_tmpl_cropped', [17 17 32 8]}, 'x_net')

    z_out_id = net_z.getOutputs();
    %%
    if ~isempty(p.gpus)
        im = gpuArray(im);
    end
    % if grayscale repeat one channel to match filters size
    if(size(im, 3)==1)
        im = repmat(im, [1 1 3]);
    end
    % Init visualization
    videoPlayer = [];
    if p.visualization && isToolboxAvailable('Computer Vision System Toolbox')
        videoPlayer = vision.VideoPlayer('Position', [100 100 [size(im,2), size(im,1)]+30]);
    end
    
    avgChans = gather([mean(mean(im(:,:,1))) mean(mean(im(:,:,2))) mean(mean(im(:,:,3)))]);

    wc_z = p.targetSize(2) + p.contextAmount*sum(p.targetSize);
    hc_z = p.targetSize(1) + p.contextAmount*sum(p.targetSize);
    s_z = sqrt(wc_z*hc_z);
    s_x = p.instanceSize/p.exemplarSize * s_z;
    scales = (p.scaleStep .^ ((ceil(p.numScale/2)-p.numScale) : floor(p.numScale/2)));
    
    scaledExemplar = s_z .* scales;
    % initialize the exemplar
    [z_crop, ~] = make_scale_pyramid(im, p.targetPosition, scaledExemplar, p.exemplarSize, avgChans, stats, p);
    z_crop = z_crop(:,:,:,ceil(p.numScale/2));

    if p.subMean
        z_crop = bsxfun(@minus, z_crop, reshape(stats.z.rgbMean, [1 1 3]));
    end

    net_z.eval({'exemplar', z_crop});
    get_vars = @(net, ids) cellfun(@(id) net.getVar(id).value, ids, 'UniformOutput', false);
    z_out_val = get_vars(net_z, z_out_id);
    
    min_s_x = p.minSFactor*s_x;
    max_s_x = p.maxSFactor*s_x;
    min_s_z = p.minSFactor*s_z;
    max_s_z = p.maxSFactor*s_z;

    bboxes = zeros(nImgs, 4);
    tot_eval_time = 0;
    tot_z_time = 0;
    
    switch p.join.method
        case 'corrfilt'
            p.id_score = 'join_out';
            % Extract scores for join_out (pre fin_adjust to have them in the range 0-1)
            net_x.vars(end-1).precious = true;
    end
    
    % windowing to penalize large displacements
    window = single(hann(p.scoreSize*p.responseUp) * hann(p.scoreSize*p.responseUp)');
    window = window / sum(window(:));

    %% Tracker main loop
    scoreId = net_x.getVarIndex(p.id_score);
    overall_tic = tic;
  
    %set camera motion parameters
    alpha = 0.85;
    prev_motion = 0;
    beta = 0.4;
    optical = opticalFlowHS;
    %zp = p.targetPosition;
    num = 1;
    sizer = size(single(p.imgFiles{p.startFrame}));
    %disp(p.totalStride);
%     vid = VideoWriter('myFile.avi');
%     vid.FrameRate = 3;
%     open(vid);
    for i = p.startFrame:nImgs
        fprintf('Frame%d\n', i);
        singlePosition = [p.targetPosition([2,1]) - p.targetSize([2,1])/2, p.targetSize([2,1])];
            
        if i > p.startFrame        
            im = single(p.imgFiles{i});
            if ~isempty(p.gpus)
                im = gpuArray(im);
            end
   			% if grayscale repeat one channel to match filters size
    		if(size(im, 3)==1), im = repmat(im, [1 1 3]); end
            %im = imbilatfilt(im, 100, 7);
            prev_im = p.imgFiles{i - 1};
            %get the previous frame and the camera motion vector
            %back_motion = background_harris(prev_im, im, prev_motion, alpha, i);
            %back_motion = background_optic(im, sizer, prev_motion, alpha, optical);
            %back_motion = estimate_background_motion(prev_im, im, prevAvg, alpha);
            %disp(prevAvg);
            %we now need to shift the search position
%             if abs(angle(prev_motion) - angle(back_motion)) > 1.57
%                 beta = 14 * beta;
%             end
            %prev_motion = back_motion;
%             disp('beta');
%             %disp(beta);
            scaledInstance = s_x .* scales;
            % update instance with crop at new frame and previous position
            %p.targetPosition(1) = p.targetPosition(1) + beta * real(back_motion);
            %p.targetPosition(2) = p.targetPosition(2) + beta * imag(back_motion);

            [x_crops, pad_masks_x] = make_scale_pyramid(im, p.targetPosition, scaledInstance, p.instanceSize, avgChans, stats, p);
            copy = @(v, n) cellfun(@(x) repmat(x, [1 1 1 n]), v, 'UniformOutput', false);
            z_out = interleave(z_out_id, copy(z_out_val, p.numScale));
            [newTargetPosition, newScale] = tracker_step(net_x, s_x, s_z, scoreId, z_out, x_crops, pad_masks_x, p.targetPosition, window, p, p.numScale, true);
            s_x = max(min_s_x, min(max_s_x, (1-p.scaleLR)*s_x + p.scaleLR*scaledInstance(newScale)));
            p.targetPosition = gather(newTargetPosition);
            %update target size
            scaledTarget = [p.targetSize(1) .* scales; p.targetSize(2) .* scales];
            p.targetSize = (1-p.scaleLR)*p.targetSize + p.scaleLR*[scaledTarget(1,newScale) scaledTarget(2,newScale)];
            if p.zLR > 0
                scaledExemplar = s_z .* scales;
                [z_crop, ~] = make_scale_pyramid(im, p.targetPosition, scaledExemplar, p.exemplarSize, avgChans, stats, p);
                z_crop = z_crop(:,:,:,ceil(p.numScale/2));
                if p.subMean,   z_crop = bsxfun(@minus, z_crop, reshape(stats.z.rgbMean, [1 1 3])); end
                eval_z_tic = tic;
                net_z.eval({'exemplar', z_crop});
                eval_z_time = toc(eval_z_tic);
                tot_z_time = tot_z_time+eval_z_time;
                z_out_val_new = get_vars(net_z, z_out_id);
                
                % template update with rolling average                                                
                update = @(curr, next) (1-p.zLR) * curr + p.zLR * next;
                z_out_val = arrayfun(@(i) update(z_out_val{i}, z_out_val_new{i}), ...
                             1:numel(z_out_id), 'UniformOutput', false);
                                                                  
                s_z = max(min_s_z, min(max_s_z, (1-p.scaleLR)*s_z + p.scaleLR*scaledExemplar(newScale)));
            end
            
            %do refined prediction again (sequential cascade)
            %p.targetPosition(1) = p.targetPosition(1) + beta * imag(back_motion);
            %p.targetPosition(2) = p.targetPosition(2) + beta * real(back_motion);

            %singlePosition = [p.targetPosition([2,1]) - p.targetSize([2,1])/2, p.targetSize([2,1])];
            
%             [x_crops, pad_masks_x] = make_scale_pyramid(im, p.targetPosition, scaledInstance, p.instanceSize, avgChans, stats, p);
%             copy = @(v, n) cellfun(@(x) repmat(x, [1 1 1 n]), v, 'UniformOutput', false);
%             z_out = interleave(z_out_id, copy(z_out_val, p.numScale));
%             [newTargetPosition, newScale] = tracker_step(net_x, s_x, s_z, scoreId, z_out, x_crops, pad_masks_x, p.targetPosition, window, p, p.numScale, true);
%             %s_x = max(min_s_x, min(max_s_x, (1-p.scaleLR)*s_x + p.scaleLR*scaledInstance(newScale)));
%             
%             % update target position
%             p.targetPosition = gather(newTargetPosition);
%             %disp(p.targetPosition);
%             %update target position based on background motion
           %p.targetPosition(1) = p.targetPosition(1) + beta * real(back_motion);
           %p.targetPosition(2) = p.targetPosition(2) + beta * imag(back_motion);
           num = num + 1;
           %alpha = alpha * 1/sqrt(num);
            %alpha = alpha * gamma;
           %beta = beta/sqrt(num);
            %disp(p.targetPosition);
            % update the exemplar with crop at new frame and new position
%             if p.zLR > 0
%                 scaledExemplar = s_z .* scales;
%                 [z_crop, ~] = make_scale_pyramid(im, p.targetPosition, scaledExemplar, p.exemplarSize, avgChans, stats, p);
%                 z_crop = z_crop(:,:,:,ceil(p.numScale/2));
%                 if p.subMean,   z_crop = bsxfun(@minus, z_crop, reshape(stats.z.rgbMean, [1 1 3])); end
%                 eval_z_tic = tic;
%                 net_z.eval({'exemplar', z_crop});
%                 eval_z_time = toc(eval_z_tic);
%                 tot_z_time = tot_z_time+eval_z_time;
%                 z_out_val_new = get_vars(net_z, z_out_id);
%                 
%                 % template update with rolling average                                                
%                 update = @(curr, next) (1-p.zLR) * curr + p.zLR * next;
%                 z_out_val = arrayfun(@(i) update(z_out_val{i}, z_out_val_new{i}), ...
%                              1:numel(z_out_id), 'UniformOutput', false);
%                                                                   
%                 s_z = max(min_s_z, min(max_s_z, (1-p.scaleLR)*s_z + p.scaleLR*scaledExemplar(newScale)));
%             end

            % update target bbox
            %scaledTarget = [p.targetSize(1) .* scales; p.targetSize(2) .* scales];
            %p.targetSize = (1-p.scaleLR)*p.targetSize + p.scaleLR*[scaledTarget(1,newScale) scaledTarget(2,newScale)];
%         else
            % at the first frame output position and size passed as input (ground truth)
%         end

        rectPosition = [p.targetPosition([2,1]) - p.targetSize([2,1])/2, p.targetSize([2,1])];
        %% output bbox in the original frame coordinates
        %ground_position = [gt([2, 1] - gt
        
        oTargetPosition = p.targetPosition;
        oTargetSize = p.targetSize;
        bboxes(i, :) = [oTargetPosition([2,1]) - oTargetSize([2,1])/2, oTargetSize([2,1])];
%         grou = gt(i, :);
%         statis = gt(1, :);
%         singlePosition = prev_boxes(i, :);
%         if p.visualization
%             if isempty(videoPlayer)
                %if i >=7 && i <= 10
%                 disp(rectPosition);
%                 disp(singlePosition);
%                 h = figure(1);
                %set(h, 'units','normalized','outerposition',[0 0 1 1]);
%                 haxes = axes(h); 
%                 imshow(im/255, 'Parent', haxes);
%                 xlabel(haxes, 'Lateral Direction');
%                 ylabel(haxes, 'Axial Direction');
%         p        text(statis(1) , statis(2) + 105, 'GT', 'Color', 'g', 'FontSize', 12);
%                 rectangle('Position', grou, 'LineWidth', 1, 'EdgeColor', 'g');
%                 text(statis(1) , statis(2) + 135, 'ST', 'Color', 'r', 'FontSize', 12);
%                 rectangle('Position', singlePosition, 'LineWidth', 1, 'EdgeColor', 'r');
%                 text(statis(1) , statis(2) + 165, 'CFNet-5-Conv', 'Color', 'b', 'FontSize', 12);
%                 rectangle('Position', rectPosition, 'LineWidth', 1, 'EdgeColor', 'b');
%                 if i >= 2
%                     frame_vid = getframe(gcf);
%                     disp(size(frame_vid));
%                     writeVideo(vid, frame_vid);
%                 end
%                 if i == 1
%                 pause (4);
%                 end
%                 if i == 7
%                     %figure(i), rectangle('Position', [60.68 74.612 61.867 61.867], 'LineWidth', 2, 'EdgeColor', 'g');
%                     %figure(i), rectangle('Position', [111.2632 67.6538 64.0147 64.0147], 'LineWidth', 2, 'EdgeColor', 'b');
%                     figure(i), rectangle('Position', [105.8129 71.8758 61.9176 61.9176], 'LineWidth', 1, 'EdgeColor', 'b');
%                     %figure(i), rectangle('Position', [106.77 73.308 60 60], 'LineWidth', 1, 'EdgeColor', 'y');
%                 elseif i == 8
%                     %figure(i), rectangle('Position', [62.139 74.612 61.867 61.867], 'LineWidth', 2, 'EdgeColor', 'g');
%                     %figure(i), rectangle('Position', [134.6705 65.8585 62.0606 62.0606], 'LineWidth', 2, 'EdgeColor', 'b');
%                     figure(i), rectangle('Position', [102.1262 71.6020 60.0275 60.0275], 'LineWidth', 1, 'EdgeColor', 'b');
%                     %figure(i), rectangle('Position', [103.93 72.363 60 60], 'LineWidth', 1, 'EdgeColor', 'y');
%                 elseif i == 9
%                     %figure(i), rectangle('Position', [66.029 74.612 61.867 61.867], 'LineWidth', 2, 'EdgeColor', 'g');
%                     %figure(i), rectangle('Position', [106.5722 65.6142 62.0606 62.0606], 'LineWidth', 2, 'EdgeColor', 'b');
%                     figure(i), rectangle('Position', [96.2041 70.4065 61.9460 61.9460], 'LineWidth', 1, 'EdgeColor', 'b');
%                     %figure(i), rectangle('Position', [97.79 71.89 60 60], 'LineWidth', 1, 'EdgeColor', 'y');
%                 else
%                     %figure(i), rectangle('Position', [67.488 74.612 61.867 61.867], 'LineWidth', 2, 'EdgeColor', 'g');
%                     %figure(i), rectangle('Position', [95.5772 65.8585 62.0606 62.0606], 'LineWidth', 2, 'EdgeColor', 'b');
%                     figure(i), rectangle('Position', [86.1749 68.1814 60.0551 60.0551], 'LineWidth', 1, 'EdgeColor', 'b');
%                     %figure(i), rectangle('Position', [90.229 70.473 60 60], 'LineWidth', 1, 'EdgeColor', 'y');
% %                 else
% %                     %figure(i), rectangle('Position', [69.433 74.612 61.867 61.867], 'LineWidth', 2, 'EdgeColor', 'g');
% %                     %figure(i), rectangle('Position', [84.0936 66.1028 62.0606 62.0606], 'LineWidth', 2, 'EdgeColor', 'b');
% %                     figure(i), rectangle('Position', [86.406 67.273 70.501 70.501], 'LineWidth', 2, 'EdgeColor', 'g');
%                 end
%                 hold on
%                 img1 = im2double(rgb2gray(prev_im/255));
%                 img2 = im2double(rgb2gray(im/255));
%                 [X,Y] = meshgrid(1:91:size(img1,2),1:91:size(img1,1));
%                 disp(size(X));
%                 disp(size(Y));
% 
%                 avgAll = zeros(size(X, 1), size(Y, 1));
% 
%                 for i = 1:size(X,1)
%                     for j = 1:size(Y,1)
%                         avgAll(i,j) = back_motion;
%                     end
%                 end
% 
% 
%                 quiver(X(:),Y(:),real(avgAll(:)),imag(avgAll(:)),'color',[0 1 1])
%                 hold off

                %end
                %disp(grou);
%                 drawnow                
%                 fprintf('Frame %d\n', p.startFrame+i);
%             else
%                 im = gather(im)/255;
%                 im = insertShape(im, 'Rectangle', rectPosition, 'LineWidth', 4, 'Color', 'yellow');
                % Display the annotated video frame using the video player object.
%                 step(videoPlayer, im);
%             end
    end

        %stop the tracker on track loss (if a 'track_lost' function is specified)
        if ~isempty(p.track_lost) && p.track_lost(i, bboxes(i,:))
            break
        end
    end
    overall_time = toc(overall_tic);
    disp(overall_time);
    n_frames_ontrack = sum(sum(bboxes==0,2)~=4);
    if isempty(p.track_lost)
        speed = 1/((nImgs-p.startFrame+1) / overall_time);
    else
        speed = n_frames_ontrack/overall_time;
    end
%     close(vid);
end
