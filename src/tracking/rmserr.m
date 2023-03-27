function rmse_ = rmserr(tracking_file,ground_truth_file)
%         tracking_file = 'CIL-02_coordinates';
%         ground_truth_file = 'CIL-02_1.txt';
        ld_tracking_file = load(tracking_file);
        boxes = ld_tracking_file.all_boxes; %names must be consistent
        tracking_res = boxes(:,1:2) ;%tracking results gives top left corner
        tracking_res = tracking_res + (boxes(:,3)/2);
        ground_truth = readtable(ground_truth_file);
        ground_truth = [ground_truth.Var2 ground_truth.Var3];
        num_of_frames = length(ground_truth);
        tracking_res = [tracking_res(:,1) tracking_res(:,2)];
        x_err = (ground_truth(:,1)-tracking_res(:,1)).^2;
        y_err = (ground_truth(:,2)-tracking_res(:,2)).^2;
        err = x_err + y_err;
        sqr_err = sqrt(err);
        rmse_ = (sum(sqr_err))/num_of_frames; 
     end
    
    