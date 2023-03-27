function paths = env_paths_tracking(varargin)
    paths.net_base = 'data/'; % e.g. '/home/luca/cfnet/networks/';
    paths.eval_set_base = 'dataset/CLUST2D/'; % e.g. '/home/luca/cfnet/data/';
    paths.stats = 'video_stats.mat'; % e.g.'/home/luca/cfnet/data/ILSVRC2015.stats.mat';
    paths = vl_argparse(paths, varargin);
end
