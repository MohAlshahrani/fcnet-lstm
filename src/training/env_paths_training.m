function opts = env_paths_training(opts)
    opts.rootDataDir = '/scratch/mqa5928/CLUST2D-FCnet/crops/CLUST2D/train/'; % where the training set is
    opts.imdbVideoPath = '/scratch/mqa5928/CLUST2D-FCnet/imdb_video.mat'; % where the training set metadata are
    opts.imageStatsPath = '/scratch/mqa5928/CLUST2D-FCnet/video_stats.mat'; % where the training set stats are
end
