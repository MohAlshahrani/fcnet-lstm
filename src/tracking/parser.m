%gr = readmatrix('D:\CSE600\Project1\cfnet-master\data\validation\imgs24\groundtruth.csv');
%disp(gr);
%writematrix(gr, 'D:\CSE600\Project1\cfnet-master\data\validation\imgs24\groundtruth.txt');
img_files = dir(['D:\CSE600\Project1\cfnet-master\data\validation\Data\' '*.png']);
img_files = sort({img_files.name});
for f = 1:length(img_files)
    %disp(img_files{f});
    [part, fn, ext] = fileparts(img_files{f});
    %disp(part);
    fname = strcat('D:\CSE600\Project1\cfnet-master\data\validation\Data\', img_files{f});
    img = imread(fname);
    newfname = strcat('D:\CSE600\Project1\cfnet-master\data\validation\Data\', fn, '.jpg');
    imwrite(img, newfname);
    %img = imread(fname);
    
end