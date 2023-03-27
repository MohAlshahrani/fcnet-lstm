img1 = im2double(imread('onion.jpg'));

hbm = vision.BlockMatcher('ReferenceFrameSource',...
        'Input port','BlockSize',[35 35]);
hbm.OutputValue = 'Horizontal and vertical components in complex form';
halphablend = vision.AlphaBlender;

img2 = imtranslate(img1,[5,5]);

motion = hbm(img1,img2);