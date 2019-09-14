clear;close all;
%% settings
folder = 'Test/Set5';
savepath_gt = fullfile(folder, 'gt');
savepath_2x = fullfile(folder, 'bicubic_2x');
savepath_3x = fullfile(folder, 'bicubic_3x');
savepath_4x = fullfile(folder, 'bicubic_4x');
if ~exist(savepath_gt, 'dir')
    mkdir(savepath_gt);
end
if ~exist(savepath_2x, 'dir')
    mkdir(savepath_2x);
end
if ~exist(savepath_3x, 'dir')
    mkdir(savepath_3x);
end
if ~exist(savepath_4x, 'dir')
    mkdir(savepath_4x);
end


%% generate data
filepaths = []
filepaths = [filepaths; dir(fullfile(folder, '*.bmp'))];
filepaths = [filepaths; dir(fullfile(folder, '*.jpg'))];
    
for i = 1 : length(filepaths)
    image = imread(fullfile(folder, filepaths(i).name));
    file_name = filepaths(i).name;
    
    if size(image, 3) == 3
        image_ycbcr = rgb2ycbcr(image);
    else
        % gray-scale image
        image_ycbcr = rgb2ycbcr(cat(3, image, image, image));
    end
    
    name_gt = sprintf('%s/%s', savepath_gt, file_name);
    imwrite(image_ycbcr, name_gt);
    
    [hei,wid,n_dim] = size(image);
    
    for scale = 2 : 4
        savepath = eval(sprintf('savepath_%dx', scale));
        im_label = image(1:hei-mod(hei, scale), 1:wid-mod(wid, scale), :);
    
        if size(image, 3) == 3
            image_ycbcr = rgb2ycbcr(im_label);
        else
            % gray-scale image
            image_ycbcr = rgb2ycbcr(cat(3, im_label, im_label, im_label));
        end
    
        image_ycbcr = im2double(image_ycbcr);
    
        color = image_ycbcr(:,:,2:3);
        image_y = image_ycbcr(:,:,1);
    
        [cropped_hei, cropped_wid] = size(image_y);
        
        im_input = imresize(imresize(image_y,1/scale,'bicubic'),[cropped_hei, cropped_wid],'bicubic');
        name = sprintf('%s/%s', savepath, file_name);
        imwrite(cat(3, im_input, color), name);
    
    end
end