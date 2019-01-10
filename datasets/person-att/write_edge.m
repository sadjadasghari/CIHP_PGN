clear;
close all;
fclose all;
%%
imglist = 'img_list.txt';
list = textread(imglist, '%s');
output_root = 'edges';
vis_output_root = 'edges_vis';

human_inst_root = fullfile('output_2');



for i = 1:length(list);
    imname = list{i};
    sz = size(imname);
    len = sz(2);
    if imname(len-9:len) == "-label.jpg";
        
        instance_map = imread(fullfile(human_inst_root, imname)); % [imname '.jpg']));
        instance_contour = uint8(imgradient(instance_map) > 0);


        imwrite(instance_contour, fullfile(output_root, [imname '.png']));
        imwrite(instance_contour*255, fullfile(vis_output_root, [imname '.png']));
    %     imshow(instance_contour);

    end
end    
