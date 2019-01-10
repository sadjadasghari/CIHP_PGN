clear;
clc;
%%
datadir = 'output_2/';
outputdir = 'labels_rev/';
imglist = 'img_list.txt';
list = textread(imglist, '%s');
for i = 1:length(list);
    imname = list{i};
    sz = size(imname);
    len = sz(2);
    if imname(len-9:len) == "-label.jpg";
        
        % fprintf('img: %d\n', i);
        img_n = [datadir,list{i}];
        im = imread(img_n);
        [row, col] = size(im);
        rev_im = fliplr(im);
        for h = 1:row
            for w = 1:col
                if rev_im(h,w) == 14
                    rev_im(h,w) = 15;
                elseif rev_im(h,w) == 15
                    rev_im(h,w) = 14;
                elseif rev_im(h,w) == 16
                    rev_im(h,w) = 17;
                elseif rev_im(h,w) == 17
                    rev_im(h,w) = 16;
                elseif rev_im(h,w) == 18
                    rev_im(h,w) = 19;
                elseif rev_im(h,w) == 19
                    rev_im(h,w) = 18;
                end
            end
        end

        imwrite(rev_im, [outputdir,list{i},'.png']);
    end
end
