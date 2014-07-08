function [img] = crop_hand(img, frame_size)
xmin = frame_size;
xmax = 0;
ymin = frame_size;
ymax = 0;
    for i = 1:frame_size,
        for j = 1:frame_size,
            if (img(i,j) == 0)
                if (i < xmin)
                    xmin = i;
                    end;
                if (j < ymin)
                    ymin = j;
                    end;
                if (i > xmax),
                    xmax = i;
                    end;
                if (j > ymax)
                    ymax = j;
                    end;
            end;
        end;
    end;
img = img(xmin:xmax, ymin:ymax);
img = imresize(img, [frame_size frame_size]);
end;
