function [img] = filter_image(img, frame_size, filter_size)

#Filters the image using a  (yet to decide, currently, averager) filter
#I'm applying linear filtering, not convolution'

averager_filter = ones(filter_size, filter_size);
averager_filter = averager_filter / (filter_size * filter_size);
#averager_filter = [0,0.125,0; 0.125,0.5,0.125; 0,0.125,0];

range = (filter_size - 1) / 2;
#for i = 1:(frame_size - filter_size + 1),
#    for j = 1:(frame_size - filter_size + 1),
for i = range+1:(frame_size - filter_size + 1),
    for j = range+1:(frame_size - filter_size + 1),
        value = 0.0;
#        for n = 1:filter_size,
#            for m = 1:filter_size,
        for n = -range:range,
            for m = -range:range,
#                value += img((i+n-1),(j+m-1)) * averager_filter(n,m);
                value += img((i+n),(j+m)) * averager_filter(n + range + 1,m + range + 1);
            end;
        end;
        img(i,j) = value;
    end;
end;
end
