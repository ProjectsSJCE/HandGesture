function [img] = blacken_palm(img, frame_size)

#Blackens the palm, whitens the background and returns the image

img = imresize(img, [frame_size frame_size]);
rgb = img;
[x, map] = rgb2ind(img);
img = ind2gray (x, map);
img = img / 255.0;
[row,col] = size(img);
white = 1.0;
black = 0.0;
for i = 1:row,
    for j = 1:col,
##        if (img1(i,j) < 0.20)
        if (img(i,j) < 0.45)
            img(i,j) = white;
#            I'm doing the below step because if I make it simply 1, feature normalize would divide something by 0 and result in Nan
            white -= 0.0000000001;
            end;
##        if (img1(i,j) < 0.8 && img1(i,j) > 0.25)
        if (img(i,j) < 0.9 && img(i,j) >= 0.45)
            img(i,j) = black;
            
            black += 0.0000000001;#same reason
            end;
##            disp(rgb(i,j,1));
#            end;
#        if (img(i,j) == 0)
#            img(i,j) = black;
#            black += 0.00001;
#            end;
        if (img(i,j) >= 0.9)
            img(i,j) = white;
            white -= 0.0000000001;#same reason
            end;
##        if (img(i,j) < 0)
##            img(i,j) = 0.01;
##            end;
        end;
    end;
end
