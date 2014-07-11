function [img] = blacken_palm(img, frame_size)


#Function is just thresholding the image, also seperates the foreground from background
###
#Converting the image to gray scay
rgb = img;
[x, map] = rgb2ind(img);
img = ind2gray (x, map);
[row,col] = size(img);
img = img / 255.0;

#Pass the image through a low pass filter, this is done to make the size reduction more smooth, thus retaining a lot of info
#low_pass = zeros(3, 3) .+ (1 / 9);
#im1 = img;
#im1 = imfilter(im1, low_pass, "replicate");
#img = im1;

#Reduce the size of the image
img = imresize(img, [frame_size frame_size]);

#Threshold the image, such that the hand is black and the background is white
#white is 1.0 and black is 0.0

img(img < 0.4 | img >= 0.9) = 1.0;
img(img < 0.9 & img >= 0.4) = 0.0;

#What you are doing now is, you took the colour of blue, then suppressed blue to white and remaining to black right? Hence its limited to blue background.
#What if we take the gray scale value for skin colour and then make all the values in the range of skin colour to be black, remaining as white. Times when this would fail is if he is wearing gloves or if the background is very much like skin colour, obviously the chances of this happening is lower than previous.
#I could not find the gray scale value for skin colour

end
