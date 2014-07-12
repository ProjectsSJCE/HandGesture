function [edgeImage] = blacken_palm(img, frame_size)

#Converting the image to gray scay
rgb = img;
[x, map] = rgb2ind(img);
img = ind2gray (x, map);
[row,col] = size(img);
img = img / 255.0;

#Function is just thresholding the image, also seperates the foreground from background
threshold = 0.3;
k = [1 2 1; 0 0 0; -1 -2 -1];
H = conv2(double(img),k, 'same');
V = conv2(double(img),k','same');
E = sqrt(H.*H + V.*V);
edgeImage = uint8((E > threshold) * 255);
imshow(edgeImage);
