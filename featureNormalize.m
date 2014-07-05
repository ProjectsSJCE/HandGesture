function [img_norm, mu, sigma] = featureNormalize(img)
%FEATURENORMALIZE Normalizes the features in img 

mu = mean(img);
img_norm = bsxfun(@minus, img, mu);

sigma = std(img_norm);
sigma(sigma == 0) = 0.0000000001;
img_norm = bsxfun(@rdivide, img_norm, sigma);

% ============================================================

end
