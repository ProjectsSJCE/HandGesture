function [U, S] = pca(img, m, n)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

U = zeros(n);
S = zeros(n);

sigma = (img' * img) / m;
[U, S, V] = svd(sigma);
end
