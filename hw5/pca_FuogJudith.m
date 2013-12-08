function Eigen = pca_FuogJudith(Data, k)
%%
% m is the number of samples
% n is the dimension
[m,n] = size(Data); 
X = Data';

%% 
% Computing PCA through SVD
% SVD gives us M = U*S*V' for any matrix M, where U and V' ar orthonormal.
% Leftmultiplication with M' gives us: 
% A = M*M' = (U*S*V')(U*S*V')' = U*S*S'U', since V is orthonormal.
% A is a square, symmetric matrix.Thus we have dagonalized M*M' and we 
% finde the eigenvectors in the columns of U as well as the eigenvalues in
% S*S'.
% For PCA we need the eigenvectors of the covariance matrix. Since our
% adjusted data X has zero means our covariance matrix Sigma = X*X'/(n-1).
% Since 1/(n-1) is only a scalar finding the eigenvectors for X*X'/(n-1) is
% the same as finding the eigenvectors of X*X'.
% As we have seen before, we get the eigenvectors of X*X' by simply using
% SVD on X alone.
% Thus U contains all eigenvectors of the covariance matrix of our data.

[U, S, V] = svd(X);

%%
% All that is left to do now is to select k of the n eigenvectors.
% Here I am a little bit at loss, since I dont know how to select them.
% I know that the eigenvalues are by default of SVD in a descending order, 
% so I pick the first k eigenvectors that correspond to those eigenvalues, 
% hoping that those are the princial components.
% Edit: The results seem fine, so I guess that's the correct way to do it.

assert(k<=size(U,1)); % debug reasons
Eigen = U(:,1:k);

