function Eigen = pca_FuogJudith(Data, k)
%%
% m is the number of samples
% n is the dimension
[m,n] = size(Data); 
X = Data';

%%
% Since the means of the dimension in the adjusted data set X are 0,
% the covariance matrix can simply be written as:
% Sigma = (XX')/(n-1)
% For this matrix Sigma, we need to find the eigenvetors. Using SVD makes
% this a lot easier. 
% SVD(M) = [U, S, V], where M = U*S*V'
% If we replace now M with X*X', we get:
% XX' = (USV')(USV')' = USV'VS'U'
% Since XX' is symetric and has zero mean, it is diagonalizable thus V is 
% orthonormal and we'll find our eigenvectors in the colums of U (and the
% eigenvalues are the squares of the singular values in S).
% XX' = USIS'U'= US²U'
[U, S, V] = svd(X);

% So we're basically done, we've got the eigenvalues which we 
% need to return, but we should only return k-principal components. 
% I would have thought, that we would return the best in descanding 
% order, but I have found no way to figure out such a thing. 
% So I'll just return the k-first principal components. Since the
% eigenvalues are descending, may be the eigenvecotrs are correstpondingly.
% But then, when I do a basistransformation in the next step, how will I know
% which dimensions will be left out from the original data set?
% *confuuuuused*

%%
% All that is left to do now is to select k of the n eigenvectors.
% Here I am a little bit at loss, since I dont know how to select them.
% I know that the eigenvalues are by default of SVD in a descending order, 
% so I pick the first k eigenvectors that correspond to those eigenvalues, 
% hoping that those are the princial components.

assert(k<=size(U,1)); % debug reasons
Eigen = U(:,1:k);

