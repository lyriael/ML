% Let's do it again, I think I got the hang of it now!

function Eigen = pca_FuogJudith(Data, k)
% Data = [1 2 3;4 5 6;7 8 9;6 5 2;7 7 1]; k=2; testing use only
% init, always handy to have.
% m - number of samples
% n - dimension
[m,n] = size(Data); 
X = Data'; % Example I follow uses it that way...

% Prepare data for svd. That means normalizing each 
% sample to have zero mean and unit standart deviantion.
%X -= mean(Data(:));
%X /=std(X(:)); % Not sure if I need to do that.

% Following PCA procedure, we would now calculate 
% the covariance of X, and then calculate the eigenvalues 
% and eigenvectors of Sigma. Using SVD makes it a lot easier:
% A = USV'and AA'=US²U'
% Here the columns of U contain the eigenvectors of AA' and 
% the eigenvalues of AA' are the squares of the singular values in S.

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

Eigen = U(:,1:k);


%Dims
%Data: MxN
%X: NxM
%Eigen: NxK
