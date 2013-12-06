% I have some problems with this task and I needed to google a lot around. To save time, I'll add the most important URL which helped me. My main guidline was:
% http://math.stackexchange.com/questions/3869/what-is-the-intuitive-relationship-between-svd-and-pca

function Eigen = pca_FuogJudith(Data, k)

[m,n] size(Data);

% Normalize data, so that it has zero mean
% http://stackoverflow.com/questions/8717139/zero-mean-and-unit-variance-of-a-signal
X = Data - mean(Data(:)); % this are the first to stepts in the previous task

% using svd. Since the data has zero means, V should be an orthogonal matrix. But it seems that there are numerical problems and it doesn't quiet work out. But since the mistake is pretty small, I just ignore it. 

[U,S,V] = svd(X);

% Now we got X = U*S*V', so if we have 
% X*X' = (USV')(USV')' = (USV')(VS'U') = USV'VS'U' = US²U'
% because V is orthogonal and S is a diagonal matrix.

% Here starts my biggest confusion: what is Sigma?
% Sigma = (X*X')/m = USU'/m

Sigma = (U*S*S'*U')/m;

Eigen = eye(size(Data,2));


