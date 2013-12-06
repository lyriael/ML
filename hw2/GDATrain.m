function [phi, mu0, mu1, Sigma] = GDATrain( DataTrain, LabelsTrain )

	dim = size(DataTrain,2);	% Dimension - number of features
	m = size(DataTrain,1);		% Get number of test data

	x = DataTrain;				
	y = (LabelsTrain+1)./2;		% Replacing -1 with 0.

	%%
	% Calculating phi is rather simple: The indicator function is 1 if
	% y_i is 1, and else it's 0. So we actually count how many 1s there 
	% are in y. Since we will use this, and the indicator function where
	% y_i is 0 again, I will store these values in some extra variables.
	% (I'm not greedy at all when it comes to variables.)

	p = sum(y);		% 1{y_i = 1}
	q = m-p;		% 1{y_i = 0}

	phi = p/m;		% phi

	%%
	% For the denominator of mu0 and mu1 the sum can be exchanged with a
	% Matrix multiplication. The indicatore function of the denominator
	% for mu0 and mu1 tells us which rows of x we'll have to add together and 
	% which one to delete. We can get the same result with a matrix 
	% multiplication.

	mu1 = ((y*x)/p); 		% (Sum of 1{y_i=1})*x_i / 1{y_i = 1} 

	%%
	% For mu0 we have temporarely swap the 0s and 1s in y to make the same
	% trick work.

	mu0 = ((~y*x)/q);	% (Sum of 1{y_i=0})*x_i / 1{y_i = 0}

	%%
	% In another effort to avoid iterations, I wil make a mu-matrix that can be
	% directly substracted from x.
	% Exp. 	y = [1 0 1], mu0 = [2 2], mu1 = [3 3]
	% 		then this mu-matrix would be:
	%		Mu = [3 3;0 0;3 3]+[0 0;2 2;0 0] = [3 3;2 2;3 3]

	Mu1 = repmat(y,dim,1).' .* repmat(mu1,m,1);
	Mu0 = repmat(~y,dim,1).' .* repmat(mu0,m,1);
	Mu = Mu1+Mu0;

	%%
	% I only later realized that I mixed up the dimensions, so I had to 
	% transpose the two mu's as well as adapt Sigma to it.

	Sigma = ((x-Mu).'*(x-Mu))/m;	% (Sum of (x_i - mu_yi)(x_i - mu_yi).')/m

	mu0 = mu0.';
	mu1 = mu1.';
end

