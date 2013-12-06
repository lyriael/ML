function [ theta ] = logisticRegressionTrain_FUOGJUDITH( DataTrain, LabelsTrain, maxIterations )
% logisticRegressionTrain train a logistic regression classifier
% [ theta ] = logisticRegressionTrain( DataTrain, LabelsTrain, MaxIterations )
% Using the training data in DataTrain and LabelsTrain trains a logistic
% regression classifier theta. 

	% Defining sigmoid function, to be used later on.
	sigmoid = @(x)  1 ./ (1 + exp(-x));

	% Replacing -1 with 0.
	LabelsTrain = (LabelsTrain+1)./2;

	% Initializing
	dim = size(DataTrain,2);		% Number of params
	m = size(DataTrain,1);			% Number of datasets
	
	theta = zeros(dim,1); 		

	% Iteration loop
	for k=1:maxIterations
		grad = zeros(dim,1);		% Initializing gradient
		hesse = zeros(dim,dim);		% Initializing hessian
		
		% Sum loop
		for i=1:m
			% Initializing
			yi = LabelsTrain(i); 
			xi = DataTrain(i,:).';
			g = sigmoid(theta.'*xi);
		
			% Gradient
			grad = grad + (yi - g)*xi;
		
			% Hessian
			hesse = hesse + (g*(1-g))*xi*xi.';
		end
		
		grad = (1/m)*grad;
		hesse = (-1/m)*hesse;
	
		theta = theta - inv(hesse)*grad;
	end
end




