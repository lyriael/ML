function [NormData] = normalizeData_FuogJudith(Data)
% init
%features = size(Data,2); %feature dimension
%examples = size(Data,1); %number of examples
%X = Data;

% normalize mean
%mu = mean(X,1); %mean of all rows
%X = X - repmat(mu,examples,1); %works faster than broadcast?
% simpler: X = X - mean(X(:));


% normalize scale
%sigma = diag(X'*X)'/examples; %thanks for the tip :)
%X = X./repmat(sigma,examples,1);


[m,n] = size(Data); 
X = Data'; 
X -= mean(Data(:));
X /=std(X(:));


NormData = X';



% X = [1 2 3;4 5 6;7 8 9;6 5 2;7 7 1] % for test use
		
		
%Dims
% Data: MxN
% NormData: MxN
