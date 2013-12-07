function [NormData] = normalizeData_FuogJudith(Data)

[m,n] = size(Data); % MxN Matrix
X = Data';

%% 
% normalize mean (step 1&2)
% x_i - mu (step 2), where mu is the mean of all samples (step 1).
X -= mean(Data(:));
%% 
% normalize standart deviation (step 3&4)
% ... thats a bit complicated.
% I had it more like it is in the script/homework sheet, then I found
% that std does exactly what I need in step 3 and division by std(X(:))
% is what happens in step 4.
X /=std(X(:));

NormData = X';

