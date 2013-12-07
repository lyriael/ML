function Projected = project_FuogJudith(Data, Basis)

[n,k] = size(Basis);
m = size(Data,1);

% for debug reasons
assert(size(Data,2),size(Basis,1)); 
assert(n>=k);

Data = Data';

%%
% Make a transformation of bases
% y_i = u'x_i
% where we have KxN * NxM = KxM
Y = Basis'*Data; 
Projected = Y';
