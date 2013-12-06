function Projected = project_FuogJudith(Data, Basis)

% I assume that the basis vectors are the colums of 'Basis'.
% 
assert(size(Data,1),size(Basis,1));
[n,k] = size(Basis);
assert(n>k);
m = size(Data,2);

Data = Data';
Y = Eigen'*Data; % kxm = kxn * nxm

Projected = Y';
