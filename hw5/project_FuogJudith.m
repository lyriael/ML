function Projected = project_FuogJudith(Data, Basis)

% I assume that the basis vectors are the colums of 'Basis'.
% 
assert(size(Data,2),size(Basis,1));
[n,k] = size(Basis);
assert(n>k);
m = size(Data,1);

Data = Data';
Y = Basis'*Data; % kxm = kxn * nxm

Projected = Y';

%Dim
%Data: NxM
