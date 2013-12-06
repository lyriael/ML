function [C, A] = Kmeans_Dummy(X, Cinit)

	n = size(X,1); 		%% Dimension
	m = size(X,2); 		%% Size of dataset
	k = size(Cinit,2); 	%% Number of clusters	
	C = Cinit;

	Diff = 100;	%% Usually only a few iterations are needed until
			%% the convergence is obvious. 

	while (Diff != 0)
		A = dsearchn(C',X'); 	%% Find closesed mu	
					%% Reset clustercentre	
		Mu = zeros(n,k); 	%% value sum
		count = zeros(1,k); 	%% count sum
		for i = 1:m
			Mu(:,A(i)) += X(:,i);
			count(A(i)) += 1;
		end	
		Mu = Mu./(repmat(count,n,1));
		Diff = sum(sum(abs(C-Mu),1)); %% Needed for convergence check
		C = Mu; 		      %% Initialize new cluster centers	
	endwhile

	d = distortion_func(X', A, C', k);	%% print costs
	fprintf('the cost function result is %d\n', d);	
end

%% cost function
function J = distortion_func(X, A, C, k)
        J = 0;
        for i = 1:k
                dif = X(A==i, :) - repmat(C(i, :), sum(A==i), 1);
                J += sum(sum(dif .^ 2, 2));
        endfor        
endfunction
