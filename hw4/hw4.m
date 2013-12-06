clear all;
close all;
clc;

%% load data
% load('data1.mat');
load('data1.mat');

K = max(labels); % number of clusters

%% display data
figure;
scatter(X(1,:),X(2,:),'b');
axis equal;
title('data');

colors = {'b', 'r', 'g'};
figure;
for j = 1:K
    scatter( X(1,labels == j), X(2,labels == j), colors{j} );
    hold on;
end
axis equal;
title('data with ground truth cluster labelling');

%% cluster with k-means
perm = randperm(size(X,2));
Cinit = X(:,perm(1:K));

fprintf('k-means ... ');
tic;
[C, A] = Kmeans_FUOGJUDITH(X, Cinit);
toc;

%% display clustering
figure;
for j = 1:K
    scatter( X(1,A == j), X(2,A == j), colors{j} );
    hold on;
    scatter( C(1,j), C(2,j), 100, colors{j}, 'filled' );
end
axis equal;
title('clustered data');
