close all;
clear all;
clc;

%%%%%%%%%%% SETUUP %%%%%%%%%%%
% Am I running octave?
isOctave = exist('OCTAVE_VERSION') ~= 0;

% Init path to vlfeat
if isOctave
    VLFEAT_FOLDER = '~/Dokumente/Studium/Informatik/Machine Learning/homework/vlfeat'; 
    addpath ([VLFEAT_FOLDER, '/toolbox']);
    vl_setup;
else
    VLFEAT_FOLDER = '~/Documents/vlfeat-0.9.17/'; 
    run([VLFEAT_FOLDER  '/toolbox/vl_setup.m']);
end

%%%%%%%%% DATA LOAD %%%%%%%%%%%%
load('faces.mat');
N = size(Data,1);
testPercent = 30; 

% Load data and select test set
testIdx = rand(1,N) <= testPercent/100;
trainIdx = ~testIdx;
DataTrain = Data(trainIdx,:);
LabelsTrain = Labels(1,trainIdx);
DataITest = Data(testIdx,:);
LabelsTest = Labels(1,testIdx);
Faces = DataTrain(LabelsTrain == 1,:);

%normalization
[NormFaces] = normalizeData_FuogJudith(Faces);


%%%%%%%%% ITERATIONS %%%%%%%%%%%%
iterations = 2;
steps = 150;
dimension = size(Data,2);

outerIterations = ceil(dimension/steps);
accuracy = zeros(1, outerIterations);

i = 1;
while i<=outerIterations
	fprintf('Outer Iteration: %i\n',i);
	tic;
	principalComponents = 1 + (i-1)*steps;
	%fprintf('start inner %i loop, k: %i\n',i, principalComponents); %debug
	% PCA
	%number of eigenvectors
	Efaces = pca_FuogJudith(NormFaces, principalComponents);

	% Project Data to PCA basis
	DataTrainTemp = project_FuogJudith(DataTrain, Efaces);
	DataITestTemp = project_FuogJudith(DataITest, Efaces);

	% get average accuracy of SVM
	accuracyTemp = zeros(1,iterations);
	for j=1:iterations
		% SVM
		lambda = 100;
		svm_iters = 10000000;
		y = LabelsTrain;
		y(y == 0) = -1;
		[w, b] = vl_svmtrain(DataTrainTemp', y, lambda, 'MaxNumIterations', svm_iters);

		% compute the test scores and accuracy of the SVM
		scores = (DataITestTemp*w + b)';
		classifierOutput = (scores >= 0.0) - (scores < 0.0);
		good = classifierOutput == LabelsTest;
		accuracyTemp(j) = 100*sum(good)/size(good,2);
	end
	accuracy(i) = mean(accuracyTemp);
	elapsed_time = toc;
	fprintf('k: %i, \taccuracy: %d, \telapsed_time: %d\n', principalComponents, accuracy(i), elapsed_time);
	i +=1;
endwhile

%%%%%%%% PLOT %%%%%%%%%%%
yAxis = accuracy;
xAxis = [1:steps:dimension];
figure;
plot(xAxis, yAxis);
axis([1 dimension 40 100]);
title('Principal Component Analysis');
xlabel('Number of Principal Components taken');
ylabel('accuracy (%)');

