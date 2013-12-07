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


%%%%%%%%% ITERATION %%%%%%%%%%%%
iterations = 1;
steps = 525;
upperLimit = size(Data,2);

innerIterations = ceil(upperLimit/steps);
A = zeros(iterations, innerIterations);

for j=1:iterations
	tic;
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
	fprintf('InnerIterations: %i\n',innerIterations);
	i = 1;
	while i<=innerIterations
		principalComponents = 1 + (i-1)*steps;
		fprintf('start inner %i loop, k: %i\n',i, principalComponents);
		% PCA
		%number of eigenvectors
		Efaces = pca_FuogJudith(NormFaces, principalComponents);

		% Project Data to PCA basis
		DataTrainTemp = project_FuogJudith(DataTrain, Efaces);
		DataITestTemp = project_FuogJudith(DataITest, Efaces);

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
		accuracy = 100*sum(good)/size(good,2);
		elapsed_time = toc;
		
		A(j,i) = accuracy;
		%fprintf('i: %i, k: %i, accuracy: %d, elapsed_time: %d\n', i, principalComponents, A(j,i), elapsed_time);
		i +=1;
	endwhile
end

%%%%%%%% PLOT %%%%%%%%%%%
for j=1:iterations
	for i=1:InnerIterations
		fprintf('%d ', A(j,i));
	end
	printf('\n');
end
