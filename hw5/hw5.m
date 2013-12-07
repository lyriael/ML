close all;
clear all;
clc;

% Am I running on octave?
isOctave = exist('OCTAVE_VERSION') ~= 0;

if isOctave
    VLFEAT_FOLDER = '~/Dokumente/Studium/Informatik/Machine Learning/homework/vlfeat'; %Put here the path to the root floder of VlFeat
    addpath ([VLFEAT_FOLDER, '/toolbox']);
    vl_setup;
else
    VLFEAT_FOLDER = '~/Documents/vlfeat-0.9.17/'; %Put here the path to the root floder of VlFeat
    run([VLFEAT_FOLDER  '/toolbox/vl_setup.m']);
end

%%
% Data contains our training examples. It is an Nx576 matrix. Each row
% represents one train example. The training examples are 24x24 grayscale
% images represented as a 1x576 row vector. The range of possible values
% are from 0 to 255. The dataset contains the mirrored pair of every face
% in the dataset.

% Labels contain the labesl of the training data. It is an 1xN matrix. If
% Labels(1,k) = 1, it means the kth sample is a face, if it is -1, than the
% sample is not a face. All the labels are either 1 or -1.

load('faces.mat');
N = size(Data,1);

%%
% Here we plot some examples images from the database. Notice, that the
% dataset contains the mirrores images.

faceIdx = Labels > 0;
nonFaceIdx = Labels < 0;
figure;
n = 6; % number of examples in a row
% faces (first row)
FaceData = Data(faceIdx,:);
for k = 1:n
    im = reshape(FaceData(k,:), [24 24]);
    subplot(2,n,k);
    imshow( im/255 );
    if k==1
        title( 'Faces' );
    end
end
% non faces (second row)
NonFaceData = Data(nonFaceIdx,:);
for k = 1:n
    im = reshape(NonFaceData(k,:), [24 24]);
    subplot(2,n,k+n);
    imshow( im/255 );
    if k==1
        title( 'Non faces' );
    end
end

%%
% First we select the train and test data randomly. We select aproximately
% testPercent% images for test images from the data. testIdx(1,k) = 1 if
% Data(k,:) belongs to the test set and 0 if it does not. trainIdx is
% defined similarly.
if isOctave
    rand ('seed', 0);
else
    rng(0,'twister');
end

testPercent = 30; 
testIdx = rand(1,N) <= testPercent/100;
trainIdx = ~testIdx;

DataTrain = Data(trainIdx,:);
LabelsTrain = Labels(1,trainIdx);
DataITest = Data(testIdx,:);
LabelsTest = Labels(1,testIdx);

Faces = DataTrain(LabelsTrain == 1,:);

%normalization
[NormFaces] = normalizeData_FuogJudith(Faces);

%SVM setup
lambda = 100;
svm_iters = 10000000;
y = LabelsTrain;
y(y == 0) = -1;


% PCA
%number of eigenvectors
k = 526;
Efaces = pca_FuogJudith(NormFaces,k);

% Project Data to PCA basis
DataTrain = project_FuogJudith(DataTrain, Efaces);
DataITest = project_FuogJudith(DataITest, Efaces);

% SVM
[w, b] = vl_svmtrain(DataTrain', y, lambda, 'MaxNumIterations', svm_iters);


%%
% Test Svm.
% compute the test scores of the SVM
scores = (DataITest*w + b)';% complete the code
classifierOutput = (scores >= 0.0) - (scores < 0.0);
% accuracy
good = classifierOutput == LabelsTest;
accuracy = 100*sum(good)/size(good,2);
fprintf( 'accuracy: %f%% (%d/%d)\n', accuracy, sum(good), size(good,2) );
