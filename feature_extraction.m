clear
close
clc

%% load images
imds = imageDatastore('C:\Users\Shuvo Saha Roy\Desktop\covid\datasets\binary\chest_xray',...
    'IncludeSubfolders',true,'LabelSource','foldernames');
%read any image
% img = readimage(imds,2);
% [rows, columns, numberOfColorChannels] = size(img);
% traning testing split
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized');


%% Alex net
% net = alexnet;
% layer = 'pool5';

%% Resnet
% net = resnet18;
% layer = 'pool5';

%% Googlenet
% net = googlenet;
% layer = 'pool5-drop_7x7_s1';

%% densenet201
% net = densenet201;
% layer = 'avg_pool';

%% shufflenet
net = shufflenet;
layer = 'node_200';


% analyzeNetwork(net)
% input layer size of resnet. input image side should be same as layer
inputSize = net.Layers(1).InputSize;
% resizing the images according to layer size
augmenter = imageDataAugmenter('RandXReflection', true);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,'ColorPreprocessing','gray2rgb',...
    'DataAugmentation', augmenter);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest,'ColorPreprocessing','gray2rgb',...
    'DataAugmentation', augmenter);

featuresTrain = activations(net,augimdsTrain,layer,OutputAs="rows");
featuresTest = activations(net,augimdsTest,layer,OutputAs="rows");


% % featuresTrain = activations(net,imdsTrain,layer,OutputAs="rows");
% % featuresTest = activations(net,imdsTest,layer,OutputAs="rows");

% 
YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;

% c = YTrain;
% [GN, ~, G] = unique(c)

% dataset = [featuresTrain YTrain];

svmtrain = fitcsvm(featuresTrain, YTrain);

YPred = predict(svmtrain, featuresTest);
accuracy = mean(YPred == YTest);
% plotconfusion(imdsTest.Labels, YPred)

actual = YTest;
predicted= YPred;
cm = confusionmat(actual,predicted);
cm = cm';
precision = diag(cm)./sum(cm,2);
overall_precision = mean(precision)
recall= diag(cm)./sum(cm,1)';
overall_recall = mean(recall)
f1_score = 2*((overall_precision*overall_recall)/(overall_precision+overall_recall));