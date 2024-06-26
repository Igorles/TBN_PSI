

% Title: Three-Branch Neural Network for No-Reference Quality Assessment of Pan-Sharpening Images
% Authors: Igor Stepien and Mariusz Oszust 
% istepien@kia.prz.edu.pl, marosz@kia.prz.edu.pl, www.marosz.kia.prz.edu.pl
% Rzeszow University of Technology



warning('off','all')
load data.mat
dir_1 = strcat(pwd,filesep,'database',filesep,'part1',filesep,'WorldView-3',filesep);
dir_2 = strcat(pwd,filesep,'database',filesep,'part2',filesep,'WorldView-3',filesep);

ref_ind=[];
image_no=6;

for j=1:160
    for i=1:image_no
        ref_ind=[ref_ind;j];
    end
end

miniBatchSize = 32;
maxEpochs = 50;


% Demo uses PS images created for WV3 dataset.
% SAM scores are used for training.


tic
for i = 1 : 10

    train = ismember(ref_ind,C_WV3(i,:));
    test = ~train;

    name_TC_trainTmp = scoreRR.rgb.WV3.name(train);
    name_TC_train = strcat(dir_1,name_TC_trainTmp);
    name_PC_train = strcat(dir_2,name_TC_trainTmp);

    name_TC_testTmp = scoreRR.rgb.WV3.name(test);
    name_TC_test = strcat(dir_1,name_TC_testTmp);
    name_PC_test = strcat(dir_2,name_TC_testTmp);
    
    trainScores_TC = scoreRR.rgb.WV3.SAM(train,:); 
    trainScores_PC = scoreRR.NIR.WV3.SAM(train,:); 

    testScores_TC = scoreRR.rgb.WV3.SAM(test,:);
    testScores_PC = scoreRR.NIR.WV3.SAM(test,:);
    
    trainScores = sqrt(trainScores_TC.*trainScores_PC); 
    testScores = sqrt(testScores_TC.*testScores_PC); 

    train1 = imageDatastore(name_TC_train);
    train2 = imageDatastore(name_PC_train);
    test1 = imageDatastore(name_TC_test);
    test2 = imageDatastore(name_PC_test);

    train3 = arrayDatastore(trainScores);
    test3 = arrayDatastore(testScores);

    train1.ReadFcn = @customReadDatstoreImageTC;
    train2.ReadFcn = @customReadDatstoreImageTC;
    test1.ReadFcn = @customReadDatstoreImageTC;
    test2.ReadFcn = @customReadDatstoreImageTC;

    dsTrain = combine(train1,train2,train3);
    dsTest = combine(test1,test2,test3);

    options = trainingOptions('adam',...
        'InitialLearnRate',0.001, ...
        'MiniBatchSize',miniBatchSize, ...
        'MaxEpochs',maxEpochs, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',10,...
        'Plots','none',...
        'ExecutionEnvironment','gpu',...
        'Shuffle','every-epoch', ...
        'Verbose',true);

net = trainNetwork(dsTrain, lgraph,options);
YPred = predict(net,dsTest,'ExecutionEnvironment','gpu');

valueTst = abs(metric_evaluation(testScores,YPred)) %SRCC, KRCC, PLCC, and RMSE

score(i,:)=valueTst;

end
toc
median(score)
plot(lgraph)

% Values reported in the paper
% SRCC, KRCC, PLCC
% 0.9240,0.7807, 0.9338
% The approaches are run in Matlab R2023b, Windows 10, 
% on a PC with an i9-12900k CPU, 128 GB RAM, and an RTX 3090 graphic card. 

function data = customReadDatstoreImageTC(filename)
onState = warning('off', 'backtrace');
c = onCleanup(@() warning(onState));
data = imread(filename);
data = imresize(data,[299 299]);
end