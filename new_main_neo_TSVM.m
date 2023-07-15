clc;
clear;
close all;

% data_set = ["amazon_sat","emotions","yeast"];
data_set = ["ecoli"];
% reverse = 1; %only in case of 2-d datasets
reverse=0;
plot=0;
save_results=0;
%next lines for xlsx.
%don't forget to modify this according to classifier no.
%we are using 3 classifier here.
xlsx_loop=1;
mat_size = 3*2;
datalen = length(data_set)*(mat_size+2);
write_xlsx =0;

for f = 1:length(data_set)
    %% Add sub-folders containing functions
    %clc;clear;
    %addpath('data','evaluation');
    %addpath(genpath('method'));
    %
    %% Load a multi-label dataset
    dataset = data_set(f);
    addpath('D:\MSc thesis\multiclass\multiclass dataset');
    load(strcat(dataset,".mat"));
    
    if reverse==1
        target=target';
    end
    
    seed = 0;
    rng(seed,'Twister');
    %scurr = rng;
    
    %% Randomly select part of data if very large
    random_select = 0; % set random_select = 1 to select otherwise 0
    max_num = 5000;
    if (size(data,1) > max_num) && random_select
        nRows = size(data,1);
        nSample = max_num;
        rndIDX = randperm(nRows);
        index = rndIDX(1:nSample);
        data = data(index, :);
        target = target(:,index);
    end
    
    %% set target labels to 1 or -1
    [m,~]=size(data);
    for i=1:m
        if target(i)~=1
            target(i)=-1;
        end
    end
    
    %% data reduction
    datareduction=0; %input('press 1 if data reduction is req else 2=');
    if datareduction==1
        [data,target]=datared(data,target,0.4);
    end
    
    
    %% select kernel
    kernel=1;%input('1 for linear, 2 for rbf, 3 for poly =');
    if kernel==1
        ker='linear';
    elseif kernel==2
        ker='rbf';
    elseif kernel==3
        ker='poly'; % not added yet
    end
    
    %% data normalize   %input('press 1 if normlization is req else 2=');
    %set off for moon datasets
    normalize = 1;
    if normalize==1
        data=svdatanorm(data,'ker');
    end
    % data = normalize(data,'range');
    % data = normalize(data,'zscore');
    
    
    %% Perform n-fold cross validation and obtain evaluation results
    
    %Cross varidation (train: 70%, test: 30%)
    
    
    num_fold = 10; num_metric = 2; num_method = 2;
    indices = crossvalind('Kfold',size(data,1),num_fold);
    
    Results = zeros(num_metric+1,num_fold,num_method);
    
    for i = 1:num_fold
        test = (indices == i);
        a = i+1;
        if a>num_fold
            a = 1;
        end
        vald = (indices == a );
        train = ~test & ~vald;
        
        fprintf(' Fold %d ',i);
        
        plotting = 0;
        
        data_train = data(logical(train+vald),:);
        target_train = target(logical(train+vald));
        data_test = data(logical(test),:);
        target_test = target(logical(test));
        
        %         %=============1. SVM_classifier(1-plane)===========================
        addpath("D:\MSc thesis\multiclass\SVM");
        %Set SVM_classifier parameter
        R=0.1; kerpara=0.1;
        %train classifier
        [w,b,~,time_taken]=svmclassifier(data_train,target_train,ker,R,1,kerpara);
        Results(1,i,1) = time_taken;
        %Results(4,i,1) = nsv;
        %classifier correctness
        TrainCorr = svmsvcoutput(data_train,target_train,data_train,target_train,ker,1,w,b,0,kerpara);
        TestCorr = svmsvcoutput(data_train,target_train,data_test,target_test,ker,1,w,b,0,kerpara);
        Results(2,i,1) = TrainCorr;
        Results(3,i,1) = TestCorr;
        %==========================plotting=============================
        if plot==1 && isequal(ker,'rbf')
            figure(1);
            plot2(data_train,target_train,data_test,target_test,[w;b],kerpara);hold on;
            hold off;
        elseif plot==1 && isequal(ker,'linear')
            figure(1);
            plot_linear(data_train,target_train,data_test,target_test,[w;b]);hold on;
            hold off;
        end
        %--------------------------------------------------------------------------------
        
        
        %================2. TWSVM(Original)================
        %Set TWSVM parameter
        addpath("D:\MSc thesis\multiclass\TWSVM matlab code");
        R=0.1; kerpara=0.1;
        %Train classifier
        [w3,gamma3,w4,gamma4,nsv,time_taken]=twsvmclassifier(data_train,target_train,ker,R,1,kerpara);
        Results(1,i,2) = time_taken;
        %classifier correctness
        switch  lower(ker)
            case 'linear'
                TrainCorr=correctnessgensvm(data_train,target_train,w3,gamma3,w4,gamma4,plotting);
                TestCorr=correctnessgensvm(data_test,target_test,w3,gamma3,w4,gamma4,0);
            case 'rbf'
                TrainCorr=correctnessgensvmker(data_train,data_train,target_train,target_train,w3,gamma3,w4,gamma4,ker,1,kerpara);
                TestCorr=correctnessgensvmker(data_test,data_train,target_test,target_train,w3,gamma3,w4,gamma4,ker,1,kerpara);
            case 'poly'
                TrainCorr=correctnessgensvmker(data_train,data_train,target_train,target_train,w3,gamma3,w4,gamma4,ker,1,kerpara);
                TestCorr=correctnessgensvmker(data_test,data_train,target_test,target_train,w3,gamma3,w4,gamma4,ker,1,kerpara);
        end
        Results(2,i,2) = TrainCorr;
        Results(3,i,2) = TestCorr;
        
        %======plotting========================================
        if plot==1 && isequal(ker,'rbf')
            figure(2);
            plot2(data_train,target_train,data_test,target_test,[w3;gamma3],kerpara);hold on;
            plot2(data_train,target_train,data_test,target_test,[w4;gamma4],kerpara);hold off;
        elseif plot==1 && isequal(ker,'linear')
            figure(2);
            plot_linear(data_train,target_train,data_test,target_test,[w3;gamma3]);hold on;
            plot_linear(data_train,target_train,data_test,target_test,[w4;gamma4]);hold off;
        end
%         %======================================================
        
        %================3. ATSVM models================
        %Set TWSVM parameter
%         addpath('C:\Users\Shuvo Saha Roy\Desktop\p\best_parameter');
%         load(strcat("Neo_TSVM_",dataset,".mat"));
%         c1=0.5; c3= 1e-4; 
%         C=0.1; kerpara=0.1;
%         %Train classifier
%         [z1,z2,nsv,time_taken]=ATSVM(data_train,target_train,ker,C,c1,c3,1,kerpara);
%         Results(1,i,3) = time_taken;
%         %classifier correctness
%         switch  lower(ker)
%             case 'linear'
%                 TrainCorr=correctnessgensvm(data_train,target_train,w3,gamma3,w4,gamma4,plotting);
%                 TestCorr=correctnessgensvm(data_test,target_test,w3,gamma3,w4,gamma4,0);
%             case 'rbf'
%                 TrainCorr=correctnessgensvmker(data_train,data_train,target_train,target_train,w3,gamma3,w4,gamma4,ker,1,kerpara);
%                 TestCorr=correctnessgensvmker(data_test,data_train,target_test,target_train,w3,gamma3,w4,gamma4,ker,1,kerpara);
%             case 'poly'
%                 TrainCorr=correctnessgensvmker(data_train,data_train,target_train,target_train,w3,gamma3,w4,gamma4,ker,1,kerpara);
%                 TestCorr=correctnessgensvmker(data_test,data_train,target_test,target_train,w3,gamma3,w4,gamma4,ker,1,kerpara);
%         end
%         Results(2,i,3) = TrainCorr;
%         Results(3,i,3) = TestCorr;
%         
%         %======plotting========================================
%         if plot==1 && isequal(ker,'rbf')
%             figure(3);
%             plot2(data_train,target_train,data_test,target_test,[w3;gamma3],kerpara);hold on;
%             plot2(data_train,target_train,data_test,target_test,[w4;gamma4],kerpara);hold off;
%         elseif plot==1 && isequal(ker,'linear')
%             figure(3);
%             plot_linear(data_train,target_train,data_test,target_test,[w3;gamma3]);hold on;
%             plot_linear(data_train,target_train,data_test,target_test,[w4;gamma4]);hold off;
%         end
        %=========================================================
        
        %================4. Neo_TSVM================
        %Set TWSVM parameter
%         addpath('C:\Users\Shuvo Saha Roy\Desktop\p\best_parameter');
%         load(strcat("Neo_TSVM_",dataset,".mat"));
        c1=0.5; c2=0; c3=0.1; c4=0.1; 
        kerpara=0.1;
        %Train classifier
        [D,Xtest] = kernel_matrix(data_train, data_test,ker,kerpara);
        [~,Testacc, Trainacc,time_taken,z1,z2]=Neo_TSVM(D,target_train,Xtest,target_test,c1,c2,c3,c4);
        Results(1,i,4) = time_taken;
        Results(2,i,4) = Trainacc;
        Results(3,i,4) = Testacc;
        
        %======plotting========================================
        if plot==1 && isequal(ker,'rbf')
            figure(4);
            plot2(data_train,target_train,data_test,target_test,z1,kerpara);hold on;
            plot2(data_train,target_train,data_test,target_test,z2,kerpara);hold off;
        elseif plot==1 && isequal(ker,'linear')
            figure(4);
            plot_linear(data_train,target_train,data_test,target_test,z1);hold on;
            plot_linear(data_train,target_train,data_test,target_test,z2);hold off;
        end
        %======================================================
        
%         %================3. GEPSVM model================
%         %Set TWSVM parameter
% %          addpath('C:\Users\Shuvo Saha Roy\Desktop\testing\best_parameter\neo_atsvm1')
% %          load(strcat("Neo_Atsvm_",dataset,".mat"));
        %Train classifier
%         [D,Xtest] = kernel_matrix(data_train, data_test,ker,kerpara);
%         [~,Testacc, Trainacc,time_taken,z1,z2]=GEPSVM(D,target_train,Xtest,target_test);
%         Results(1,i,3) = time_taken;
%         Results(2,i,3) = Trainacc;
%         Results(3,i,3) = Testacc;
%         
%         %======plotting========================================
%         if plot==1 && isequal(ker,'rbf')
%             figure(3);
%             plot2(data_train,target_train,data_test,target_test,z1,kerpara);hold on;
%             plot2(data_train,target_train,data_test,target_test,z2,kerpara);hold off;
%         elseif plot==1 && isequal(ker,'linear')
%             figure(3);
%             plot_linear(data_train,target_train,data_test,target_test,z1);hold on;
%             plot_linear(data_train,target_train,data_test,target_test,z2);hold off;
%         end
%         %======================================================
        
    end
    
    ignore = [];  Results(:,:,ignore) = [];
    meanResults = squeeze(mean(Results,2));
    stdResults = squeeze(std(Results,0,2) / sqrt(size(Results,2)));
    
%     macc = meanResults(3,3)
    %addpath('D:\research-track\table generators latex');
    %meanResults=three_decimals(meanResults);
    %stdResults=three_decimals(stdResults);
    %% Save the evaluation results
    
    if write_xlsx==1
        filename = 'Kernel_results.xlsx';
        data_result= [meanResults; stdResults];
        dataset_name = strcat('Dataset_', dataset);
        header1 = [dataset_name "Twin" "Neo_TSVM" "GEPSVM" "ATSVM"];
        levels = ["execution_time" "training_acc" "test_acc" "std_exe_time" "std_tr_acc" "std_test_acc"]';
        final_data = [levels data_result];
        final_data = [header1; final_data];
        sheet = 1;
        for xlsx_loop= xlsx_loop : mat_size : datalen
            xlRange = strcat('A',num2str(xlsx_loop));
            xlsx_loop=xlsx_loop+8;
            break;
        end

        xlswrite(filename,final_data,sheet,xlRange)
    end
    % filename=strcat('path',dataset,'.mat');
    
    if save_results ==1
        filename=strcat('Result_',dataset,'.mat');
        save(filename,'Results','meanResults','stdResults','-mat');
    end
    %% Show the experimental results, 
    disp(dataset);
    fprintf('\nRows are Time, train accuracy, test accuracy\n')
    fprintf('columns are %s models, Twsvm, Neo_TSVM, GEPSVM, ATSVM\n',ker);
    disp(meanResults);
    disp('=========================standard deviation==================================');
    disp(stdResults);
   
end
