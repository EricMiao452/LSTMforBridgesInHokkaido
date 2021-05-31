clear
clc

%% load data
load Bridge_series.mat
load Bridge_grade.mat
Num_cell=numel(Bridge_series)
% TypeIndex=xlsread('TypeIndex.xlsx')

XTrain=cell(Num_cell,1)
for i=1:Num_cell
    sequence=cell2mat(Bridge_series{i});
    newCell=sequence(:,2:12)
    n=size(newCell,1)
    if n>30
        newCell=newCell(n-30+1:n,:)
        XTrain{i}=newCell'
    else
        XTrain{i}=newCell';
    end
%     XTrain{i}=newCell';
end

% regularization the data
YTrain=bridge_grade(1:Num_cell);
YTrain=categorical(YTrain);

% Xtrain:
%         Features: 11
%         Length:vary
% visualize
figure
plot(XTrain{1}')
xlabel("Time Step")
title("Training Observation 1")
numFeatures=size(XTrain{1}',2)
legend("Features"+string(1:numFeatures),'location','northeastoutside')

%% See all the data
% Prepare Data for Padding
% numberobservations=numel(XTrain);
% for i=1:numberobservations
%     sequence=XTrain{i};
%     sequenceLengths(i)=size(sequence,2);
% end
% 
% %plot shade
% minBatchSize=27;
% 
% for i=1:round(Num_cell/minBatchSize)-1
%     shadeBar(i)=max(sequenceLengths((i-1)*minBatchSize+1:i*minBatchSize));
%     positions(i)=(i-1)*27+13.5;
% end
% 
% figure(2)
% title("Unsorted Data")
% xlabel("Sequence")
% ylabel("Length")
% ylim([0 270])
% bar(positions,shadeBar,'BarWidth',1,'facecolor','yellow','edgecolor','red','FaceAlpha',0.5)
% hold on;
% bar(sequenceLengths,'facecolor','blue')
% 
% 
% % sorting data
% [sequenceLengths,idx]=sort(sequenceLengths);
% XTrain=XTrain(idx);
% YTrain=YTrain(idx);
% YTrain=categorical(YTrain)
% for i=1:round(Num_cell/minBatchSize)
%     if i<round(Num_cell/minBatchSize)
%         shadeBarSorted(i)=max(sequenceLengths((i-1)*minBatchSize+1:i*minBatchSize));
%         positionsSorted(i)=(i-1)*27+13.5;
%     else
%         shadeBarSorted(i)=max(sequenceLengths((i-1)*minBatchSize+1:Num_cell));
%         positionsSorted(i)=(i-1)*27+(Num_cell-((i-1)*minBatchSize+1))/2;
%     end
%     
% end
% figure(3)
% bar(positionsSorted,shadeBarSorted,'BarWidth',1,'facecolor','yellow','edgecolor','red','FaceAlpha',0.5)
% hold on;
% bar(sequenceLengths,'facecolor','blue');
% ylim([0 100])
% xlabel("Sequence")
% ylabel("Length")
% title("Sorted Data")


% delete bridges that pedestrain pass way
% BridgeTypeIndex=xlsread('TypeIndex.xlsx')
% XTrain_new={}
% YTrain_new=[]
% % for i=1:Num_cell
% %     temp=XTrain{i,1}
% %     if XTrain{i,1}(9,sequenceLengths(i)) ==0
% %         temp(9,:)=0.001            
% %     end    
% %      if XTrain{i,1}(10,sequenceLengths(i)) ==0
% %         temp(10,:)=0.001     
% %      end  
% %       XTrain_new=[XTrain_new;temp]      
% % end
% YTrain_new=YTrain
% for i=1:Num_cell
%     if  BridgeTypeIndex(i)~=4
%             temp=XTrain{i,1}
%             XTrain_new=[XTrain_new;temp]   
%             YTrain_new=[YTrain_new;YTrain(i)]
%     end      
% end

%% splite data
[trainInd,valInd,testInd] = dividerand(3368,0.7,0.15,0.15);

XTrain_new=[XTrain(1:3339,1)]
YTrain_new=[YTrain(1:3339,1)]
[XTrain_new,YTrain_new]=Sort_Data(XTrain_new,YTrain_new) % sort data

XValidation=[XTrain(valInd,1)]
YValidation=[YTrain(valInd,1)]
[XValidation,YValidation]=Sort_Data(XValidation,YValidation) % sort data

XTest=[XTrain(testInd,1)]
YTest=[YTrain(testInd,1)]
[XTest,YTest]=Sort_Data(XTest,YTest) % sort data


%% Define LSTM Network Arcjitecture
inputSize=11;
numHiddenUnits=150;
numClasses=3;

layers=[...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer  
    ]
% train parameter
maxEpochs=2000;
minBachSize=1024;
checkpointPath ='C:\Users\Administrator\Desktop\RNN\checkpoint';
options=trainingOptions('adam',...
    'ExecutionEnvironment','gpu',...
    'GradientThreshold',3,...
    'MaxEpochs',maxEpochs,...
    'MiniBatchSize',minBachSize,...
    'ValidationData',{XValidation,YValidation},...
    'ValidationFrequency',6,...
    'SequenceLength','longest',...
    'Shuffle','never',...
    'Verbose',0,...
    'Plots','training-progress' ) 
%     'ValidationData',{XValidation,YValidation},...
%% clear all files in the folder
% delete C:\Users\Administrator\Desktop\RNN\checkpoint\*.mat;
%% Train LsTM Network
[Trainingnet,TrainigInfo]=trainNetwork(XTrain_new,YTrain_new,layers,options);

%% Accuracy for All dataset

miniBatchSize=1024;
[allPred,allscores]=classify(Trainingnet,XTrain_new,...
     'MiniBatchSize',miniBatchSize,...
    'SequenceLength','longest');
figure(4)
plotconfusion(YTrain_new,allPred,'All Accuracy') % confusion matrix
% roc curves
alllabels=zeros(3,size(allPred,1))
for i=1:size(allPred,1)
    alllabels(YTrain_new(i),i)=1;
end
figure(5)
plotroc(alllabels,allscores')
[all_tpr,all_fpr,all_thresholds]=roc(alllabels,allscores')   % get detail information of the roc curves of the three class
Cal_Auc0=CalculateAUC(all_fpr,all_tpr) % calculate the AUC value



figure(6)
plot(TrainigInfo.TrainingLoss)
hold on
plot(smoothdata(TrainigInfo.TrainingLoss,'movmean',5))
% plot accuracy for training dataset and validation dataset
figure(7)
plot(TrainigInfo.TrainingAccuracy)
hold on
idxs = ~isnan(TrainigInfo.ValidationAccuracy);
x = 1:size(TrainigInfo.ValidationAccuracy,2);
plot(x(idxs), TrainigInfo.ValidationAccuracy(idxs),'--o','MarkerEdgeColor','black','MarkerFaceColor','black');
plot(TrainigInfo.ValidationAccuracy)
hold on
plot(smoothdata(TrainigInfo.TrainingAccuracy,'movmean',5))
%% Accuracy for training datset
XTrain1=[XTrain(trainInd,1)]
YTrain1=[YTrain(trainInd,1)]
[XTrain1,YTrain1]=Sort_Data(XTrain1,YTrain1) % sort data

[trainPred,trainscores]=classify(Trainingnet,XTrain1,...
    'SequenceLength','longest');
figure(8)
plotconfusion(YTrain1,trainPred,'Training Accuracy') % confusion matrix
% roc curves
trainlabels=zeros(3,size(trainPred,1))
for i=1:size(trainPred,1)
    trainlabels(YTrain1(i),i)=1;
end
figure(9)
plotroc(trainlabels,trainscores')
[train_tpr,train_fpr,train_thresholds]=roc(trainlabels,trainscores')   % get detail information of the roc curves of the three class
Cal_Auc1=CalculateAUC(train_fpr,train_tpr) % calculate the AUC value
%% Accuracy for Validation dataset
validationPred=classify(Trainingnet,XValidation,...
    'SequenceLength','longest');
figure(10)
plotconfusion(YValidation,validationPred,'Validation Accuracy')

%plot roc
rocPred=predict(Trainingnet,XValidation)
rocTarget=zeros(3,size(rocPred,1))
for i=1:size(rocPred,1)
     rocTarget(YValidation(i),i)=1;
end
figure(11)
plotroc(rocTarget,rocPred')  % draw graphs of roc
[vali_tpr,vali_fpr,vali_thresholds]=roc(rocTarget,rocPred')   % get detail information of the roc curves of the three class
Cal_Auc2=CalculateAUC(vali_fpr,vali_tpr) % calculate the AUC value

%% accuracy for test dataset
[testPred,testScores]=classify(Trainingnet,XTest,...
    'SequenceLength','longest');
figure(12)
plotconfusion(YTest,testPred,'Test Accuracy')

% roc curves
testLabels=zeros(3,size(testPred,1))
for i=1:size(testPred,1)
    testLabels(YTest(i),i)=1;
end
figure(13)
plotroc(testLabels,testScores')
[test_tpr,test_fpr,test_thresholds]=roc(testLabels,testScores')   % get detail information of the roc curves of the three class
Cal_Auc3=CalculateAUC(test_fpr,test_tpr) % calculate the AUC value



%save Best_Results.mat

%% resume training with validation
% load('convnet_checkpoint__6__2020_04_07__13_25_42.mat','net')
maxEpochs=2000
minBachSize2=128
options=trainingOptions('adam',...
    'ExecutionEnvironment','gpu',...
    'GradientThreshold',1,...
    'MaxEpochs',maxEpochs,...
    'MiniBatchSize',minBachSize2,...
    'SequenceLength','longest',...
    'GradientThresholdMethod','l2norm',...
    'Shuffle','never',...
    'Verbose',0,...
    'Plots','training-progress')
[net2, infor]= trainNetwork(XValidation,YValidation,layers,options);
figure(14)
plot(smoothdata(infor.TrainingLoss,'movmean',5))

%% Summary part
% plot training loss and validation loss with epoch
figure(15)
Final_Training_loss=smoothdata(TrainigInfo.TrainingLoss,'movmean',6)
plot(Final_Training_loss)
hold on
Final_Validation_loss=smoothdata(infor.TrainingLoss(1:6000),'movmean',5)
plot(Final_Validation_loss)
legend('Validation Loss','Training Loss')

% plot training accuracy and validation accuracy with epoch
figure(16)
Final_Training_Accuracy=smoothdata(TrainigInfo.TrainingAccuracy,'movmean',6)
plot(Final_Training_Accuracy)
hold on
Final_Validation_Accuracy=smoothdata(infor.TrainingAccuracy(1:6000),'movmean',5)
plot(Final_Validation_Accuracy)
legend('Validation Accuracy','Training Accuracy')

%% Error Analysis
% XError=XTrain_new;
% YTest=YTrain_new;
% sequenceLengthsTest=[]
% numObservationsTest=numel(XTest);
% for i=1:numObservationsTest
%     sequence=XTest{i};
%     sequenceLengthsTest(i)=size(sequence,2);
% end
% [sequenceLengthsTest,idx]=sort(sequenceLengthsTest);
% XTest=XTest(idx);
% YTest=YTest(idx);
% miniBatchSize=2048;
% YPred=classify(net,XTest,...
%     'MiniBatchSize',miniBatchSize,...
%     'SequenceLength','longest');
% acc=sum(YPred==YTest)./numel(YTest)
Error_index=(YTrain_new==allPred)
Error_m=size(Error_index,1)
Error_m_index=0
Corect_m_index=0
Error_cell={}
clear  Error_year  Error_length Error_width  Error_elevation Error_highT Error_lowT
for i=1:Error_m
    if Error_index(i)==0
        Error_m_index=Error_m_index+1
        EleSize=size(XTrain_new{i},2)
        Error_year(Error_m_index)=XTrain_new{i}(11,EleSize)
        Error_length(Error_m_index)=XTrain_new{i}(1,EleSize)
        Error_width(Error_m_index)=XTrain_new{i}(2,EleSize)
        Error_elevation(Error_m_index)=XTrain_new{i}(3,EleSize)   
        Error_highT(Error_m_index)=XTrain_new{i}(3,EleSize)         
        Error_cell{Error_m_index,1}=XTrain_new{i}
    else
        Corect_m_index=Corect_m_index+1
         EleSize=size(XTrain_new{i},2)
         Correct_year(Corect_m_index)=XTrain_new{i}(11,EleSize)
    end
end

% Years
Error_year=Error_year'
Error_year=sort(Error_year)
Error_d=diff([Error_year;max(Error_year)+1]);
Error_count = diff(find([1;Error_d])) ;
Error_y_year =[Error_year(find(Error_d)) Error_count]

Correct_year=Correct_year'
Correct_year=sort(Correct_year)
Correct_d=diff([Correct_year;max(Correct_year)+1]);
Correct_count = diff(find([1;Correct_d])) ;
Correct_y =[Correct_year(find(Correct_d)) Correct_count]

% length
Error_length=Error_length'
Error_length=sort(Error_length)
Error_d=diff([Error_length;max(Error_length)+1]);
Error_count = diff(find([1;Error_d])) ;
Error_y_length =[Error_length(find(Error_d)) Error_count]

% width
Error_width=Error_width'
Error_width=sort(Error_width)
clear Error_d
clear Error_count
Error_d=diff([Error_width;max(Error_width)+1]);
Error_count = diff(find([1;Error_d])) ;
Error_y_width=[Error_width(find(Error_d)) Error_count]

% elevation
Error_elevation=Error_elevation'
Error_elevation=sort(Error_elevation)
clear Error_d
clear Error_count
Error_d=diff([Error_elevation;max(Error_elevation)+1]);
Error_count = diff(find([1;Error_d])) ;
Error_y_elevation =[Error_elevation(find(Error_d)) Error_count]

num=numel(Error_cell)
for i=1:num
    EleSize=size(XTrain_new{i},2);
    Error_highT(i)=Error_cell{i}(4,EleSize);
    Error_lowT(i)=Error_cell{i}(5,EleSize);
    Error_snow(i)=Error_cell{i}(6,EleSize);
    Error_rain(i)=Error_cell{i}(7,EleSize);
    Error_carbon(i)=Error_cell{i}(8,EleSize);
    Error_chloride(i)=Error_cell{i}(9,EleSize);
    Error_traffic(i)=Error_cell{i}(10,EleSize);
end

% highest 
Error_highT=Error_highT'
Error_highT=sort(Error_highT)
clear Error_d
clear Error_count
Error_d=diff([Error_highT;max(Error_highT)+1]);
Error_count = diff(find([1;Error_d])) ;
Error_y_highT =[Error_highT(find(Error_d)) Error_count]

% lowest
Error_lowT=Error_lowT'
Error_lowT=sort(Error_lowT)
clear Error_d
clear Error_count
Error_d=diff([Error_lowT;max(Error_lowT)+1]);
Error_count = diff(find([1;Error_d])) ;
Error_y_lowT =[Error_lowT(find(Error_d)) Error_count]

 % snow
Error_snow=Error_snow'
Error_snow=sort(Error_snow)
clear Error_d
clear Error_count
Error_d=diff([Error_snow;max(Error_snow)+1]);
Error_count = diff(find([1;Error_d])) ;
Error_y_snow =[Error_snow(find(Error_d)) Error_count]

% rain
Error_rain=Error_rain'
Error_rain=sort(Error_rain);
clear Error_d
clear Error_count
Error_d=diff([Error_rain;max(Error_rain)+1]);
Error_count = diff(find([1;Error_d])) ;
Error_y_rain =[Error_rain(find(Error_d)) Error_count]

% carbon
Error_carbon=Error_carbon'
Error_carbon=sort(Error_carbon)
clear Error_d
clear Error_count
Error_d=diff([Error_carbon;max(Error_carbon)+1]);
Error_count = diff(find([1;Error_d])) ;
Error_y_carbon =[Error_carbon(find(Error_d)) Error_count]

% chloride
Error_chloride=Error_chloride'
Error_chloride=sort(Error_chloride)
clear Error_d
clear Error_count
Error_d=diff([Error_chloride;max(Error_chloride)+1]);
Error_count = diff(find([1;Error_d])) ;
Error_y_chloride =[Error_chloride(find(Error_d)) Error_count]

% traffic
Error_traffic=Error_traffic'
Error_traffic=sort(Error_traffic)
clear Error_d
clear Error_count
Error_d=diff([Error_traffic;max(Error_traffic)+1]);
Error_count = diff(find([1;Error_d])) ;
Error_y_traffic =[Error_traffic(find(Error_d)) Error_count]

numberobservations=numel(XTrain);
for i=1:numberobservations
    sequence=XTrain{i};
    sequenceLengths(i)=size(sequence,2);
end
[~,Error_ID]=sort(sequenceLengths)
Error_ID=Error_ID'


%% write all data that needed into a .xlsx
% write training loss
name={'Trainingloss'}
xlswrite('Model training process and results data.xlsx',name,'sheet1','A1')
datanum=size(Final_Training_loss,2)
xlswrite('Model training process and results data.xlsx',Final_Validation_loss','sheet1','A2:A6001')
% write training accuracy
name={'Trainingaccuracy'}
xlswrite('Model training process and results data.xlsx',name,'sheet1','C1')
datanum=size(Final_Training_loss,2)
xlswrite('Model training process and results data.xlsx',Final_Training_Accuracy','sheet1','C2:C6001')

% write validation loss
name={'ValidationLoss'}
xlswrite('Model training process and results data.xlsx',name,'sheet1','B1')
datanum=size(Final_Training_loss,2)
xlswrite('Model training process and results data.xlsx',Final_Training_loss','sheet1','B2:B6001')
% write for validation accuracy
name={'Validationaccuracy'}
xlswrite('Model training process and results data.xlsx',name,'sheet1','D1')
xlswrite('Model training process and results data.xlsx',Final_Validation_Accuracy','sheet1','D2:D6001')


% write fpr and tpr for all dataset
num_ele=numel(all_fpr)
class_name={'Class1','Class2','Class3'}
for i=1:num_ele
    xlswrite('Model training process and results data.xlsx',class_name,'sheet2','A1:C1')
    xlswrite('Model training process and results data.xlsx',Cal_Auc0,'sheet2','A2:C2')
    % fpr
    num_data=size(all_fpr{1,i},2)
    m_ascii=65+(i-1)*2
    mrange=strcat(char(m_ascii),num2str(3),':',char(m_ascii),num2str(num_data+2))
    xlswrite('Model training process and results data.xlsx',all_fpr{1,i}','sheet2',mrange)
    % tpr
    m_ascii1=66+(i-1)*2
    mrange1=strcat(char(m_ascii1),num2str(3),':',char(m_ascii1),num2str(num_data+2))
    xlswrite('Model training process and results data.xlsx',all_tpr{1,i}','sheet2',mrange1)   
end

% write tpr and fpr for training dataset
num_ele=numel(train_fpr)
for i=1:num_ele
    xlswrite('Model training process and results data.xlsx',class_name,'sheet3','A1:C1')
    xlswrite('Model training process and results data.xlsx',Cal_Auc1,'sheet3','A2:C2')
    % tpr
    num_data=size(train_fpr{1,i},2)
    m_ascii=65+(i-1)*2
    mrange=strcat(char(m_ascii),num2str(3),':',char(m_ascii),num2str(num_data+2))
    xlswrite('Model training process and results data.xlsx',train_fpr{1,i}','sheet3',mrange)
    m_ascii1=66+(i-1)*2
    mrange1=strcat(char(m_ascii1),num2str(3),':',char(m_ascii1),num2str(num_data+2))
    xlswrite('Model training process and results data.xlsx',train_tpr{1,i}','sheet3',mrange1)   
end

% for validation 
num_ele=numel(vali_fpr)
for i=1:num_ele
    xlswrite('Model training process and results data.xlsx',class_name,'sheet4','A1:C1')
    xlswrite('Model training process and results data.xlsx',Cal_Auc2,'sheet4','A2:C2')
    % fpr
    num_data=size(vali_fpr{1,i},2)
    m_ascii=65+(i-1)*2
    mrange=strcat(char(m_ascii),num2str(3),':',char(m_ascii),num2str(num_data+2))
    xlswrite('Model training process and results data.xlsx',vali_fpr{1,i}','sheet4',mrange)
    %tpr
    m_ascii1=66+(i-1)*2
    mrange1=strcat(char(m_ascii1),num2str(3),':',char(m_ascii1),num2str(num_data+2))
    xlswrite('Model training process and results data.xlsx',vali_tpr{1,i}','sheet4',mrange1)   
end

% for test
num_ele=numel(test_fpr)
for i=1:num_ele
    xlswrite('Model training process and results data.xlsx',class_name,'sheet5','A1:C1')
    xlswrite('Model training process and results data.xlsx',Cal_Auc3,'sheet5','A2:C2')
    % fpr
    num_data=size(test_fpr{1,i},2)
    m_ascii=65+(i-1)*2
    mrange=strcat(char(m_ascii),num2str(3),':',char(m_ascii),num2str(num_data+2))
    xlswrite('Model training process and results data.xlsx',test_fpr{1,i}','sheet5',mrange)
    %tpr
    m_ascii1=66+(i-1)*2
    mrange1=strcat(char(m_ascii1),num2str(3),':',char(m_ascii1),num2str(num_data+2))
    xlswrite('Model training process and results data.xlsx',test_tpr{1,i}','sheet5',mrange1)   
end

% test for one bridge
for i=1:3339
randID=randperm(3339,1) % find a bridge ID=326
rand_X=XTrain_new{randID}
rand_Y=YTrain_new(randID)
[randPred,testScores]=classify(Trainingnet,rand_X,...
    'SequenceLength','longest')
if size(rand_X,2)<30 && randPred==rand_Y
    break
end
end

randID326=randID326'
for i=1:4
    rand_X_new=randID326(:,1:12+(i-1))
    rand_pre_new(i)=classify(Trainingnet,rand_X_new,...
    'SequenceLength','longest')
end
    
%%  calculate the accuracy for different deck area
deckIndex=Error_index;
deck_m=size(deckIndex,1);
success_deck=[];
all_deck=zeros(deck_m,1);
for i=1:deck_m
    all_deck(i)=Bridge_series{i, 1}{1, 2}* Bridge_series{i, 1}{1, 3};
    if deckIndex(i)==1 
        success_deck=[success_deck; Bridge_series{i, 1}{1, 2}* Bridge_series{i, 1}{1, 3}];
    end
end
hh1=histogram(success_deck)
edges1 = hh1.BinEdges';
counts1 = hh1.BinCounts';
values1 = hh1.Values';
group_m=size(values1,1)
for jj=1:group_m
    
end
hold on
hh2=histogram(all_deck)
edges2 = hh2.BinEdges';
counts2 = hh2.BinCounts';
values2 = hh2.Values';
%%  calculate the accuracy for costal and noncoastal environemt
costalIndex=Error_index;
costal_m=size(costalIndex,1);
success_costal=[];
all_costal=zeros(costal_m,1);
Xcoastal={};
Xnoncoastal={};
Ycoastal=[];
Ynoncoastal=[];
mn=0;
mn2=0;
for i=1:costal_m
    if Bridge_series{i, 1}{end, 10} >0 % if bigger than 0, the bridge is belong to the coastal area
         all_costal(i)=1; 
         mn=mn+1;
         Xcoastal{mn,1}=cell2mat(Bridge_series{i,1}(:,2:end))';
         Ycoastal(mn,1)=YTrain_new(i);
    else
         mn2=mn2+1;
         Xnoncoastal{mn2,1}=cell2mat(Bridge_series{i,1}(:,2:end))';
         Ynoncoastal(mn2,1)=YTrain_new(i);        
    end
    if costalIndex(i)==1 
        if Bridge_series{i, 1}{end, 10} >0
            success_costal=[success_costal; 1];
        else
            success_costal=[success_costal; 0];
        end
    end
end
Ycoastal=categorical(Ycoastal);
[Xcoastal1,Ycoastal1]=Sort_Data(Xcoastal,Ycoastal) % sort data
coastalPred=classify(Trainingnet, Xcoastal1,...
    'SequenceLength','longest');
figure
plotconfusion(Ycoastal1,coastalPred,'Coastal Accuracy')

% for non_coastal
Ynoncoastal=categorical(Ynoncoastal);
[Xnoncoastal1,Ynoncoastal1]=Sort_Data(Xnoncoastal,Ynoncoastal) % sort data
noncoastalPred=classify(Trainingnet, Xnoncoastal1,...
    'SequenceLength','longest');
figure
plotconfusion(Ynoncoastal1,noncoastalPred,'noncoastalCoastal Accuracy')


%% The distribution of each factor
num_dis=numel(Bridge_series)
Dis_all=[];
for ii=1:num_dis
    temp=cell2mat(Bridge_series{ii,1}(:,5:10));
    temp2=mean(temp);
    Dis_all=[Dis_all;temp2];
end
