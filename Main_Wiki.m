function Main_Wiki()

nbits_set=[8, 16, 32, 64, 128];

%% load dataset
load('wiki_data.mat');
XTrain = I_tr; YTrain = T_tr; LTrain = L_tr;
XTest = I_te; YTest = T_te; LTest = L_te;

%% initialization
fprintf('initializing...\n')
param.lambdaX = 0.5;
param.alpha = 500;
param.Xmu = 1000;
param.gamma = 5;
param.iter = 20;

%% centralization
fprintf('centralizing data...\n');
XTest = bsxfun(@minus, XTest, mean(XTrain, 1)); XTrain = bsxfun(@minus, XTrain, mean(XTrain, 1));
YTest = bsxfun(@minus, YTest, mean(YTrain, 1)); YTrain = bsxfun(@minus, YTrain, mean(YTrain, 1));

%% kernelization
fprintf('kernelizing...\n\n');
[XKTrain,XKTest]=Kernelize(XTrain,XTest); [YKTrain,YKTest]=Kernelize(YTrain,YTest);
XKTest = bsxfun(@minus, XKTest, mean(XKTrain, 1)); XKTrain = bsxfun(@minus, XKTrain, mean(XKTrain, 1));
YKTest = bsxfun(@minus, YKTest, mean(YKTrain, 1)); YKTrain = bsxfun(@minus, YKTrain, mean(YKTrain, 1));

%% evaluation
for bit=1:length(nbits_set) 
    nbits=nbits_set(bit);
    
    %% SCRATCH
    param.nbits=nbits;
    eva_info =evaluate(XKTrain,YKTrain,XKTest,YKTest,LTest,LTrain,param);
    
    % train time
    trainT = eva_info.trainT;
    
    % MAP
    Image_to_Text_MAP = eva_info.Image_to_Text_MAP;
    Text_to_Image_MAP=eva_info.Text_to_Image_MAP;
    
    fprintf('SCRATCH %d bits --  Image_to_Text_MAP: %f ; Text_to_Image_MAP: %f ; train time: %f\n\n',nbits,Image_to_Text_MAP,Text_to_Image_MAP,trainT);

end

end
