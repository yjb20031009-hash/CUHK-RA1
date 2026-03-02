function [ ggvalue , gvalue , betamat ] = my_estimation_prepost(myparam,varargin)
%% 重要：本函数是内置函数，不需要单独使用，也没有关键参数

%% 函数说明
% 本函数有两个功能
% 功能1（主要）：属于my_estimation_prepostdid1等函数的内置函数，不单独使用
% 功能2：单独使用，估计全样本在横截面下的参数
% 目前只用到了功能1，即内置于my_estimation_prepostdid1等函数使用


%% 预先载入设定好的参
stept = 2;

tb     = 20/stept; %家户开始的年龄
tr     = 62/stept; %家户退休的年龄
td     = 100/stept; %家户最大的年龄（最后一期）
tn     = td-tb+1; %家户最长的存活时间

global ncash nh incaa incb1 incb2 incb3
%ncash  = 11;
%nh     = 6; % housing购买量 网格分组

adjcost = 0.07; %住房的调整成本 7%
ret_fac     = 0.6; %退休后的收入因子
maxhouse  = 40.0;
minhouse  = 0.00;  %房产的最小需清偿额度
%minhouse2 = 2.8184;  %房产最小可购买门槛
minhouse2_value   = 250000*0; %房产最小可购买金额，25万人民币。
minhouse2         = minhouse2_value/(exp( incaa + incb1*60 + incb2*60^2 + incb3*60^3 )+exp( incaa + incb1*61 + incb2*61^2 + incb3*61^3 )); %技术处理，将最低购买金额标准化，等于2.8184

maxcash     = 40.0; 
mincash     = 0.25;

% 实验组对照组 全样本

%% 跟fminsearch有关的参数

ppcost = myparam(1);
otcost = myparam(2);
rho    = myparam(3);
delta  = myparam(4);
psi    = myparam(5);


%% 区分是实验前还是试验后
%ppt = 0.01;
if nargin == 1 || (nargin >= 2 && length(varargin)==2) %假如没有varargin，导入正常样本;或者varargin有且仅有两个值时，导入正常样本
load Sample_prepost.mat
else
load sim_mySample2.mat
mySample = sim_mySample;
end


%% 打格子（for 状态变量）
l_maxcash = log(maxcash);
l_mincash = log(mincash);
stepcash = (l_maxcash-l_mincash)/(ncash-1); %最大最小财富之差除以50

l_maxhouse = log(maxhouse+1);
l_minhouse = log(minhouse+1);
stephouse = (l_maxhouse-l_minhouse)/(nh-1); %最大最小房产之差除以50

for i1=1:ncash
   lgcash(i1,1)=l_mincash+(i1-1.0)*stepcash;  %把财富按ln切割成50个区间，lgcash是50*1
end
for i1=1:ncash
   gcash(i1,1)=exp(lgcash(i1,1)); %对lgcash取指数，50*1
end

for i1=1:nh
   lghouse(i1,1)=l_minhouse+(i1-1.0)*stephouse;   
end
for i1=1:nh
   ghouse(i1,1)=exp(lghouse(i1,1))-1;  
end





%% 求Policy Function
% 房产税之前
ppt = 0.00;

if nargin >= 2 && (length(varargin)==2  || (length(varargin)==4 && varargin{1}==0)) % 如果期望收益率被参数化了，且需要重新计算Policy function

[ C, A, H, C1, A1, H1] = ...,
mymain_se(ppt,ppcost,otcost,rho,delta,psi,varargin{length(varargin)-2},varargin{length(varargin)-1});
%[ Cp, Ap, Hp, C1p, A1p, H1p] = ...,
%mymain_se(ppt,ppcost,otcost,rho,delta,psi,varargin{length(varargin)-2},varargin{length(varargin)});
save PFunction_prepostdid1_pre.mat C  A  H  C1  A1  H1 %Cp  Ap  Hp  C1p  A1p H1p

elseif nargin >= 2 && (length(varargin)==4||length(varargin)==1) && varargin{1}>=1 % 参数化 且用sim data 且 不计算pfunction
load PFunction_prepostdid1_pre.mat

else
[ C, A, H, C1, A1, H1] = ...,
mymain_se(ppt,ppcost,otcost,rho,delta,psi);
save PFunction_prepostdid1_pre.mat C  A  H  C1  A1  H1

end



%% 神经网络加密格点
%{
% Combine inputs into a single matrix
% Generate all possible combinations
%load PFunction_prepostdid1_pre.mat
[gcash_mesh, ghouse_mesh, t_mesh] = meshgrid(ghouse,gcash,[1:tn]);

[gcash_mesh, ghouse_mesh] = meshgrid(ghouse,gcash);


% Convert the combinations to a matrix
inputs = [gcash_mesh(:), ghouse_mesh(:), t_mesh(:)];

inputs = [gcash_mesh(:), ghouse_mesh(:)];

% Combine outputs into a single matrix
outputs = [A(:), H(:), C(:),A1(:), H1(:), C1(:)];

predictedA=zeros(ncash,ncash,41);
predictedH=zeros(ncash,ncash,41);
predictedC=zeros(ncash,ncash,41);

for i=1:41
    AA=A(:,:,i);
    HH=H(:,:,i);
    CC=C(:,:,i);
    AA1=A1(:,:,i);
    HH1=H1(:,:,i);
    CC1=C1(:,:,i);
    outputs = [AA(:), HH(:), CC(:),AA1(:), HH1(:), CC1(:)];
    net = train(net, inputs', outputs','useGPU','yes');
    predictedOutputs = net(inputs');
    predictedA(:,:,i) = reshape(predictedOutputs(1, :), size(AA));
    predictedH(:,:,i) = reshape(predictedOutputs(2, :), size(HH));
    predictedC(:,:,i) = reshape(predictedOutputs(3, :), size(CC));
end




% Create and configure the neural network
net = feedforwardnet([50, 50]);  % Specify the architecture of the network
net.trainFcn = 'trainscg';  % Specify the training algorithm (Levenberg-Marquardt)
net.divideFcn = 'dividerand';  % Divide the data randomly for training, validation, and testing
net.trainParam.lr = 0.05; % learning rate
net.trainParam.mc = 0.9; % momentum
net.divideParam.trainRatio = 0.7;  % 70% of the data for training
net.divideParam.valRatio = 0.15;  % 15% of the data for validation
net.divideParam.testRatio = 0.15;  % 15% of the data for testing

% Train the neural network
tic
net = train(net, inputs', outputs','useGPU','yes');
toc

% Use the trained network for prediction
predictedOutputs = net(inputs');

% Reshape the predicted outputs to match the original matrices
%{
predictedA = reshape(predictedOutputs(:, 1), size(A));
predictedH = reshape(predictedOutputs(:, 2), size(H));
predictedC = reshape(predictedOutputs(:, 3), size(C));
%}
predictedA = reshape(predictedOutputs(1, :), size(A));
predictedH = reshape(predictedOutputs(2, :), size(H));
predictedC = reshape(predictedOutputs(3, :), size(C));


A(:,:,6)
predictedA(:,:,6)

H(:,:,6)
predictedH(:,:,6)

C(:,:,6)
predictedC(:,:,6)

subplot(1,2,1)
mesh(C(:,:,20))
subplot(1,2,2)
mesh(predictedC(:,:,20))

subplot(1,2,1)
mesh(A(:,:,20))
subplot(1,2,2)
mesh(predictedA(:,:,20))

%}



%% 根据policy function和mySample，计算3*3种情况下的模拟值
n = 3;
nn = n^2;
[grid,weig2] = tauchenHussey(n,0,0,1,1);
weig = weig2(1,:)';


corr_hs     = -0.08; % 股票和房产之间的相关性

r                 = 1.05 - 0.048; %2年的无风险利率 减去样本期间全国平均通胀（cpi）的2倍

if nargin >= 2 && length(varargin)>=2 %如果声明了两类资产的回报预期
    mu          = varargin{length(varargin)-2}; %股票超额回报
    muh         = varargin{length(varargin)-1};
else
    mu          = 0.08; %股票超额回报
    muh         = 0.08;
end


sigr        = 0.42; %股票波动率
sigrh       = 0.28; %房产波动率

gret        = zeros(n,1);
greth        = zeros(n,n);
gret_sh      = zeros(nn,3);

for i1=1:n
    gret(i1,1) = r+mu+grid(i1,1)*sigr; %模拟的风险资产回报
end

for i1=1:n
    grid2(:,1) = grid(i1,1)*corr_hs+grid(:,1).*ones(n,1)*(1-corr_hs^2)^(0.5);
    greth(:,i1) = r+muh+grid2(:,1)*sigrh; %模拟的房产回报
end
greth = reshape(greth,nn,1);

for i1=1:nn
    gret_sh(i1,1)=gret(ceil((i1)/n),1);
    gret_sh(i1,2)=greth(i1);
    gret_sh(i1,3)=weig(ceil((i1)/n))*weig(mod(i1-1,n)+1);
end



















%% Simulation
if nargin == 1 || (nargin >= 2 && length(varargin)==2) %假如没有varargin，或者varargin有且仅有两个值时，说明是横截面，这里取三组人
mySample1 = mySample(mySample(:,8)==0|mySample(:,7)==0,:);
else % 如果是did 这里导入的是模拟值 只需要选取 非上海即可
mySample1 = mySample(mySample(:,7)==0,:);
end

l = length(mySample1);


gyp = zeros(tn,1);

for t=1:tn
if t>=tr-tb+1
    gyp(t,1) = 1;
else
    f_y1   = exp(incaa+incb1*(stept*(t+tb-1))+incb2*(stept*(t+tb-1))^2+incb3*(stept*(t+tb-1))^3);
    f_y1_2 = exp(incaa+incb1*(stept*(t+tb-1)-1)+incb2*(stept*(t+tb-1)-1)^2+incb3*(stept*(t+tb-1)-1)^3);
    f_y2   = exp(incaa+incb1*(stept*(t+tb))+incb2*(stept*(t+tb))^2+incb3*(stept*(t+tb))^3);
    f_y2_2 = exp(incaa+incb1*(stept*(t+tb)-1)+incb2*(stept*(t+tb)-1)^2+incb3*(stept*(t+tb)-1)^3);

    gyp(t,1)  = exp((f_y2+f_y2_2)/(f_y1+f_y1_2)-1.0);
end
end


otcost_t = zeros(tn,1);
ppcost_t = zeros(tn,1);
minhouse2_t = zeros(tn,1);

otcost_t(tn,1) = otcost;
ppcost_t(tn,1) = ppcost;
minhouse2_t(tn,1) = minhouse2;

for t=tn-1:-1:1
otcost_t(t,1) = otcost_t(t+1,1)*gyp(t);
ppcost_t(t,1) = ppcost_t(t+1,1)*gyp(t);
minhouse2_t(t,1)= minhouse2_t(t+1)*gyp(t);
end




% 设置初值
initialW     = mySample1(:,4);
initialH     = mySample1(:,5);
initialAge   = floor((mySample1(:,3)-20)/2)+1;
initialAge(find(initialAge<1),1) = 1;
initialAge(find(initialAge>40),1) = 40;
initialIpart = mySample1(:,6);

simC = zeros(l,1);
simA = zeros(l,1);
simH = zeros(l,1);
simS = zeros(l,1);

simW      = zeros(l,9);
simH2     = zeros(l,9);
simAge    = initialAge+1;
simI      = zeros(l,1);

simAp1   = zeros(l,9);
simCp1   = zeros(l,9);



for i = 1:l
t = initialAge(i,1);

        if t > tr-tb
       cash  = initialW(i,1) + ret_fac;
        else
       cash  = initialW(i,1) + 1;
        end
       house = initialH(i,1);

        if initialIpart(i,1) == 0
       simC(i,1) = interp2(ghouse,gcash,C1(:,:,t),house,cash);
       simA(i,1) = interp2(ghouse,gcash,A1(:,:,t),house,cash);
       simH(i,1) = interp2(ghouse,gcash,H1(:,:,t),house,cash);

       simAa = interp2(ghouse,gcash,A1(:,:,t),house,cash,"nearest");
       %simHa = interp2(ghouse,gcash,H1(:,:,t),house,cash,"nearest");

       simA(i,1)= (1 - (simAa==0)) *  simA(i,1);
       %simH(i,1)= (1 - (simHa==0)) *  simH(i,1);

        else
       simC(i,1) = interp2(ghouse,gcash,C(:,:,t),house,cash);
       simA(i,1) = interp2(ghouse,gcash,A(:,:,t),house,cash);
       simH(i,1) = interp2(ghouse,gcash,H(:,:,t),house,cash);

       simAa = interp2(ghouse,gcash,A(:,:,t),house,cash,"nearest");
       %simHa = interp2(ghouse,gcash,H(:,:,t),house,cash,"nearest");

       simA(i,1)= (1 - (simAa==0)) *  simA(i,1);
       %simH(i,1)= (1 - (simHa==0)) *  simH(i,1);
        end

        %如果Ipart=0且当期参加了股票，则取1；如果Ipart=1，则也取1
        if (initialIpart(i,1) == 0 && simA(i,1)>0) || initialIpart(i,1) == 1
            simI(i,1) = 1;
        end

        if abs((simH(i,1)-house)) <= 0.05*house || (house==0.0 && simH(i,1)<minhouse2_t(t,1)*0.9)
            simH(i,1) = house;
        end

        if house==0.0 && simH(i,1)<minhouse2_t(t,1) && simH(i,1)>=minhouse2_t(t,1)*0.9
            simH(i,1) = minhouse2_t(t,1);
        end



        con1 = simH(i,1) == house ;
        con2 = simI(i,1)-initialIpart(i,1) == 1;
        con3 = simA(i,1)>0;

    simS(i,1) = cash + house*(1-ppt-(1-con1)*adjcost)-simC(i,1)-simH(i,1) - con2*otcost_t(t,1) - con3.*ppcost_t(t,1);



    simW(i,:)  = (simS(i,1)*simA(i,1)*gret_sh(:,1)'+simS(i,1)*(1-simA(i,1))*r*ones(1,nn))/gyp(t,1);
    simH2(i,:) = simH(i,1)*gret_sh(:,2)'/gyp(t,1);

    simW(i,:) = min(simW(i,:),maxcash);
    simH2(i,:) = min(simH2(i,:),maxhouse);

% 接下来用T+1的状态变量代入Policy Function，求解T+1期的Choice
t =  simAge(i,1);

        if t > tr-tb
       cash  = simW(i,:) + ret_fac;
        else
       cash  = simW(i,:) + 1;
        end
       house = simH2(i,:);

        if simI(i,1) == 0
       simCp1(i,:) = interp2(ghouse,gcash,C1(:,:,t),house,cash);
       simAp1(i,:) = interp2(ghouse,gcash,A1(:,:,t),house,cash);

       simAa = interp2(ghouse,gcash,A1(:,:,t),house,cash,"nearest");
       simAp1(i,simAa==0) = 0;

        else
       simCp1(i,:) = interp2(ghouse,gcash,C(:,:,t),house,cash);
       simAp1(i,:) = interp2(ghouse,gcash,A(:,:,t),house,cash);

       simAa = interp2(ghouse,gcash,A(:,:,t),house,cash,"nearest");
       simAp1(i,simAa==0) = 0;

        end


end

if nargin == 1 || (nargin >= 2 && length(varargin)==2) %假如没有varargin，或者varargin有且仅有两个值时，进行简单回归（横截面）

    for k = 1:nn
        x_sim1(:,:,k) = [ones(l,1) , (mySample1(:,3)+2) , (mySample1(:,3)+2).^2  , simW(:,k) , simW(:,k).^2  , simH2(:,k) , simH2(:,k).^2  , simI ,mySample1(:,7),mySample1(:,8)];
        %x_sim1(:,:,k) = [ones(l,1) , (mySample1(:,3)+2) , simW(:,k) ,   simH2(:,k) , simI ];
        y1_1(:,k) = simCp1(:,k);
        y2_1(:,k) = simAp1(:,k)>0;
        y3_1(:,k) = simAp1(:,k) ;

    end

else %（DID）

    for k = 1:nn
        x_sim1(:,:,k) = [ones(l,1) , (mySample1(:,3)+2) ,(mySample1(:,3)+2).^2  , simW(:,k) ,simW(:,k).^2  ,   simH2(:,k) , simH2(:,k).^2  , simI , zeros(l,1) , ones(l,1) , zeros(l,1)];
        y1_1(:,k) = simCp1(:,k);
        y2_1(:,k) = simAp1(:,k)>0;
        y3_1(:,k) = simAp1(:,k) ;

    end

% 保存模拟的结果1
sim_mySample = [];
sim_mySample(:,1) = simCp1(:,:) * gret_sh(:,3);
sim_mySample(:,2) = (simAp1(:,:)>0) * gret_sh(:,3);
sim_mySample(:,3) = (mySample1(:,3)+2);
sim_mySample(:,4) =   simW(:,:) * gret_sh(:,3);
sim_mySample(:,5) =  simH2(:,:) * gret_sh(:,3);
sim_mySample(:,6) =  simI(:,1);
sim_mySample(:,7) =  (mySample1(:,7));
sim_mySample(:,8) =  1;
sim_mySample(:,9) =  (mySample1(:,9));
sim_mySample(:,10) =  sim_mySample(:,4)./(sim_mySample(:,4)+sim_mySample(:,5)); % fin ratio
sim_mySample(:,11) =  (simAp1(:,:)) * gret_sh(:,3);
%save sim_mySample2.mat sim_mySample
sim_mySample_p1 = sim_mySample;

end


%% 以下是估计有房产税的部分

%% 求Policy Function
% 房产税之后
%if length(myparam)>7
%    ppt = myparam(8)*2;
%else
    ppt = 0.008;
%end

if nargin >= 2 && (length(varargin)==2  || (length(varargin)==4 && varargin{1}==0)) % 如果期望收益率被参数化了，且需要重新估Policy Function(因为加入了房产税)

[ C, A, H, C1, A1, H1] = ...,
mymain_se(ppt,ppcost,otcost,rho,delta,psi,varargin{length(varargin)-2},varargin{length(varargin)});
%[ Cp, Ap, Hp, C1p, A1p, H1p] = ...,
%mymain_se(ppt,ppcost,otcost,rho,delta,psi,varargin{length(varargin)-2},varargin{length(varargin)});
save PFunction_prepostdid1_post.mat C  A  H  C1  A1  H1 %Cp  Ap  Hp  C1p  A1p H1p


elseif nargin >= 2 && (length(varargin)==4||length(varargin)==1) && varargin{1}>=1 % 参数化 且用sim data 且 不计算pfunction
load PFunction_prepostdid1_post.mat

else
[ C, A, H, C1, A1, H1] = ...,
mymain_se(ppt,ppcost,otcost,rho,delta,psi);
save PFunction_prepostdid1_post.mat C  A  H  C1  A1  H1

end



%% 根据policy function和mySample，计算3*3种情况下的模拟值
n = 3;
nn = n^2;
[grid,weig2] = tauchenHussey(n,0,0,1,1);
weig = weig2(1,:)';


corr_hs     = -0.08; % 股票和房产之间的相关性

r                 = 1.05 - 0.048; %2年的无风险利率 减去样本期间全国平均通胀（cpi）的2倍

if nargin >= 2 && length(varargin)>=2 %如果声明了两类资产的回报预期
    mu          = varargin{length(varargin)-2}; %股票超额回报
    muh         = varargin{length(varargin)-1};

else
    mu          = 0.08; %股票超额回报
    muh         = 0.08;
end

sigr        = 0.42; %股票波动率
sigrh       = 0.28; %房产波动率

gret        = zeros(n,1);
greth        = zeros(n,n);
gret_sh      = zeros(nn,3);

for i1=1:n
    gret(i1,1) = r+mu+grid(i1,1)*sigr; %模拟的风险资产回报
end

for i1=1:n
    grid2(:,1) = grid(i1,1)*corr_hs+grid(:,1).*ones(n,1)*(1-corr_hs^2)^(0.5);
    greth(:,i1) = r+muh+grid2(:,1)*sigrh; %模拟的房产回报
end
greth = reshape(greth,nn,1);

for i1=1:nn
    gret_sh(i1,1)=gret(ceil((i1)/n),1);
    gret_sh(i1,2)=greth(i1);
    gret_sh(i1,3)=weig(ceil((i1)/n))*weig(mod(i1-1,n)+1);
end
% 0904a



%{
gret        = zeros(n,1);
greth        = zeros(n,n);
gret_sh_p      = zeros(nn,3);
for i1=1:n
    gret(i1,1) = r+mu+grid(i1,1)*sigr; %模拟的风险资产回报
end

for i1=1:n
    grid2(:,1) = grid(i1,1)*corr_hs+grid(:,1).*ones(n,1)*(1-corr_hs^2)^(0.5);
    greth(:,i1) = r+muhp+grid2(:,1)*sigrh; %模拟的房产回报
end
greth = reshape(greth,nn,1);

for i1=1:nn
    gret_sh_p(i1,1)=gret(ceil((i1)/n),1);
    gret_sh_p(i1,2)=greth(i1);
    gret_sh_p(i1,3)=weig(ceil((i1)/n))*weig(mod(i1-1,n)+1);
end
%}


















%% Simulation
if nargin == 1 || (nargin >= 2 && length(varargin)==2) %假如没有varargin，或者varargin有且仅有两个值时，进行简单回归
mySample2 = mySample(mySample(:,8)==1&mySample(:,7)==1,:); % Post##Treat
else
mySample2 = mySample(mySample(:,7)==1,:); 
end

l = length(mySample2);

gyp = zeros(tn,1);

for t=1:tn
if t>=tr-tb+1
    gyp(t,1) = 1;
else
    f_y1   = exp(incaa+incb1*(stept*(t+tb-1))+incb2*(stept*(t+tb-1))^2+incb3*(stept*(t+tb-1))^3);
    f_y1_2 = exp(incaa+incb1*(stept*(t+tb-1)-1)+incb2*(stept*(t+tb-1)-1)^2+incb3*(stept*(t+tb-1)-1)^3);
    f_y2   = exp(incaa+incb1*(stept*(t+tb))+incb2*(stept*(t+tb))^2+incb3*(stept*(t+tb))^3);
    f_y2_2 = exp(incaa+incb1*(stept*(t+tb)-1)+incb2*(stept*(t+tb)-1)^2+incb3*(stept*(t+tb)-1)^3);

    gyp(t,1)  = exp((f_y2+f_y2_2)/(f_y1+f_y1_2)-1.0);
end
end


otcost_t = zeros(tn,1);
ppcost_t = zeros(tn,1);
minhouse2_t = zeros(tn,1);

otcost_t(tn,1) = otcost;
ppcost_t(tn,1) = ppcost;
minhouse2_t(tn,1) = minhouse2;

for t=tn-1:-1:1
otcost_t(t,1) = otcost_t(t+1,1)*gyp(t);
ppcost_t(t,1) = ppcost_t(t+1,1)*gyp(t);
minhouse2_t(t,1)= minhouse2_t(t+1)*gyp(t);
end




% 设置初值
initialW     = mySample2(:,4);
initialH     = mySample2(:,5);
initialAge   = floor((mySample2(:,3)-20)/2)+1;
initialAge(find(initialAge<1),1) = 1;
initialAge(find(initialAge>40),1) = 40;
initialIpart = mySample2(:,6);

simC = zeros(l,1);
simA = zeros(l,1);
simH = zeros(l,1);
simS = zeros(l,1);

simW      = zeros(l,9);
simH2     = zeros(l,9);
simAge    = initialAge+1;
simI      = zeros(l,1);

simAp1   = zeros(l,9);
simCp1   = zeros(l,9);



for i = 1:l
t = initialAge(i,1);

        if t > tr-tb
       cash  = initialW(i,1) + ret_fac;
        else
       cash  = initialW(i,1) + 1;
        end
       house = initialH(i,1);

        if initialIpart(i,1) == 0
       simC(i,1) = interp2(ghouse,gcash,C1(:,:,t),house,cash);
       simA(i,1) = interp2(ghouse,gcash,A1(:,:,t),house,cash);
       simH(i,1) = interp2(ghouse,gcash,H1(:,:,t),house,cash);

       simAa = interp2(ghouse,gcash,A1(:,:,t),house,cash,"nearest");
       simHa = interp2(ghouse,gcash,H1(:,:,t),house,cash,"nearest");

       simA(i,1)= (1 - (simAa==0)) *  simA(i,1);
       simH(i,1)= (1 - (simHa==0)) *  simH(i,1);

        else
       simC(i,1) = interp2(ghouse,gcash,C(:,:,t),house,cash);
       simA(i,1) = interp2(ghouse,gcash,A(:,:,t),house,cash);
       simH(i,1) = interp2(ghouse,gcash,H(:,:,t),house,cash);

       simAa = interp2(ghouse,gcash,A(:,:,t),house,cash,"nearest");
       simHa = interp2(ghouse,gcash,H(:,:,t),house,cash,"nearest");

       simA(i,1)= (1 - (simAa==0)) *  simA(i,1);
       simH(i,1)= (1 - (simHa==0)) *  simH(i,1);
        end

        %如果Ipart=0且当期参加了股票，则取1；如果Ipart=1，则也取1
        if (initialIpart(i,1) == 0 && simA(i,1)>0) || initialIpart(i,1) == 1
            simI(i,1) = 1;
        end

        if abs((simH(i,1)-house)) <= 0.05*house || (house==0.0 && simH(i,1)<minhouse2_t(t,1)*0.9)
            simH(i,1) = house;
        end

        if house==0.0 && simH(i,1)<minhouse2_t(t,1) && simH(i,1)>=minhouse2_t(t,1)*0.9
            simH(i,1) = minhouse2_t(t,1);
        end



        con1 = simH(i,1) == house ;
        con2 = simI(i,1)-initialIpart(i,1) == 1;
        con3 = simA(i,1)>0;

    simS(i,1) = cash + house*(1-ppt-(1-con1)*adjcost)-simC(i,1)-simH(i,1) - con2*otcost_t(t,1) - con3.*ppcost_t(t,1);



    simW(i,:)  = (simS(i,1)*simA(i,1)*gret_sh(:,1)'+simS(i,1)*(1-simA(i,1))*r*ones(1,nn))/gyp(t,1);
    simH2(i,:) = simH(i,1)*gret_sh(:,2)'/gyp(t,1);

% 接下来用T+1的状态变量代入Policy Function，求解T+1期的Choice
t =  simAge(i,1);

        if t > tr-tb
       cash  = simW(i,:) + ret_fac;
        else
       cash  = simW(i,:) + 1;
        end
       house = simH2(i,:);

        if simI(i,1) == 0
       simCp1(i,:) = interp2(ghouse,gcash,C1(:,:,t),house,cash);
       simAp1(i,:) = interp2(ghouse,gcash,A1(:,:,t),house,cash);

       simAa = interp2(ghouse,gcash,A1(:,:,t),house,cash,"nearest");
       simAp1(i,simAa==0) = 0;

        else
       simCp1(i,:) = interp2(ghouse,gcash,C(:,:,t),house,cash);
       simAp1(i,:) = interp2(ghouse,gcash,A(:,:,t),house,cash);

       simAa = interp2(ghouse,gcash,A(:,:,t),house,cash,"nearest");
       simAp1(i,simAa==0) = 0;

        end

end




if nargin == 1 || (nargin >= 2 && length(varargin)==2) %假如没有varargin，或者varargin有且仅有两个值时，did

    for k = 1:nn
        x_sim2(:,:,k) = [ones(l,1) , (mySample2(:,3)+2) , (mySample2(:,3)+2).^2  , simW(:,k) , simW(:,k).^2  , simH2(:,k) , simH2(:,k).^2  , simI ];
 
        y1_2(:,k) = simCp1(:,k);
        y2_2(:,k) = simAp1(:,k)>0;
        y3_2(:,k) = simAp1(:,k) ;

    end


else %DID


    for k = 1:nn
        x_sim2(:,:,k) = [ones(l,1) , (mySample2(:,3)+2) ,(mySample2(:,3)+2).^2  , simW(:,k) ,simW(:,k).^2  ,   simH2(:,k) , simH2(:,k).^2  , simI , ones(l,1) , ones(l,1), ones(l,1)];
        y1_2(:,k) = simCp1(:,k);
        y2_2(:,k) = simAp1(:,k)>0;
        y3_2(:,k) = simAp1(:,k) ;
    end

% 保存模拟的结果2
sim_mySample = [];
sim_mySample(:,1) = simCp1(:,:) * gret_sh(:,3);
sim_mySample(:,2) = (simAp1(:,:)>0) * gret_sh(:,3);
sim_mySample(:,3) = (mySample2(:,3)+2);
sim_mySample(:,4) =   simW(:,:) * gret_sh(:,3);
sim_mySample(:,5) =  simH2(:,:) * gret_sh(:,3);
sim_mySample(:,6) =  simI(:,1);
sim_mySample(:,7) =  (mySample2(:,7));
sim_mySample(:,8) =  1;
sim_mySample(:,9) =  (mySample2(:,9));
sim_mySample(:,10) =  sim_mySample(:,4)./(sim_mySample(:,4)+sim_mySample(:,5)); % fin ratio
sim_mySample(:,11) =  (simAp1(:,:)) * gret_sh(:,3);

sim_mySample = [sim_mySample_p1;sim_mySample];
save sim_mySample2.mat sim_mySample 

end



%% 求两组回归系数
if nargin == 1 || (nargin >= 2 && length(varargin)==2) %假如没有varargin，或者varargin有且仅有两个值时，进行简单回归
% 模拟数据的回归系数
beta1_sim = zeros(10,nn);
beta2_sim = zeros(10,nn);
beta3_sim = zeros(10,nn);
beta4_sim = zeros(8,nn);
beta5_sim = zeros(8,nn);
beta6_sim = zeros(8,nn);

for k = 1:nn

beta1_sim(:,k) = (x_sim1(:,:,k)'*x_sim1(:,:,k))\x_sim1(:,:,k)'*y1_1(:,k);
beta2_sim(:,k) = (x_sim1(:,:,k)'*x_sim1(:,:,k))\x_sim1(:,:,k)'*y2_1(:,k);
beta3_sim(:,k) = (x_sim1(:,:,k)'*x_sim1(:,:,k))\x_sim1(:,:,k)'*y3_1(:,k);

beta4_sim(:,k) = (x_sim2(:,:,k)'*x_sim2(:,:,k))\x_sim2(:,:,k)'*y1_2(:,k);
beta5_sim(:,k) = (x_sim2(:,:,k)'*x_sim2(:,:,k))\x_sim2(:,:,k)'*y2_2(:,k);
beta6_sim(:,k) = (x_sim2(:,:,k)'*x_sim2(:,:,k))\x_sim2(:,:,k)'*y3_2(:,k);

end

beta1_sim_mean = beta1_sim * gret_sh(:,3);
beta2_sim_mean = beta2_sim * gret_sh(:,3);
beta3_sim_mean = beta3_sim * gret_sh(:,3);
beta4_sim_mean = beta4_sim * gret_sh(:,3);
beta5_sim_mean = beta5_sim * gret_sh(:,3);
beta6_sim_mean = beta6_sim * gret_sh(:,3);

beta1_sim_mean([9,10])=[];
beta2_sim_mean([9,10])=[];
beta3_sim_mean([9,10])=[];

% 现实数据的回归系数
%beta1 = x'*x\x'*mySample(:,1);
%beta2 = x'*x\x'*mySample(:,2);
betamat =  [[beta1;beta2;beta3;beta4;beta5;beta6],[beta1_sim_mean;beta2_sim_mean;beta3_sim_mean;beta4_sim_mean;beta5_sim_mean;beta6_sim_mean]];

gvalue   = [beta1;beta2;beta3;beta4;beta5;beta6]-[beta1_sim_mean;beta2_sim_mean;beta3_sim_mean;beta4_sim_mean;beta5_sim_mean;beta6_sim_mean];

ggvalue  = gvalue'*W*gvalue;
end






if nargin >= 2 && ( length(varargin)==1 || length(varargin)==4 ) %如果varargin是1和3 且第一个元素为2 说明是did计算
%{
load Sample_did_nosample.mat %我们只需要beta和W
load sim_mySample_append.mat
mySample = sim_mySample_app;

beta1_sim = zeros(11,nn);
beta2_sim = zeros(11,nn);
beta3_sim = zeros(11,nn);

l = length(mySample);

x_sim = [ ones(l,1) , mySample(:,3) , mySample(:,3).^2  , mySample(:,4) , mySample(:,4).^2 , mySample(:,5) , mySample(:,5).^2  , mySample(:,6), mySample(:,7) , mySample(:,8) ,mySample(:,7).*mySample(:,8)];
y1 = mySample(:,1);
y2 = mySample(:,2);
y3 = mySample(:,11);

beta1_sim_mean = (x_sim'*x_sim)\x_sim'*y1;
beta2_sim_mean = (x_sim'*x_sim)\x_sim'*y2;
beta3_sim_mean = (x_sim'*x_sim)\x_sim'*y3;

beta1_sim_mean([9,10])=[];
beta2_sim_mean([9,10])=[];
beta3_sim_mean([9,10])=[];


betamat =  [[beta1;beta2;beta3],[beta1_sim_mean;beta2_sim_mean;beta3_sim_mean]];
gvalue   = [beta1;beta2;beta3]-[beta1_sim_mean;beta2_sim_mean;beta3_sim_mean];
ggvalue  = gvalue'*W*gvalue;
%}

betamat=0;
gvalue=0;
ggvalue=0;

end

end

%{
for k = 1:nn
x_sim = [x_sim1(:,:,k) ; x_sim2(:,:,k) ;
         ones(l,1) , mySample(:,3) , mySample(:,3).^2  , mySample(:,4) , mySample(:,4).^2 , mySample(:,5) , mySample(:,5).^2  , mySample(:,6), mySample(:,7) ,zeros(l,1)  ,zeros(l,1)];

y1 = [y1_1(:,k);y1_2(:,k);mySample(:,1)];
y2 = [y2_1(:,k);y2_2(:,k);mySample(:,2)];
y3 = [y2_1(:,k);y2_2(:,k);mySample(:,11)];

beta1_sim(:,k) = (x_sim'*x_sim)\x_sim'*y1;
beta2_sim(:,k) = (x_sim'*x_sim)\x_sim'*y2;
beta3_sim(:,k) = (x_sim'*x_sim)\x_sim'*y3;

end

beta1_sim_mean = beta1_sim * gret_sh(:,3);
beta2_sim_mean = beta2_sim * gret_sh(:,3);
beta3_sim_mean = beta3_sim * gret_sh(:,3);

beta1_sim_mean([9,10])=[];
beta2_sim_mean([9,10])=[];
beta3_sim_mean([9,10])=[];


betamat =  [[beta1;beta2;beta3],[beta1_sim_mean;beta2_sim_mean;beta3_sim_mean]];
gvalue   = [beta1;beta2;beta3]-[beta1_sim_mean;beta2_sim_mean;beta3_sim_mean];
ggvalue  = gvalue'*W*gvalue;


end




end

%}