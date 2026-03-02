function [ C, A, H, W, C1, A1, H1, W1 ]=picture_inner_func(ppt,ppcost,otcost,rho,delta,psi,varargin)
%% 【关键内层函数】求解policy function
% 输入一组可变的参数，通过grid search和fmincon结合的方法，求解policy function

%% 【固定参数的取值基本都在这个函数里】
% 内置于外层函数my_estimation_prepostdid1（等）中
% 输入 一组参数，包括家户偏好、参与成本及房产税等
% 输出 各个Choice（消费、股票和房产投资）的policy function。结果是一组二维矩阵，两个维度分别为期初的cash和housing，即两个状态变量
% 其中C  A  H 是【人上人】（已经pay过one-time cost的人）的policy funtion
%     C1 A1 H1是【屌丝】（没参与过股市）的policy funtion

%%
global incaa incb1 incb2 incb3
incb1 = .0427089;
incb2 = -.0216383/100;
incb3 = -.1124394/10000;
incaa = 8.191964;

%% 参数设定
% A 生命周期
stept = 2; %每一期包括2年
tb     = 20/stept; %家户开始的年龄
tr     = 62/stept; %家户退休的年龄
td     = 100/stept; %家户最大的年龄（最后一期）
tn     = td-tb+1; %家户最长的存活时间

% B 格点
global ncash nh
ncash  = 21; % 状态变量cash格点
nh     = 11; % 状态变量housing格点

na     = 11; % 股票比例格点
nc     = 11; % 消费格点
nh2    = 6; % 房产投资选择的格点

n      = 3; %将股票收益率的连续分布 离散成3种情况
nn     = n*n; %允许房产收益率也离散为3种可能性，故股票-房产收益率组合共有9种



% C 收入 （使用60岁以下家庭的收入对年龄的三阶多项式 回归 得到）
smay        = nan; %收入风险。本模型不需要
smav        = nan; %收入风险。本模型不需要
corr_v      = nan; %收入(暂时性)风险与股票风险的关系。本模型不需要
corr_y      = nan; %收入(永久性)风险与股票风险的关系。本模型不需要

ret_fac     = 0.6; %退休后的收入因子

% D 资产和现金
adjcost           = 0.07; %住房的调整成本 7%
maxhouse          = 40.0; % 房产的上界，收入的200倍。技术设定。
minhouse          = 0.00;  %不需要的设定，取0即可。如果不取0，当房产价值低于此值时，房产必须被清偿。
minhouse2_value   = 250000; %房产最小可购买金额，25万人民币。
minhouse2         = minhouse2_value/(exp( incaa + incb1*60 + incb2*60^2 + incb3*60^3 )+exp( incaa + incb1*61 + incb2*61^2 + incb3*61^3 )); %技术处理，将最低购买金额标准化，等于2.8184

maxcash           = 40.0; % cash的上界，收入的200倍。技术设定。
mincash           = 0.25; % cash的下界，收入的0.25倍。cash里包括当期收入，故不会取0。
minalpha          = 0.01; %技术处理。股票的最小投资比例。
corr_hs           = -0.08; % 股票和房产之间的相关性 -8%
r                 = 1.05 - 0.048; %2年的无风险利率 减去样本期间全国平均通胀（cpi）的2倍

if nargin==6 % 如果参数中不含股票的预期收益率
    mu          = 0.08 ; %股票 超额回报 取8%
    muh         = 0.08 ; %房产 超额回报 取8%
else %如果参数指定了股票收益率
    mu          = varargin{1}; %股票超额回报
    muh         = varargin{2};
end

sigr        = 0.42; %股票波动率（30%*sqrt(2)）
sigrh       = 0.28; %房产波动率（20%*sqrt(2)）


y60 = exp( incaa + incb1*60 + incb2*60^2 + incb3*60^3 );
y61 = exp( incaa + incb1*61 + incb2*61^2 + incb3*61^3 );
ppcost = ppcost/(y60+y61);
otcost = otcost/(y60+y61);



% E 技术处理
%       1  2  3  4  5  6     7 8  9    10      11     12     13        14
Param1=[tb,tr,td,tn,na,ncash,n,nc,nh,adjcost,ppcost,otcost,minalpha,maxhouse];
%        1         2          3     4       5  6  7 8    9       10  11  12
Param2=[minhouse,minhouse2,maxcash,mincash,incaa,incb1,incb2,incb3,ret_fac,smay,smav,corr_hs];
%         1      2     3    4    5  6 7  8    9    10    11  12  13
Param3=[corr_v,corr_y,rho,delta,psi,r,mu,muh,sigr,sigrh,ppt,nh2,stept]; 



% F 预设空矩阵
gret        = zeros(n,1);
greth        = zeros(n,n);
gret_sh      = zeros(nn,3);
riskret     = zeros(na,nn);
ones_n_1    = ones(n,1);
grid2       = zeros(n,1);
gcash       = zeros(ncash,1); %加入housing
lgcash      = zeros(ncash,1); %加入housing
ghouse       = zeros(nh,1); %加入housing
lghouse      = zeros(nh,1); %加入housing


ga          = zeros(na,1);
C           = zeros(ncash,nh,tn); 
V           = zeros(ncash,nh,tn); 
A           = ones(ncash,nh,tn);
H           = ones(ncash,nh,tn); 

C1           = zeros(ncash,nh,tn); 
V1           = zeros(ncash,nh,tn); 
A1           = ones(ncash,nh,tn);
H1           = ones(ncash,nh,tn); 

%% Conditional Survival Probabilities & Return Grid
load surv.mat %这个数据集是条件生存概率，由2010年人口普查数据计算得来

% 根据Tauchen-Hussey (1991)将标准正态分布离散为3种情况
[grid,weig2] = tauchenHussey(n,0,0,1,1);
weig = weig2(1,:)';


%% Additional Computations
for i1=1:n
    gret(i1,1) = r+mu+grid(i1,1)*sigr; %3种情况下 股票的资产回报
end

for i1=1:n
    grid2(:,1) = grid(i1,1)*corr_hs+grid(:,1).*ones_n_1(:,1)*(1-corr_hs^2)^(0.5);
    greth(:,i1) = r+muh+grid2(:,1)*sigrh; %模拟出房产回报，一共3*3种情况
end
greth = reshape(greth,nn,1);

for i1=1:nn
    gret_sh(i1,1)=gret(ceil((i1)/n),1);
    gret_sh(i1,2)=greth(i1);
    gret_sh(i1,3)=weig(ceil((i1)/n))*weig(mod(i1-1,n)+1);
end % gret_sh矩阵中，第一列是股票回报，第二列是房产回报，第三列是发生的概率

%% Grids for the State Variables and for Portfolio Rule

% 以下是对choice中的stock share打格子
% 但是choice搜索方法改成fmincon后，riskret已经没有用了，没删是防止后面出bug
for i1=1:na
   ga(i1,1)= minalpha+(i1-1)*(1-minalpha)/(na-1); 
end

for i5=1:na
   for i8=1:nn
      riskret(i5,i8)=r*(1-ga(i5,1))+gret_sh(i8,1)*ga(i5,1); 
   end
end

% 以下是对state var打格子，分别得到gcash和ghouse
% 打格子的方法是先取ln，然后对ln(cash)均匀等分，再求exp(lncash)转换回cash
l_maxcash = log(maxcash);
l_mincash = log(mincash);
stepcash = (l_maxcash-l_mincash)/(ncash-1); 

l_maxhouse = log(maxhouse+1);
l_minhouse = log(minhouse+1);
stephouse = (l_maxhouse-l_minhouse)/(nh-1); 

for i1=1:ncash
    lgcash(i1,1)=l_mincash+(i1-1.0)*stepcash;  
end
for i1=1:ncash
   gcash(i1,1)=exp(lgcash(i1,1)); 
end

for i1=1:nh
    lghouse(i1,1)=l_minhouse+(i1-1.0)*stephouse;   
end
for i1=1:nh
   ghouse(i1,1)=exp(lghouse(i1,1))-1;  
end

%% Terminal Period
for i1=1:ncash
    for i2 = 1 : nh
        C(i1,i2,tn) = gcash(i1,1)+ghouse(i2,1)*(1-adjcost-ppt); %最后一期消费等于全部Cash+House
    end
end
A(:,:,tn) = 0.0; %A是股票的ratio，最后一期等于0
H(:,:,tn) = 0.0; %H是投资性房产的量，最后一期等于0
for i1=1:ncash
   for i2 = 1 : nh
        V(i1,i2,tn) = C(i1,i2,tn)*((1.0-delta)^(psi/(psi-1.0))); % Value function
   % V = C * (1-beta)^(psi/(psi-1))
   end
end 

V1(:,:,tn) = V(:,:,tn);
C1(:,:,tn) = C(:,:,tn);
A1(:,:,tn) = A(:,:,tn);
H1(:,:,tn) = H(:,:,tn); %不管是人上人还是屌丝，临死前都是一样的

%% 参数计算
theta = (1.0-rho)/(1.0-1.0/psi); 
psi_1 = 1.0-1.0/psi;
psi_2 = 1.0/psi_1;
options = optimoptions('fmincon','Display','off','ConstraintTolerance',1e-3,'OptimalityTolerance',1e-3); % 防止fmincon的输出刷屏


%% 循环1 -- 【人上人】
% one-time cost(otcost) =0的情况下，计算policy function
ppcost = Param1(11);
minhouse2 = Param2(2);
otcost = 0;
i2=1:nh;
i1=1:ncash;
[i1n,i2n]=ndgrid(i1,i2);
result = zeros(6,5,ncash*nh);

for t = tn-1:-1:1 
    V_next = V(:,:,t+1);
    if t >= tr-tb
        income = ret_fac;
        gyp = 1;
    else
        income = 1;
        f_y1   = exp(incaa+incb1*(stept*(t+tb+1))+incb2*(stept*(t+tb+1))^2+incb3*(stept*(t+tb+1))^3);
        f_y1_2 = exp(incaa+incb1*(stept*(t+tb+1)+1)+incb2*(stept*(t+tb+1)+1)^2+incb3*(stept*(t+tb+1)+1)^3);
        f_y2   = exp(incaa+incb1*(stept*(t+tb))+incb2*(stept*(t+tb))^2+incb3*(stept*(t+tb))^3);
        f_y2_2 = exp(incaa+incb1*(stept*(t+tb)+1)+incb2*(stept*(t+tb)+1)^2+incb3*(stept*(t+tb)+1)^3);
        gyp  = exp((f_y1+f_y1_2)/(f_y2+f_y2_2)-1.0);
    end

    otcost = otcost*gyp;
    ppcost = ppcost*gyp;
    minhouse2 = minhouse2*gyp; % 将cost、最小房产购买金额 按照收入进行标准化
    
    % 先估计插值模型，再把模型传入目标函数、直接插值，比interp2更快（否则每次都要重新估计插值模型）
    [XOut, YOut, ZOut] = prepareSurfaceData(ghouse, gcash, V_next);
    [model]=fit([XOut, YOut],ZOut,'cubicinterp'); 
    
    param_cell = {t rho delta psi_1 psi_2 theta gyp model adjcost ppt ppcost otcost income nn survprob gret_sh r};% 一些需要传入目标函数的参数

parfor i=1:ncash*nh
% 对于每个状态变量的grid，都进行fmincon的搜索来找到最合适的choice
% 由于不同的choice对应不同的约束条件，所以需要分4种情况讨论

        % 1 处置房产 + 参与股票 （意味着要从cash里付adjustment cost和股票的cost）
        b1 = ghouse(i2n(i))*(1-adjcost-ppt)+gcash(i1n(i))-otcost-ppcost; %b1 is cash on hand
        choice1=[nan,nan,nan];
        choice2=[nan,nan,nan];
        value1 = 0.00;
        value2 = 0.00;
        % 下面的my_auxV_cal是目标函数，输入一组choice (consumption,alpha,housing)，输出-V
        try
        if b1>=0.25+minhouse2
            [choice1,value1] = fmincon( @(x)my_auxV_cal(x,param_cell,gcash(i1n(i)) ,ghouse(i2n(i)))  ,  ...  
                [0.25 ,0.2 ,minhouse2] ,  ... %Initial value: minComsumption; 20%Stock; minHouseValue
                [1,0,1;-1,0,0;0,0,-1;0,1,0;0,-1,0],[b1;-0.25;-minhouse2;1;-minalpha]  ,  ...
                [],[],[],[],[],options);
        end
        if b1>=0.25
            [choice2,value2] = fmincon( @(x)my_auxV_cal(x,param_cell,gcash(i1n(i)) ,ghouse(i2n(i)))   ,  ...
                [0.25 ,0.2 ,0.0]  ,  ...
                [1,0,1;-1,0,0;0,1,0;0,-1,0],[b1;-0.25;1;-minalpha]  ,  ...
                [0,0,1],[0.00],[],[],[],options); %房产取值为0
        end
        end

        % 2 处置房产 + 不参与股票
        b2 = ghouse(i2n(i))*(1-adjcost-ppt)+gcash(i1n(i));
        choice3=[nan,nan,nan];
        choice4=[nan,nan,nan];
        value3 = 0.00;
        value4 = 0.00;
        try
        if b2>=0.25+minhouse2
            [choice3,value3] = fmincon( @(x)my_auxV_cal(x,param_cell,gcash(i1n(i)) ,ghouse(i2n(i)))   ,  ...
                [0.25 ,0.0 ,minhouse2]  ,  ...
                [1,0,1;-1,0,0;0,0,-1],[b2;-0.25;-minhouse2]  ,  ...
                [0,1,0],[0.00],[],[],[],options);%不参与股票
        end
        if b2>=0.25
            [choice4,value4] = fmincon( @(x)my_auxV_cal(x,param_cell,gcash(i1n(i)) ,ghouse(i2n(i)))  ,  ...
                [0.25 ,0.0 ,0.0]  ,  ...
                [1,0,1;-1,0,0],[b2;-0.25]  ,  ...
                [0,1,0;0,0,1],[0.00;0.00],[],[],[],options); %房产取值为0 股票取值为0
        end
        end

        % 3 不处置房产 + 参与股票
        b3 = ghouse(i2n(i))*(-ppt)+gcash(i1n(i))-otcost-ppcost;
                    choice5=[nan,nan,nan];
            value5 = 0.00;
        try
        if b3>=0.25
            [choice5,value5] = fmincon( @(x)my_auxV_cal(x,param_cell,gcash(i1n(i)) ,ghouse(i2n(i)))   ,  ...
                [0.25 ,0.2 ,ghouse(i2n(i))]  ,  ...
                [1,0,0;-1,0,0;0,1,0;0,-1,0],[b3;-0.25;1;-minalpha]  ,  ...
                [0,0,1],ghouse(i2n(i)),[],[],[],options);
        end
        end

        % 4 不处置房产 + 不参与股票
        b4 = ghouse(i2n(i))*(-ppt)+gcash(i1n(i));
            choice6=[nan,nan,nan];
            value6 = 0.00;
            try
        if b4>=0.25
            [choice6,value6] = fmincon( @(x)my_auxV_cal(x,param_cell,gcash(i1n(i)) ,ghouse(i2n(i)))   ,  ...
                [0.25 ,0.0 ,ghouse(i2n(i))]  ,  ...
                [1,0,0;-1,0,0],[b4;-0.25]  ,  ...
                [0,0,1;0,1,0],[ghouse(i2n(i));0.00],[],[],[],options);
        end
        end

        result(:,:,i) = [value1,choice1,b1;value2,choice2,b1;value3,choice3,b2;value4,choice4,b2;value5,choice5,b3+ghouse(i2n(i));value6,choice6,b4+ghouse(i2n(i))];

end

for i=1:ncash*nh
index = find(result(:,1,i)==min(result(:,1,i))); % 由于函数输出的是-V，这里需要取最小值
C(i1n(i),i2n(i),t) = result(index(1),2,i);
A(i1n(i),i2n(i),t) = result(index(1),3,i);
H(i1n(i),i2n(i),t) = result(index(1),4,i);
V(i1n(i),i2n(i),t) =-result(index(1),1,i); % 由于函数输出的是-V，这里需要取负值
W(i1n(i),i2n(i),t) = result(index(1),5,i); % W is cash on hand
end


end

H(H<1e-3)=0;


%% 循环2 -- 【屌丝】
ppcost = Param1(11);
minhouse2 = Param2(2);
otcost = Param1(12);
i2=1:nh;
i1=1:ncash;
[i1n,i2n]=ndgrid(i1,i2);
result1 = zeros(6,5,ncash*nh);
result2 = zeros(3,5,ncash*nh);


for t = tn-1:-1:1
% 循环的目的：假设是屌丝，没付过one-time cost
% 分别计算【当期pay掉one-time cost】和【依然不pay one-time cost】的后果
% 进行比较后选择最优化方案
    if t >= tr-tb
        income = ret_fac;
        gyp = 1;
    else
        income = 1;
        f_y1   = exp(incaa+incb1*(stept*(t+tb+1))+incb2*(stept*(t+tb+1))^2+incb3*(stept*(t+tb+1))^3);
        f_y1_2 = exp(incaa+incb1*(stept*(t+tb+1)+1)+incb2*(stept*(t+tb+1)+1)^2+incb3*(stept*(t+tb+1)+1)^3);
        f_y2   = exp(incaa+incb1*(stept*(t+tb))+incb2*(stept*(t+tb))^2+incb3*(stept*(t+tb))^3);
        f_y2_2 = exp(incaa+incb1*(stept*(t+tb)+1)+incb2*(stept*(t+tb)+1)^2+incb3*(stept*(t+tb)+1)^3);
        gyp  = exp((f_y1+f_y1_2)/(f_y2+f_y2_2)-1.0);
    end

    otcost = otcost*gyp;
    ppcost = ppcost*gyp;
    minhouse2 = minhouse2*gyp;

% 假设当期要付ot cost
V_next = V(:,:,t+1);
[XOut, YOut, ZOut] = prepareSurfaceData(ghouse, gcash, V_next);
[model]=fit([XOut, YOut],ZOut,'cubicinterp');
%param_cell = {t rho delta psi_1 psi_2 theta gyp V_next adjcost ppt ppcost otcost income nn survprob gret_sh r gcash ghouse};% 一些需要传入目标函数的参数
param_cell =  {t rho delta psi_1 psi_2 theta gyp model adjcost ppt ppcost otcost income nn survprob gret_sh r};% 一些需要传入目标函数的参数

parfor i=1:ncash*nh
        % 1 处置房产 + 参与股票
        b1 = ghouse(i2n(i))*(1-adjcost-ppt)+gcash(i1n(i))-otcost-ppcost;
        choice1=[nan,nan,nan];
        choice2=[nan,nan,nan];
        value1 = 0.00;
        value2 = 0.00;
        try
        if b1>=0.25+minhouse2
            [choice1,value1] = fmincon( @(x)my_auxV_cal(x,param_cell,gcash(i1n(i)) ,ghouse(i2n(i)))  ,  ...
                [0.25 ,0.2 ,minhouse2]  ,  ...
                [1,0,1;-1,0,0;0,0,-1;0,1,0;0,-1,0],[b1;-0.25;-minhouse2;1;-minalpha]  ,  ...
                [],[],[],[],[],options);
        end
        if b1>=0.25
            [choice2,value2] = fmincon( @(x)my_auxV_cal(x,param_cell,gcash(i1n(i)) ,ghouse(i2n(i)))   ,  ...
                [0.25 ,0.2 ,0.0]  ,  ...
                [1,0,1;-1,0,0;0,1,0;0,-1,0],[b1;-0.25;1;-minalpha]  ,  ...
                [0,0,1],0.00,[],[],[],options); %房产取值为0
        end
        end

        % 2 处置房产 + 不参与股票
        b2 = ghouse(i2n(i))*(1-adjcost-ppt)+gcash(i1n(i));
        choice3=[nan,nan,nan];
        choice4=[nan,nan,nan];
        value3 = 0.00;
        value4 = 0.00;
        try
        if b2>=0.25+minhouse2
            [choice3,value3] = fmincon( @(x)my_auxV_cal(x,param_cell,gcash(i1n(i)),ghouse(i2n(i)))   ,  ...
                [0.25 ,0.0 ,minhouse2]  ,  ...
                [1,0,1;-1,0,0;0,0,-1],[b2;-0.25;-minhouse2]  ,  ...
                [0,1,0],0.00,[],[],[],options);
        end
        if b2>=0.25
            [choice4,value4] = fmincon( @(x)my_auxV_cal(x,param_cell,gcash(i1n(i)),ghouse(i2n(i)))  ,  ...
                [0.25 ,0.0 ,0.0]  ,  ...
                [1,0,1;-1,0,0],[b2;-0.25]  ,  ...
                [0,1,0;0,0,1],[0.00;0.00],[],[],[],options); %房产取值为0 股票取值为0
        end
        end

        % 3 不处置房产 + 参与股票
        b3 = ghouse(i2n(i))*(-ppt)+gcash(i1n(i))-otcost-ppcost;
        choice5=[nan,nan,nan];
        value5 = 0.00;
        try
        if b3>=0.25
            [choice5,value5] = fmincon( @(x)my_auxV_cal(x,param_cell,gcash(i1n(i)),ghouse(i2n(i)))   ,  ...
                [0.25 ,0.2 ,ghouse(i2n(i))]  ,  ...
                [1,0,0;-1,0,0;0,1,0;0,-1,0],[b3;-0.25;1;-minalpha]  ,  ...
                [0,0,1],ghouse(i2n(i)),[],[],[],options);
        end
        end

        % 4 不处置房产 + 不参与股票
        b4 = ghouse(i2n(i))*(-ppt)+gcash(i1n(i));
        choice6=[nan,nan,nan];
        value6 = 0.00;
        try
        if b4>=0.25
            [choice6,value6] = fmincon( @(x)my_auxV_cal(x,param_cell,gcash(i1n(i)),ghouse(i2n(i)))   ,  ...
                [0.25 ,0.0 ,ghouse(i2n(i))]  ,  ...
                [1,0,0;-1,0,0],[b4;-0.25]  ,  ...
                [0,0,1;0,1,0],[ghouse(i2n(i));0.00],[],[],[],options);
        end
        end

        result1(:,:,i) = [value1,choice1,b1;value2,choice2,b1;value3,choice3,b2;value4,choice4,b2;value5,choice5,b3+ghouse(i2n(i));value6,choice6,b4+ghouse(i2n(i))];
end



% 假设当期不付ot cost
V_next = V1(:,:,t+1);
[XOut, YOut, ZOut] = prepareSurfaceData(ghouse, gcash, V_next);
[model]=fit([XOut, YOut],ZOut,'cubicinterp');
%param_cell = {t rho delta psi_1 psi_2 theta gyp model adjcost ppt ppcost otcost income nn survprob gret_sh r };
%param_cell = {t rho delta psi_1 psi_2 theta gyp V_next adjcost ppt ppcost otcost income nn survprob gret_sh r gcash ghouse};% 一些需要传入目标函数的参数
param_cell =  {t rho delta psi_1 psi_2 theta gyp model adjcost ppt ppcost otcost income nn survprob gret_sh r};% 一些需要传入目标函数的参数

parfor i=1:ncash*nh
        % 2 处置房产 + 不参与股票
        b2 = ghouse(i2n(i))*(1-adjcost-ppt)+gcash(i1n(i));
        choice3=[nan,nan,nan];
        choice4=[nan,nan,nan];
        value3 = 0.00;
        value4 = 0.00;
        try
        if b2>=0.25+minhouse2
            [choice3,value3] = fmincon( @(x)my_auxV_cal(x,param_cell,gcash(i1n(i)),ghouse(i2n(i)))   ,  ...
                [0.25 ,0.0 ,minhouse2]  ,  ...
                [1,0,1;-1,0,0;0,0,-1],[b2;-0.25;-minhouse2]  ,  ...
                [0,1,0],0.00,[],[],[],options);
        end
        if b2>=0.25
            [choice4,value4] = fmincon( @(x)my_auxV_cal(x,param_cell,gcash(i1n(i)),ghouse(i2n(i)))  ,  ...
                [0.25 ,0.0 ,0.0]  ,  ...
                [1,0,1;-1,0,0],[b2;-0.25]  ,  ...
                [0,1,0;0,0,1],[0.00;0.00],[],[],[],options); %房产取值为0 股票取值为0
        end
        end

        % 4 不处置房产 + 不参与股票
        b4 = ghouse(i2n(i))*(-ppt)+gcash(i1n(i));
        choice6=[nan,nan,nan];
        value6 = 0.00;
        try
        if b4>=0.25
            [choice6,value6] = fmincon( @(x)my_auxV_cal(x,param_cell,gcash(i1n(i)),ghouse(i2n(i)))   ,  ...
                [0.25,0.0,ghouse(i2n(i))]  ,  ...
                [1,0,0;-1,0,0],[b4;-0.25]  ,  ...
                [0,0,1;0,1,0],[ghouse(i2n(i));0.00],[],[],[],options);
        end
        end
        result2(:,:,i) = [value3,choice3,b2;value4,choice4,b2;value6,choice6,b4+ghouse(i2n(i))];
end

result = [result1;result2];
for i=1:ncash*nh
index = find(result(:,1,i)==min(result(:,1,i)));
C1(i1n(i),i2n(i),t) = result(index(1),2,i);
A1(i1n(i),i2n(i),t) = result(index(1),3,i);
H1(i1n(i),i2n(i),t) = result(index(1),4,i);
V1(i1n(i),i2n(i),t) =-result(index(1),1,i);
W1(i1n(i),i2n(i),t) = result(index(1),5,i); % W is cash on hand
end
H1(H1<1e-3)=0;


end




end