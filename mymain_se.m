function [ C, A, H, C1, A1, H1]=mymain_se(ppt,ppcost,otcost,rho,delta,psi,varargin)
%% 【关键内层函数】求解policy function
% 输入一组可变的参数，通过grid search和fmincon结合的方法，求解policy function

%% 【固定参数的取值基本都在这个函数里】
% 内置于外层函数my_estimation_prepostdid1（等）中
% 输入 一组参数，包括家户偏好、参与成本及房产税等
% 输出 各个Choice（消费、股票和房产投资）的policy function。结果是一组二维矩阵，两个维度分别为期初的cash和housing，即两个状态变量
% 其中C  A  H 是【人上人】（已经pay过one-time cost的人）的policy funtion
%     C1 A1 H1是【屌丝】（没参与过股市）的policy funtion

%%
%{
clear
ppt =   0;
ppcost =   0.024689;
otcost = 0.097854;
rho =  9.759;
delta = 0.9871;
psi = 0.67324;
mu          = 0.08 ; %股票 超额回报 取8%
muh         = 0.16 ; %房产 超额回报 取8%

[ C, A, H, C1, A1, H1]=mymain_se(ppt,ppcost,otcost,rho,delta,psi,mu,muh);
[ ggvalue , gvalue , betamat ] =my_estimation_prepostdid1([ 0.1990    0.0354    0.6103    0.9940    0.9885],0.3096 , 0.7269);

%}

%{

xx= [0.10824  0.16913  0.087013   0.87906   0.7491 0.84164  0.57357];
xx= [0.12824  0.06913  0.107013   0.87906   0.7491 0.54164  0.41357];
xx= [0.10824  0.12913  0.097013   0.87906   0.6491 0.69164  0.51357];
xx= [0.10824  0.16913  0.087013   0.87906   0.6491 0.72164  0.51357];
xx= [0.07824  0.11913  0.387013   0.92906   0.5791 0.68164  0.51357];
xx= [0.08824  0.06913  0.207013   0.89906   0.6791 0.52164  0.45357];
xx= [0.30098  0.13176  0.15429    0.98253   0.5977 0.57863  0.57315];***
xx= [ 0.36121   0.163049    0.21044    0.84114  0.62986 0.51233 0.63338];%****
xx= [ 0.36121   0.163049    0.21044    0.84114  0.62986 0.41233 0.43338];
xx=[ 0.3047    0.2123    0.3837    0.9900    0.5744    0.5932    0.7176]; %***
xx=[0.3300    0.2042    0.3719    0.9631    0.6499    0.4502    0.3462];%**
xx=[0.2300    0.2042    0.3719    0.9631    0.6499    0.4502    0.4462];%**
xx=[ 0.2386    0.4552    0.5730    0.9900    0.7300    0.4369    0.4483];
xx=[ 0.0998    0.2452    0.3876    0.8233    0.4241 0.4 0.4];
xx=[0.2170    0.3883    0.6628    0.9900    0.8447    0.8923 0.4766];%****
xx=[0.2170    0.3883    0.6628    0.9900    0.8447    0.6923    0.6766];%*可以
xx=[0.2170    0.2883    0.6628    0.9900    0.8447    0.6923 0.6766];%调低fixed cost
xx=[0.1970    0.1183    0.2628    0.9900    0.8947    0.3223 0.3066];%调低fixed cost

tic
[~,~,bt]=my_estimation_prepostdid1(xx)
toc




xx= [ 0.166121   0.0483049    0.21044    0.94114  0.62986 0.61233 0.60338]; 
xx = [0.186121   0.0083049    0.21044    0.94114  0.62986 0.61233 0.63338]; %**
xx = [0.13549     0.072368       0.2241      0.91098      0.64206      0.56953      0.61164];
xx = [ 0.16813     0.070541      0.40842      0.80425      0.50617      0.72596      0.47883];
xx = [0.12258 0.14673 0.50665  0.78016  0.55273  0.88093  0.47246];
xx = [0.081906 0.079669 0.47396   0.74455   0.58186 0.95962  0.5442];%***
xx = [0.081906  0.079669  0.47396 0.74455  0.58186    0.75962 0.5442];%**
xx=[ 0.3953    0.1589    0.6188    0.6260    0.3472    0.8430    0.7164];
xx=[ 0.2177    0.1725    0.5753    0.6540    0.2137    0.6608    0.7022];
xx=[0.4761    0.1622    0.5324    0.4663    0.2406    0.7438 0.7507];%******
xx=[0.2761    0.1922    0.5324    0.4663    0.2406    0.6438    0.6507];
xx=[0.2070    0.0922        0.0628    0.8800    0.4947    0.5438 0.5507];%来自全样本
xx=[0.2070    0.0422        0.0628    0.8800    0.4947    0.5438 0.5507];%来自全样本

tic
[~,~,bt]=my_estimation_prepostdid1_high(xx)
toc



xx= [0.30098  0.13176  0.15429    0.98253   0.5977 0.57863  0.57315];
xx=[0.33516  0.17103  0.10314  0.85812 0.37009 0.66279  0.63997];
xx=[ 0.3796    0.1769    0.0158    0.9831    0.5104    0.5902    0.4655];
xx=[0.3574    0.2495    0.0214    0.9900    0.6677    0.5687    0.3388];
xx=[0.3574    0.2495    0.0214    0.9900    0.6677    0.4687    0.3388];
xx=[0.3506    0.2293    0.0453    0.9604    0.6533    0.4684    0.3280];
xx=[ 0.3534    0.2064    0.0238    0.9708    0.6761    0.5442    0.1111];
xx=[ 0.3754    0.3624    0.0727    0.9284    0.7410    0.6252    0.2848];
xx=[0.3040    0.3052    0.0302    0.9557    0.8917    0.4926    0.4262];

xx=[0.30040    0.3052    0.0062    0.9987    0.6917    0.3326    0.3462];
xx=[0.30040    0.2552    0.0062    0.9987    0.6917    0.2026    0.4462];
xx=[ 0.30040    0.3052    0.062    0.9987    0.6917 0.2826 0.2862];
tic
[~,~,bt]=my_estimation_prepostdid1_low(xx)
toc







x = [ 0.36121   0.053049    0.21044    0.84114  0.62986]; **
x = [ 0.20121   0.103049    0.51044    0.94114  0.62986]
x=[ 0.467         0.5    0.5481    0.9900    0.6540];
x=[0.00    0.2376    0.4985    0.8740    0.0481];
x=[ 0.0998    0.2452    0.3876    0.8233    0.4241];
x=[ 0.1328    0.1932    0.3175    0.9594    0.1506];
x=[0.1328    0.4932    0.6175    0.9594    0.1506];

x=[0.1328    0.3932    0.4175    0.9594    0.1506];%**
x=[0.1328    0.3932    0.3975    0.9594    0.1506];
x=[0.1328    0.2232    0.1775    0.9994    0.8806];

tic
[~,~,bt]=my_estimation_prepostdid1(x)
toc


x = [0.33516  0.17103  0.10314  0.85812 0.37009];
x=[ 0.2523    0.3904    0.0111    0.9684    0.9820];
tic
[~,~,bt]=my_estimation_prepostdid1_low(x)
toc


x = [0.081906  0.079669   0.47396  0.74455  0.58186 ];
x=[0.2126    0.03572    0.15104    0.9802    0.1641];
x=[ 0.08026    0.0272    0.5104    0.8702    0.7841];

x = [0.081906  0.079669   0.47396  0.74455  0.58186 ];
x = [0.081906  0.0079669   0.57396  0.74455  0.58186 ];
x = [0.081906  0.079669   0.57396  0.74455  0.58186 ];

tic
[~,~,bt]=my_estimation_prepostdid1_high(x)
toc

%}


%% 参数设定
% A 生命周期
stept = 2; %每一期包括2年
tb     = 20/stept; %家户开始的年龄
tr     = 62/stept; %家户退休的年龄
td     = 100/stept; %家户最大的年龄（最后一期）
tn     = td-tb+1; %家户最长的存活时间
%t_jump = -4;
%tn_jump = (td-tb)/(-t_jump)+1;
% B 格点

%ncash  = 11;
%nh     = 6; % housing购买量 网格分组

na     = 5; % 股票比例格点
nc     = 3; % 消费格点
nh2    = 5; % 房产投资选择的格点

n      = 3; %将股票收益率的连续分布 离散成3种情况
nn     = n*n; %允许房产收益率也离散为3种可能性，故股票-房产收益率组合共有9种



% C 收入 （使用60岁以下家庭的收入对年龄的三阶多项式 回归 得到）
global ncash nh incaa incb1 incb2 incb3
smay        = nan; %收入风险。本模型不需要
smav        = nan; %收入风险。本模型不需要
corr_v      = nan; %收入(暂时性)风险与股票风险的关系。本模型不需要
corr_y      = nan; %收入(永久性)风险与股票风险的关系。本模型不需要

ret_fac     = 0.6; %退休后的收入因子




% D 资产和现金
adjcost           = 0.07; %住房的调整成本 7%
maxhouse          = 40.0; % 房产的上界，收入的200倍。技术设定。
minhouse          = 0.00;  %不需要的设定，取0即可。如果不取0，当房产价值低于此值时，房产必须被清偿。
minhouse2_value   = 250000*0; %房产最小可购买金额，25万人民币。
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
options = optimoptions('fmincon','Display','off','ConstraintTolerance',1e-2,'OptimalityTolerance',1e-2,'Algorithm', 'interior-point'); % 防止fmincon的输出刷屏


%% 循环1 -- 【人上人】
% one-time cost(otcost) =0的情况下，计算policy function

ppcost = Param1(11);
minhouse2 = Param2(2);
otcost = 0;
i2=1:nh;
i1=1:ncash;
[i1n,i2n]=ndgrid(i1,i2);
result = zeros(6,4,ncash*nh);

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
    %[XOut, YOut, ZOut] = prepareSurfaceData(ghouse, gcash, V_next);
    %[model]=fit([XOut, YOut],ZOut,'cubicinterp'); 
    [model]=griddedInterpolant({gcash,ghouse},V_next,'spline'); 

    
    param_cell = {t rho delta psi_1 psi_2 theta gyp model adjcost ppt ppcost otcost income nn survprob gret_sh r};% 一些需要传入目标函数的参数

%parfor i=1:ncash*nh
parfor i=1:ncash*nh

% 对于每个状态变量的grid，都进行fmincon的搜索来找到最合适的choice
% 由于不同的choice对应不同的约束条件，所以需要分4种情况讨论

        % 1 处置房产 + 参与股票 （意味着要从cash里付adjustment cost和股票的cost）
        b1 = ghouse(i2n(i))*(1-adjcost-ppt)+gcash(i1n(i))-otcost-ppcost;
        choice1=[nan,nan,nan];
        choice2=[nan,nan,nan];
        value1 = 0.00;
        value2 = 0.00;
        % 下面的my_auxV_cal是目标函数，输入一组choice (consumption,alpha,housing)，输出-V
        try
        if b1>=0.25+minhouse2
            [choice1,value1] = fmincon( @(x)my_auxV_cal(x,param_cell,gcash(i1n(i)), ghouse(i2n(i)))  ,  ...  
                [0.25 ,0.2 , minhouse2]  ,  ...
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
        b1 = ghouse(i2n(i))*(-ppt)+gcash(i1n(i))-otcost-ppcost;
                    choice5=[nan,nan,nan];
            value5 = 0.00;
            try
        if b1>=0.25
            [choice5,value5] = fmincon( @(x)my_auxV_cal(x,param_cell,gcash(i1n(i)) ,ghouse(i2n(i)))   ,  ...
                [0.25 ,0.2 ,ghouse(i2n(i))]  ,  ...
                [1,0,0;-1,0,0;0,1,0;0,-1,0],[b1;-0.25;1;-minalpha]  ,  ...
                [0,0,1],ghouse(i2n(i)),[],[],[],options);
        end
        end

        % 4 不处置房产 + 不参与股票
        b1 = ghouse(i2n(i))*(-ppt)+gcash(i1n(i));
            choice6=[nan,nan,nan];
            value6 = 0.00;
            try
        if b1>=0.25
            [choice6,value6] = fmincon( @(x)my_auxV_cal(x,param_cell,gcash(i1n(i)) ,ghouse(i2n(i)))   ,  ...
                [0.25 ,0.0 ,ghouse(i2n(i))]  ,  ...
                [1,0,0;-1,0,0],[b1;-0.25]  ,  ...
                [0,0,1;0,1,0],[ghouse(i2n(i));0.00],[],[],[],options);
        end
        end

        result(:,:,i) = [value1,choice1;value2,choice2;value3,choice3;value4,choice4;value5,choice5;value6,choice6];
    
end

for i=1:ncash*nh
index = find(result(:,1,i)==min(result(:,1,i))); % 由于函数输出的是-V，这里需要取最小值
C(i1n(i),i2n(i),t) = result(index(1),2,i);
A(i1n(i),i2n(i),t) = result(index(1),3,i);
H(i1n(i),i2n(i),t) = result(index(1),4,i);
V(i1n(i),i2n(i),t) =-result(index(1),1,i); % 由于函数输出的是-V，这里需要取负值
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
result1 = zeros(6,4,ncash*nh);
result2 = zeros(3,4,ncash*nh);


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
    %[XOut, YOut, ZOut] = prepareSurfaceData(ghouse, gcash, V_next);
    %[model]=fit([XOut, YOut],ZOut,'cubicinterp'); 
    [model]=griddedInterpolant({gcash,ghouse},V_next,'spline'); 
%param_cell = {t rho delta psi_1 psi_2 theta gyp V_next adjcost ppt ppcost otcost income nn survprob gret_sh r gcash ghouse};% 一些需要传入目标函数的参数
param_cell =  {t rho delta psi_1 psi_2 theta gyp model adjcost ppt ppcost otcost income nn survprob gret_sh r};% 一些需要传入目标函数的参数

%parfor i=1:ncash*nh
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
        b1 = ghouse(i2n(i))*(-ppt)+gcash(i1n(i))-otcost-ppcost;
        choice5=[nan,nan,nan];
        value5 = 0.00;
        try
        if b1>=0.25
            [choice5,value5] = fmincon( @(x)my_auxV_cal(x,param_cell,gcash(i1n(i)),ghouse(i2n(i)))   ,  ...
                [0.25 ,0.2 ,ghouse(i2n(i))]  ,  ...
                [1,0,0;-1,0,0;0,1,0;0,-1,0],[b1;-0.25;1;-minalpha]  ,  ...
                [0,0,1],ghouse(i2n(i)),[],[],[],options);
        end
        end

        % 4 不处置房产 + 不参与股票
        b1 = ghouse(i2n(i))*(-ppt)+gcash(i1n(i));
        choice6=[nan,nan,nan];
        value6 = 0.00;
        try
        if b1>=0.25
            [choice6,value6] = fmincon( @(x)my_auxV_cal(x,param_cell,gcash(i1n(i)),ghouse(i2n(i)))   ,  ...
                [0.25 ,0.0 ,ghouse(i2n(i))]  ,  ...
                [1,0,0;-1,0,0],[b1;-0.25]  ,  ...
                [0,0,1;0,1,0],[ghouse(i2n(i));0.00],[],[],[],options);
        end
        end

        result1(:,:,i) = [value1,choice1;value2,choice2;value3,choice3;value4,choice4;value5,choice5;value6,choice6];
end



% 假设当期不付ot cost
V_next = V1(:,:,t+1);
    %[XOut, YOut, ZOut] = prepareSurfaceData(ghouse, gcash, V_next);
    %[model]=fit([XOut, YOut],ZOut,'cubicinterp'); 
    [model]=griddedInterpolant({gcash,ghouse},V_next,'spline'); 
%param_cell = {t rho delta psi_1 psi_2 theta gyp model adjcost ppt ppcost otcost income nn survprob gret_sh r };
%param_cell = {t rho delta psi_1 psi_2 theta gyp V_next adjcost ppt ppcost otcost income nn survprob gret_sh r gcash ghouse};% 一些需要传入目标函数的参数
param_cell =  {t rho delta psi_1 psi_2 theta gyp model adjcost ppt ppcost otcost income nn survprob gret_sh r};% 一些需要传入目标函数的参数

%parfor i=1:ncash*nh
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
        b1 = ghouse(i2n(i))*(-ppt)+gcash(i1n(i));
        choice6=[nan,nan,nan];
        value6 = 0.00;
        try
        if b1>=0.25
            [choice6,value6] = fmincon( @(x)my_auxV_cal(x,param_cell,gcash(i1n(i)),ghouse(i2n(i)))   ,  ...
                [0.25,0.0,ghouse(i2n(i))]  ,  ...
                [1,0,0;-1,0,0],[b1;-0.25]  ,  ...
                [0,0,1;0,1,0],[ghouse(i2n(i));0.00],[],[],[],options);
        end
        end
        result2(:,:,i) = [value3,choice3;value4,choice4;value6,choice6];
end

result = [result1;result2];
for i=1:ncash*nh
index = find(result(:,1,i)==min(result(:,1,i)));
C1(i1n(i),i2n(i),t) = result(index(1),2,i);
A1(i1n(i),i2n(i),t) = result(index(1),3,i);
H1(i1n(i),i2n(i),t) = result(index(1),4,i);
V1(i1n(i),i2n(i),t) =-result(index(1),1,i);
end
H1(H1<1e-3)=0;


end




end