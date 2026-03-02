
%% 本文档用于迭代求解模型参数
% 优化方法：CMA-ES
clc
clear
%% 定义优化参数
opts.StopFitness = 1e-2;
opts.TolX = 1e-2;
opts.MaxIter = 8;
%opts.MaxFunEvals = 10;
opts.LBounds = 0.01; 
opts.UBounds = 0.99; 

% 不同参数的取值范围不同。设定如下：
% per-period cost：0元-9900元
% one-time cost：0元-180000元
% rho：2-10
% beta：0.7-0.99
% psi：0.3-0.7
% mu_stock：0%-20%
% mu_house：0%-20%
% ppt: 0%-9.9%

start_time0 = clock
%% 测试
tic
[y1,y2,y3]=my_estimation_prepostdid1_high([0.2090    0.11054    0.6103    0.9940    0.9885  0.3096  0.3269 0.2]);
toc
%[y1,y2,y3]=my_estimation_prepostdid1_high([0.2652    0.01303    0.7262    0.9589    0.8936],0.5124,0.4742);
end_time0 = clock

start_time = clock
%% DID 使用默认的预期回报率
% 需要设定5个参数：[per-period cost, one-time cost, rho, beta, psi]

tic
[x_prepostdid1p ,fval_prepostdid1p] = cmaes2('my_estimation_prepostdid1', [0.39094  0.5168  0.30887 0.56021 0.81398] ,[0.1;0.1;0.1;0.1;0.1],opts);
toc
%  0.01      0.17285      0.24236      0.82137      0.96671

tic
[x_prepostdid1lowp ,fval_prepostdid1lowp] = cmaes2( 'my_estimation_prepostdid1_low' ,[0.36106      0.70974      0.36162      0.80249      0.66785] ,[0.1;0.1;0.1;0.1;0.1],opts);
toc
% 0.14533      0.23565      0.90618      0.98705      0.84383

tic
[x_prepostdid1highp ,fval_prepostdid1highp] = cmaes2('my_estimation_prepostdid1_high',[0.10646      0.24896      0.12587      0.75735      0.61842   ] ,[0.1;0.1;0.1;0.1;0.1],opts);
toc
% 0.071933     0.063913      0.75004      0.95579      0.77177


end_time1 = clock

%% DiD 将预期回报率参数化
% 需要设定7个参数：[[per-period cost, one-time cost, rho, beta, psi] ,mu_stock, mu_house]

tic
[x_prepostdid1 ,fval_prepostdid1] = cmaes2('my_estimation_prepostdid1',[0.184507     0.152546      0.62454      0.96237         0.80      0.24914      0.53005] ,[0.1;0.1;0.1;0.1;0.1;0.1;0.1],opts);
toc
% 0.054507     0.032546      0.62454      0.96237         0.99      0.44914      0.21005

tic
[x_prepostdid1low ,fval_prepostdid1low] = cmaes2( 'my_estimation_prepostdid1_low' ,[0.25216      0.18172      0.73077      0.95984      0.84534      0.16423      0.52141]  ,[0.1;0.1;0.1;0.1;0.1;0.1;0.1],opts);
toc
% 0.15216      0.18172      0.73077      0.95984      0.94534      0.46423      0.22141

tic
[x_prepostdid1high ,fval_prepostdid1high] = cmaes2('my_estimation_prepostdid1_high',[0.156438     0.112763      0.71454      0.98519      0.69758      0.34492      0.55081],[0.1;0.1;0.1;0.1;0.1;0.1;0.1],opts);
toc
% 0.076438     0.062763      0.71454      0.98519      0.69758      0.34492      0.55081

end_time2 = clock


%% DiD 将预期回报率参数化+将税率参数化
% 需要设定7个参数：[[per-period cost, one-time cost, rho, beta, psi] ,mu_stock, mu_house,ppt]

tic
[x_prepostdid1x ,fval_prepostdid1x] = cmaes2('my_estimation_prepostdid1',[x_prepostdid1x] ,[0.08;0.08;0.06;0.06;0.06;0.06;0.06;0.06],opts);
toc
% 0.070731     0.057129      0.86931      0.88607      0.65098      0.33048      0.41979     0.089663
% [0.057711      0.28882      0.41003      0.80579      0.39534      0.42001      0.22075      0.16878]
% [0.057711      0.28882      0.41003      0.80579      0.39534      0.22001      0.22075      0.16878]
% 0.25961  0.28963   0.48659   0.99 0.69384  0.60016 0.60046 0.096075


tic
[x_prepostdid1lowx ,fval_prepostdid1lowx] = cmaes2( 'my_estimation_prepostdid1_low' ,[x_prepostdid1lowx]  ,[0.08;0.08;0.06;0.06;0.06;0.06;0.06;0.06],opts);
toc
% 0.01      0.24726       0.6384       0.8132      0.94995      0.42439      0.28296       0.1616
% 0.10898  0.42073  0.6962 0.96522  0.75846 0.48588  0.50902 0.035878
tic
[x_prepostdid1highx ,fval_prepostdid1highx] = cmaes2('my_estimation_prepostdid1_high', [x_prepostdid1highx] ,[0.08;0.08;0.06;0.06;0.06;0.06;0.06;0.06],opts);
toc

% 0.042176     0.078503      0.86647      0.86119      0.77409      0.35865      0.55484     0.088007
%  0.20828      0.10396         0.99      0.69763      0.83414      0.46305      0.58922     0.043706
% 0.15828 0.15396  0.49  0.89763  0.63414  0.46305  0.58922  0.043706
% 0.15828 0.15396  0.39  0.89763  0.63414  0.46305  0.49922  0.031706


end_time3 = clock

[~,~,y3t]=my_estimation_prepostdid1(      [0.39094  0.5168  0.30887 0.56021 0.81398 0.34485 0.43339 0.37555])
% 0.5860    0.6436    0.2388    0.3400    0.8135    0.3408    0.4500    0.3668
[~,~,y3at]=my_estimation_prepostdid1_low( [0.36106      0.70974      0.36162      0.80249      0.66785      0.38534      0.38451       0.34775])
% 0.4456    0.7131    0.1380    0.7865    0.8123    0.2697    0.4572    0.3667
[~,~,y3bt]=my_estimation_prepostdid1_high([0.10646      0.24896      0.12587      0.75735      0.61842      0.65657      0.43493      0.39775])
% 0.3839    0.5524    0.4457    0.4684    0.9090    0.4705    0.4577    0.3702
%  0.10646      0.24896      0.12587      0.75735      0.61842      0.75657      0.43493      0.42775
% 0.10646      0.24896      0.12587      0.75735      0.61842      0.75657      0.43493      0.39775

x_prepostdid1x'
[y1,y2,y3]=my_estimation_prepostdid1([x_prepostdid1x]);
x_prepostdid1lowx'
[y1,y2,y3a]=my_estimation_prepostdid1_low([x_prepostdid1lowx]);
x_prepostdid1highx'
[y1,y2,y3b]=my_estimation_prepostdid1_high([x_prepostdid1highx]);

x_prepostdid1p'
%  0.23657      0.74076 0.24765      0.83453      0.55563
[y1,y2,y3]=my_estimation_prepostdid1([x_prepostdid1p]);
x_prepostdid1lowp'
% [0.26122      0.88806      0.25246      0.89596      0.35241]
[y1,y2,y3a]=my_estimation_prepostdid1_low([0.26122    0.85806  0.25246      0.89596      0.35241]);
x_prepostdid1highp'
% 0.25315      0.215708      0.30468         0.71      0.72634
[y1,y2,y3b]=my_estimation_prepostdid1_high([x_prepostdid1highp]);

