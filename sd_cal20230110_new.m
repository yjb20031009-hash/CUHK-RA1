%% DID-full 第一种方法
did = [0.39094  0.5168  0.30887 0.56021 0.81398 0.34485 0.43339 0.37555];

theta   = reshape(did,1,8).*[10000,50000,10,0.29,0.4,0.2,0.2,0.2]+[0,0,2,0.7,0.3,0.0,0.0,0.0];
nmoments = 34;
load Sample_did.mat
l = length(mySample);

dtv = [100,2000,0.005,0.005,0.002,0.0001,0.0001,0.001];
ntheta = length(dtv);
g1 = zeros(nmoments,ntheta);
g2 = zeros(nmoments,ntheta);
g0 = zeros(nmoments,1);
G  = zeros(nmoments,ntheta);

[ y , g0(:,1) , xdid ] = my_estimation_prepostdid1(theta(1:8));

for i = 1:ntheta
    %for i = [2,7]
    dt = dtv(1,i);
    theta1  = theta;
    theta1(1,i)  = theta1(1,i) + dt;

    theta2  = theta;
    theta2(1,i)  = theta2(1,i) - dt;

    [ ~ , g1(:,i)] = my_estimation_prepostdid1(theta1(1:8));
    [ ~ , g2(:,i)] = my_estimation_prepostdid1(theta2(1:8));

    G(:,i) = (g1(:,i) - g2(:,i))./(2*dt);

end


save did1.mat l G W g0 g1 g2 theta xdid y

clear
load did1
cov = 1/(l) * (1 + 1/9) * pinv(G'*W*G) ;
se_prepostdid1 = sqrt(diag(cov))';
overid_prepostdid1 = l/(1+1/9)*g0(:,1)'*W*g0(:,1)
theta
se_prepostdid1



%% DID-full reduced
did = [0.23657      0.74076 0.24765      0.83453      0.55563];

theta   = reshape(did,1,5).*[10000,50000,10,0.29,0.4]+[0,0,2,0.7,0.3];
nmoments = 34;
load Sample_did.mat
l = length(mySample);

dtv = [10,2500,0.1,0.01,0.01];
ntheta = length(dtv);
g1 = zeros(nmoments,ntheta);
g2 = zeros(nmoments,ntheta);
g0 = zeros(nmoments,1);
G  = zeros(nmoments,ntheta);

[ y , g0(:,1) , xdid ] = my_estimation_prepostdid1(theta(1:5));

for i = 1:ntheta
    dt = dtv(1,i);
    theta1  = theta;
    theta1(1,i)  = theta1(1,i) + dt;

    theta2  = theta;
    theta2(1,i)  = theta2(1,i) - dt;

    [ ~ , g1(:,i)] = my_estimation_prepostdid1(theta1(1:5));
    [ ~ , g2(:,i)] = my_estimation_prepostdid1(theta2(1:5));

    G(:,i) = (g1(:,i) - g2(:,i))./(2*dt);

end

save did2.mat l G W g0 g1 g2 theta y xdid

clear
load did2
cov = 1/(l) * (1 + 1/9) * pinv(G'*W*G) ;
se_prepostdid1 = sqrt(diag(cov))';
overid_prepostdid1 = l*9/(1+9)*g0(:,1)'*W*g0(:,1)
theta
se_prepostdid1



%% DID-high 第一种方法
didhigh = [0.10646      0.24896      0.12587      0.75735      0.61842      0.65657      0.43493      0.39775];
theta   = reshape(didhigh,1,8).*[10000,50000,10,0.29,0.4,0.2,0.2,0.2]+[0,0,2,0.7,0.3,0.0,0.0,0.0];
nmoments = 34;
load Sample_did_high.mat
l = length(mySample);

%dtv = [30,1000,0.05,0.01,0.01,0.001,0.001];
dtv = [100,500,0.005,0.005,0.002,0.0002,0.0002,0.0002];
% dtv = [80,500,0.005,0.005,0.002,0.0006,0.0006,0.0006];

ntheta = length(dtv);
g1 = zeros(nmoments,ntheta);
g2 = zeros(nmoments,ntheta);
g0 = zeros(nmoments,1);
G  = zeros(nmoments,ntheta);

[ ydidhigh , g0(:,1) , xdidhigh ] = my_estimation_prepostdid1_high(theta(1:8));

for i = 1:ntheta
    %for i = [1,2]
    dt = dtv(1,i);
    theta1  = theta;
    theta1(1,i)  = theta1(1,i) + dt;

    theta2  = theta;
    theta2(1,i)  = theta2(1,i) - dt;

    [ ~ , g1(:,i)] = my_estimation_prepostdid1_high(theta1(1:8));
    [ ~ , g2(:,i)] = my_estimation_prepostdid1_high(theta2(1:8));

    G(:,i) = (g1(:,i) - g2(:,i))./(2*dt);

end


save didhigh1.mat l G W g0 g1 g2 theta xdidhigh

clear
load didhigh1
cov = 1/(l) * (1 + 1/9) * pinv(G'*W*G) ;
se_prepostdid1high = sqrt(diag(cov))';
overid_prepostdid1high = l*9/(1+9)*g0(:,1)'*W*g0(:,1)
theta
se_prepostdid1high



%% DID-high reduced
didhigh = [0.25315      0.215708      0.30468         0.71      0.72634];

theta   = reshape(didhigh,1,5).*[10000,50000,10,0.29,0.4]+[0,0,2,0.7,0.3];
nmoments = 34;
load Sample_did_high.mat
l = length(mySample);

dtv = [50,100,0.01,0.01,0.01];
ntheta = length(dtv);
g1 = zeros(nmoments,ntheta);
g2 = zeros(nmoments,ntheta);
g0 = zeros(nmoments,1);
G  = zeros(nmoments,ntheta);

[ ydidhigh , g0(:,1) , xdidhigh ] = my_estimation_prepostdid1_high(theta(1:5));

for i = 1:ntheta
% for i=[1,3]
    dt = dtv(1,i);
    theta1  = theta;
    theta1(1,i)  = theta1(1,i) + dt;

    theta2  = theta;
    theta2(1,i)  = theta2(1,i) - dt;

    [ ~ , g1(:,i)] = my_estimation_prepostdid1_high(theta1(1:5));
    [ ~ , g2(:,i)] = my_estimation_prepostdid1_high(theta2(1:5));

    G(:,i) = (g1(:,i) - g2(:,i))./(2*dt);

end


save didhigh2.mat l G W g0 g1 g2 theta

clear
load didhigh2
cov = 1/(l) * (1 + 1/9) * pinv(G'*W*G) ;
se_prepostdid1high = sqrt(diag(cov))';
overid_prepostdid1high = l*9/(1+2)*g0(:,1)'*W*g0(:,1)
theta
se_prepostdid1high





%% DID-low 第一种方法
didlow = [0.36106      0.70974      0.36162      0.80249      0.66785      0.38534      0.38451       0.34775];

theta   = reshape(didlow,1,8).*[10000,50000,10,0.29,0.4,0.2,0.2,0.2]+[0,0,2,0.7,0.3,0.0,0.0,0.0];
nmoments = 34;
load Sample_did_low.mat
l = length(mySample);

dtv = [100,2000,0.01,0.01,0.001,0.002,0.005,0.002];

ntheta = length(dtv);
g1 = zeros(nmoments,ntheta);
g2 = zeros(nmoments,ntheta);
g0 = zeros(nmoments,1);
G  = zeros(nmoments,ntheta);

[ ydidlow , g0(:,1) , xdidlow ] = my_estimation_prepostdid1_low(theta(1:8));

for i = 1:ntheta

    dt = dtv(1,i);
    theta1  = theta;
    theta1(1,i)  = theta1(1,i) + dt;

    theta2  = theta;
    theta2(1,i)  = theta2(1,i) - dt;

    [ ~ , g1(:,i)] = my_estimation_prepostdid1_low(theta1(1:8));
    [ ~ , g2(:,i)] = my_estimation_prepostdid1_low(theta2(1:8));

    G(:,i) = (g1(:,i) - g2(:,i))./(2*dt);

end


save didlow1.mat l G W g0 g1 g2 theta xdidlow

clear
load didlow1
cov = 1/(l) * (1 + 1/9) * pinv(G'*W*G) ;
se_prepostdid1low = sqrt(diag(cov))';
overid_prepostdid1low = l/(1+1/9)*g0(:,1)'*W*g0(:,1)
theta
se_prepostdid1low


%% DID-low reduced
didlow = [0.26122      0.88806      0.25246      0.89596      0.35241];
theta   = reshape(didlow,1,5).*[10000,50000,10,0.29,0.4]+[0,0,2,0.7,0.3];
nmoments = 34;
load Sample_did_low.mat
l = length(mySample);

dtv = [500,2000,0.01,0.01,0.01];
ntheta = length(dtv);
g1 = zeros(nmoments,ntheta);
g2 = zeros(nmoments,ntheta);
g0 = zeros(nmoments,1);
G  = zeros(nmoments,ntheta);

[ ydidlow , g0(:,1) , xdidlow ] = my_estimation_prepostdid1_low(theta(1:5));

%for i = 1:ntheta
for i=[1,2]    
    dt = dtv(1,i);
    theta1  = theta;
    theta1(1,i)  = theta1(1,i) + dt;

    theta2  = theta;
    theta2(1,i)  = theta2(1,i) - dt;

    [ ~ , g1(:,i)] = my_estimation_prepostdid1_low(theta1(1:5));
    [ ~ , g2(:,i)] = my_estimation_prepostdid1_low(theta2(1:5));

    G(:,i) = (g1(:,i) - g2(:,i))./(2*dt);

end


save didlow2.mat l G W g0 g1 g2 theta

clear
load didlow2
cov = 1/(l) * (1 + 1/9) * pinv(G'*W*G) ;
se_prepostdid1low = sqrt(diag(cov))';
overid_prepostdid1low = l*9/(1+9)*g0(:,1)'*W*g0(:,1)
theta
se_prepostdid1low

