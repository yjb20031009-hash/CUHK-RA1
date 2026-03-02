%% DID-full 第一种方法
did = [0.2170    0.2883    0.6628    0.9900    0.8447    0.6923 0.6766];

theta   = reshape(did,1,7).*[10000,200000,10,0.29,0.4,0.2,0.2]+[0,0,2,0.7,0.3,0.0,0.0];
nmoments = 27;
load Sample_did.mat
l = length(mySample);

dtv = [30,2500,0.5,0.01,0.01,0.001,0.001];
ntheta = length(dtv);
g1 = zeros(nmoments,ntheta);
g2 = zeros(nmoments,ntheta);
g0 = zeros(nmoments,1);
G  = zeros(nmoments,ntheta);

[ ~ , g0(:,1) , xdid ] = my_estimation_prepostdid1(theta(1:7));

for i = 1:ntheta
    dt = dtv(1,i);
    theta1  = theta;
    theta1(1,i)  = theta1(1,i) + dt;

    theta2  = theta;
    theta2(1,i)  = theta2(1,i) - dt;

    [ ~ , g1(:,i)] = my_estimation_prepostdid1(theta1(1:7));
    [ ~ , g2(:,i)] = my_estimation_prepostdid1(theta2(1:7));

    G(:,i) = (g1(:,i) - g2(:,i))./(2*dt);

end

cov = 1/(l) * (1 + 1/9) * pinv(G'*W*G) ;
se_prepostdid1 = sqrt(diag(cov))';
overid_prepostdid1 = l*9/(1+9)*g0(:,1)'*W*g0(:,1)
theta
se_prepostdid1



%% DID-low 第一种方法
didlow = [  0.0758    0.0567    0.9232    0.9399    0.9813  0.2665   0.3372];

theta   = reshape(didlow,1,7).*[10000,200000,8,0.29,0.4,0.2,0.2]+[0,0,2,0.7,0.3,0.0,0.0];
nmoments = 27;
load Sample_did_low.mat
l = length(mySample);

dtv = [30,25000,0.5,0.01,0.01,0.001,0.001];
ntheta = length(dtv);
g1 = zeros(nmoments,ntheta);
g2 = zeros(nmoments,ntheta);
g0 = zeros(nmoments,1);
G  = zeros(nmoments,ntheta);

[ ydidlow , g0(:,1) , xdidlow ] = my_estimation_prepostdid1_low(theta(1:5),theta(6) , theta(7));

for i = 1:ntheta
    dt = dtv(1,i);
    theta1  = theta;
    theta1(1,i)  = theta1(1,i) + dt;

    theta2  = theta;
    theta2(1,i)  = theta2(1,i) - dt;

    [ ~ , g1(:,i)] = my_estimation_prepostdid1_low(theta1(1:5),theta1(6) , theta1(7));
    [ ~ , g2(:,i)] = my_estimation_prepostdid1_low(theta2(1:5),theta2(6) , theta2(7));

    G(:,i) = (g1(:,i) - g2(:,i))./(2*dt);

end

cov = 1/(l) * (1 + 1/9) * pinv(G'*W*G) ;
se_prepostdid1low = sqrt(diag(cov))';
overid_prepostdid1low = l*9/(1+9)*g0(:,1)'*W*g0(:,1)
theta
se_prepostdid1low




%% DID-high 第一种方法
didhigh = [0.2761    0.1922    0.5324    0.4663    0.2406    0.6438    0.6507];
theta   = reshape(didhigh,1,7).*[10000,200000,10,0.29,0.4,0.2,0.2]+[0,0,2,0.7,0.3,0.0,0.0];

nmoments = 27;
load Sample_did_high.mat
l = length(mySample);

dtv = [10,100,0.002,0.01,0.01,0.0005,0.0005];
ntheta = length(dtv);
g1 = zeros(nmoments,ntheta);
g2 = zeros(nmoments,ntheta);
g0 = zeros(nmoments,1);
G  = zeros(nmoments,ntheta);

[ ydidhigh , g0(:,1) , xdidhigh ] = my_estimation_prepostdid1_high(theta(1:7));

%for i = 1:ntheta
 for i = [6,7]   
    dt = dtv(1,i);
    theta1  = theta;
    theta1(1,i)  = theta1(1,i) + dt;

    theta2  = theta;
    theta2(1,i)  = theta2(1,i) - dt;

    [ ~ , g1(:,i)] = my_estimation_prepostdid1_high(theta1(1:7));
    [ ~ , g2(:,i)] = my_estimation_prepostdid1_high(theta2(1:7));

    G(:,i) = (g1(:,i) - g2(:,i))./(2*dt);

end

cov = 1/(l) * (1 + 1/9) * pinv(G'*W*G) ;
se_prepostdid1high = sqrt(diag(cov))';
overid_prepostdid1high = l*9/(1+9)*g0(:,1)'*W*g0(:,1)
theta
se_prepostdid1high


