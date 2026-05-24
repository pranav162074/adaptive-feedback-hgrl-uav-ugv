function output = solve_problem(x_start, y_start, x_end, y_end, H, a, b, c, d, V, u_max)
    m = size(a,2); % 障碍物数目
    % 优化变量的界限
    l_x = [-2*ones(H-2,1); 0*ones(H-2,1); 0; 0];
    u_x = [12*ones(H-2,1); 12*ones(H-2,1); 20/(H-1); (20/(H-1))^2];

    % 定义目标矩阵（最小化路径长度） %minimize( (H-1)*Δt )
    Q0 = zeros(2*H-1,2*H-1);
    Q0(2*H-1,2*H-3) = (H-1)/2;
    Q0(2*H-3,2*H-1) = (H-1)/2;

    %define equality constraints
    Eqcon(1).Q = zeros(2*H-1,2*H-1); %α^2=1 确保参数α的平方等于1
    Eqcon(1).Q(2*H-1,2*H-1) = 1;
    Eqcon(1).c = -1;
    
    Eqcon(2).Q = zeros(2*H-1,2*H-1); %t'=(Δt)^2 确保时间步长t的平方等于（Δt）的平方，有助于在离散化路径时保持时间的一致性
    Eqcon(2).Q(2*H-3,2*H-3) = -1;
    Eqcon(2).Q(2*H-1,2*H-2) = 0.5;
    Eqcon(2).Q(2*H-2,2*H-1) = 0.5;
    Eqcon(2).c = 0;
    
    Eqcon(3).Q = zeros(2*H-1,2*H-1); %x2^2+y2^2- V^2*(Δt)^2=0 确保路径经过的每个点（除了起点和终点）的速度在给定速度V的条件下达到
   Eqcon(3).Q(1,1) = 1;
   Eqcon(3).Q(H-1,H-1) = 1;
   Eqcon(3).Q(2*H-1,1) = -x_start;
   Eqcon(3).Q(1,2*H-1) = -x_start;
   Eqcon(3).Q(2*H-1,H-1) = -y_start;
   Eqcon(3).Q(H-1,2*H-1) = -y_start;
   Eqcon(3).Q(2*H-3,2*H-3) = -V^2;
   Eqcon(3).c = x_start^2+y_start^2;
    
   % (x_{h+1}-x_h)^2 + (y_{h+1}-y_h)^2 - V^2*(Δt)^2 = 0, h=2,...,H-2 确保路径上每个点之间的速度在给定速度V的条件下达到
   for i = 1:H-3
       Eqcon(3+i).Q = zeros(2*H-1,2*H-1);
       Eqcon(3+i).Q(i,i) = 1;
       Eqcon(3+i).Q(i,i+1) = -1;
       Eqcon(3+i).Q(i+1,i) = -1;
       Eqcon(3+i).Q(i+1,i+1) = 1;
       Eqcon(3+i).Q(H-2+i,H-2+i) = 1;
       Eqcon(3+i).Q(H-2+i,H-2+i+1) = -1;
       Eqcon(3+i).Q(H-2+i+1,H-2+i) = -1;
       Eqcon(3+i).Q(H-2+i+1,H-2+i+1) = 1;
       Eqcon(3+i).Q(2*H-3,2*H-3) = -V^2;
       Eqcon(3+i).c = 0;
   end

  %  x_{H-1}^2 + y_{H-1}^2 - 20*x_{H-1} - 20*y_{H-1} - V^2*(Δt)^2=0，确保无人机在最后一个时间步长 H-1H?1 时，其位置和速度与预定的终点坐标和速度要求相匹配
   Eqcon(H+1).Q = zeros(2*H-1,2*H-1);
   Eqcon(H+1).Q(H-2,H-2) = 1;
   Eqcon(H+1).Q(2*H-4,2*H-4) = 1;
   Eqcon(H+1).Q(2*H-1,H-2) = -x_end;
   Eqcon(H+1).Q(H-2,2*H-1) = -x_end;
   Eqcon(H+1).Q(2*H-1,2*H-4) = -y_end;
   Eqcon(H+1).Q(2*H-4,2*H-1) = -y_end;
   Eqcon(H+1).Q(2*H-3,2*H-3) = -V^2;
   Eqcon(H+1).c = x_end^2+y_end^2;%设置为终点坐标的平方和，这是约束的常数项
    
    
    %define inequality constraints
    %l_x<=x<=u_x
    for i = 1:2*H-2  %x2~x_{H-1}的取值范围
        Incon(i).Q = zeros(2*H-1,2*H-1); %x(i)-u_x(i)<=0
        Incon(i).Q(2*H-1,i) = 0.5;
        Incon(i).Q(i,2*H-1) = 0.5;
        Incon(i).c = -u_x(i);
    end
    for i = 1:2*H-2  %x2~x_{H-1}的取值范围
        Incon(2*H-2+i).Q = zeros(2*H-1,2*H-1); %-x(i)+l_x(i)<=0
        Incon(2*H-2+i).Q(2*H-1,i) = -0.5;
        Incon(2*H-2+i).Q(i,2*H-1) = -0.5;
        Incon(2*H-2+i).c = l_x(i);
    end
    
    %x3^2+4*x2^2 - 4*x2*x3 + x3^2+4*x2^2 - 4*x2*x3 - V^2*u_max^2*(t')^2 = 0
    Incon(4*H-3).Q = zeros(2*H-1,2*H-1); %确保无人机的加速度在整个路径规划过程中不超过其最大加速度限制
    Incon(4*H-3).Q(1,1) = 4;
    Incon(4*H-3).Q(1,2) = -2;
    Incon(4*H-3).Q(2,1) = -2;
    Incon(4*H-3).Q(2,2) = 1;
    Incon(4*H-3).Q(H-1,H-1) = 4;
    Incon(4*H-3).Q(H-1,H) = -2;
    Incon(4*H-3).Q(H,H-1) = -2;
    Incon(4*H-3).Q(H,H) = 1;
    Incon(4*H-3).Q(2*H-1,1) = -2*x_start;
    Incon(4*H-3).Q(1,2*H-1) = -2*x_start;
    Incon(4*H-3).Q(2*H-1,2) = x_start;
    Incon(4*H-3).Q(2,2*H-1) = x_start;
    Incon(4*H-3).Q(2*H-1,H-1) = -2*y_start;
    Incon(4*H-3).Q(H-1,2*H-1) = -2*y_start;
    Incon(4*H-3).Q(2*H-1,H) = y_start;
    Incon(4*H-3).Q(H,2*H-1) = y_start;
    Incon(4*H-3).Q(1,1) = 4;
    
    Incon(4*H-3).Q(2*H-2,2*H-2) = -V^2*u_max^2;
    Incon(4*H-3).c = x_start^2+y_start^2;
    
    for i = 1:H-4
        Incon(4*H-3+i).Q = zeros(2*H-1,2*H-1);
        Incon(4*H-3+i).Q(i,i) = 1;
        Incon(4*H-3+i).Q(i,i+1) = -2;
        Incon(4*H-3+i).Q(i,i+2) = 1;
        Incon(4*H-3+i).Q(i+1,i) = -2;
        Incon(4*H-3+i).Q(i+1,i+1) = 4;
        Incon(4*H-3+i).Q(i+1,i+2) = -2;
        Incon(4*H-3+i).Q(i+2,i) = 1;
        Incon(4*H-3+i).Q(i+2,i+1) = -2;
        Incon(4*H-3+i).Q(i+2,i+2) = 1;
        Incon(4*H-3+i).Q(H-2+i,H-2+i) = 1;
        Incon(4*H-3+i).Q(H-2+i,H-2+i+1) = -2;
        Incon(4*H-3+i).Q(H-2+i,H-2+i+2) = 1;
        Incon(4*H-3+i).Q(H-2+i+1,H-2+i) = -2;
        Incon(4*H-3+i).Q(H-2+i+1,H-2+i+1) = 4;
        Incon(4*H-3+i).Q(H-2+i+1,H-2+i+2) = -2;
        Incon(4*H-3+i).Q(H-2+i+2,H-2+i) = 1;
        Incon(4*H-3+i).Q(H-2+i+2,H-2+i+1) = -2;
        Incon(4*H-3+i).Q(H-2+i+2,H-2+i+2) = 1;
        Incon(4*H-3+i).Q(2*H-2,2*H-2) = -V^2*u_max^2;
        Incon(4*H-3+i).c = 0;
    end
    
    %x_{H-2}^2+4*x_{H-1}^2+20*x_{H-2}-40*x_{H-1}-4*x_{H-2}*x_{H-1}+
    %y_{H-2}^2+4*y_{H-1}^2+20*y_{H-2}-40*y_{H-1}-4*y_{H-2}*y_{H-1}
    %-V^2*u_max^2*(t')^2+200 = 0
    Incon(5*H-6).Q = zeros(2*H-1,2*H-1); 
    Incon(5*H-6).Q(H-3,H-3) = 1;
    Incon(5*H-6).Q(H-3,H-2) = -2;
    Incon(5*H-6).Q(H-2,H-3) = -2;
    Incon(5*H-6).Q(H-2,H-2) = 4;
    Incon(5*H-6).Q(2*H-1,H-3) = x_end;
    Incon(5*H-6).Q(H-3,2*H-1) = x_end;
    Incon(5*H-6).Q(2*H-1,H-2) = -2*x_end;
    Incon(5*H-6).Q(H-2,2*H-1) = -2*x_end;
    Incon(5*H-6).Q(2*H-5,2*H-5) = 1;
    Incon(5*H-6).Q(2*H-5,2*H-4) = -2;
    Incon(5*H-6).Q(2*H-4,2*H-5) = -2;
    Incon(5*H-6).Q(2*H-4,2*H-4) = 4;
    Incon(5*H-6).Q(2*H-1,2*H-5) = y_end;
    Incon(5*H-6).Q(2*H-5,2*H-1) = y_end;
    Incon(5*H-6).Q(2*H-1,2*H-4) = -2*y_end;
    Incon(5*H-6).Q(2*H-4,2*H-1) = -2*y_end;
    Incon(5*H-6).Q(2*H-2,2*H-2) = -V^2*u_max^2;
    Incon(5*H-6).c = x_end^2+y_end^2;
    
    %1-(xi-cj)^2/aj^2-(yi-dj)^2/bj^2 <= 0，避障约束，当无人机位于障碍物外部时取正值，而当无人机进入障碍物内部时取负值
    for j =1:m
        for i = 1:H-2
            Incon(5*H-6+(j-1)*(H-2)+i).Q = zeros(2*H-1,2*H-1);
            Incon(5*H-6+(j-1)*(H-2)+i).Q(i,i) = -1/a(j)^2;
            Incon(5*H-6+(j-1)*(H-2)+i).Q(H-2+i,H-2+i) = -1/b(j)^2;
            Incon(5*H-6+(j-1)*(H-2)+i).Q(2*H-1,i) = c(j)/a(j)^2;
            Incon(5*H-6+(j-1)*(H-2)+i).Q(2*H-1,H-2+i) = d(j)/b(j)^2;
            Incon(5*H-6+(j-1)*(H-2)+i).Q(i,2*H-1) = c(j)/a(j)^2;
            Incon(5*H-6+(j-1)*(H-2)+i).Q(H-2+i,2*H-1) = d(j)/b(j)^2;
            Incon(5*H-6+(j-1)*(H-2)+i).c = 1-c(j)^2/a(j)^2-d(j)^2/b(j)^2;
        end
    end
  
    % Solve the optimization problem for this segment
    output = irma(Q0, Incon, Eqcon); % Call to your IRMA function
end
function output = irma(Q0,Incon,Eqcon,options)

%Initialization
if nargin < 3
    error('At least three input arguments are required.')
elseif nargin == 3
%     options.w0 = 1;
    options.wt = 3;%<6e154(1.5,879),(3,324)
   
    options.delta = 1e-5;
    options.max = 300;
elseif nargin > 4
    error('Too many input arguments.')
end

% 获取目标矩阵Q0的大小
[nVar, mVar] = size(Q0);
%确保目标矩阵是方阵
if nVar ~= mVar
    error('Objective matrix should be a square matrix!')
end

% Initialize flag value
output.flag = 0;

% 初始化权重因子，这里默认初始权重因子w0为1
w = 3;% options.w0*options.wt;

% Find number of inequality and equality constraints
nIncon = length(Incon);
nEqcon = length(Eqcon);


% Start Timer
tic

% 通过半定松弛找到矩阵X
cvx_solver SeDuMi;
cvx_begin

% 定义未知矩阵X的大小
variable X(nVar,nVar);

% 定义目标函数，最小化Q0和X的迹乘积的迹
minimize (trace(Q0*X));

% Define equality and inequality quadratic constraints遍历不等式约束，并添加到约束中
subject to
for j=1:nIncon
    trace(Incon(j).Q*X) + Incon(j).c <= 0;
end
% 遍历等式约束，并添加到约束中
for j=1:nEqcon
    trace(Eqcon(j).Q*X) + Eqcon(j).c == 0;
end


% 将未知矩阵X放松为半定矩阵
X == semidefinite(nVar);
cvx_end

% 计算半定规划得到的矩阵X的特征向量和特征值
% EV将包含X的特征向量，这里只关心特征向量，所以~表示忽略特征值
[EV,~] = eig(X);%eig函数计算矩阵特征值和特征向量
EVi = EV;
clear X

minr=10;
% Start the iterative rank minimization loop
for kk=1:options.max
    w = w*options.wt;
    
    cvx_solver SeDuMi;
    cvx_begin
    variable X(nVar,nVar);
    variable r nonnegative;
    
     % 最小化原始代价函数和加权后的第二大特征值r
    % 目标是最小化Q0和X的迹乘积加上加权的r
    minimize (trace(Q0*X) + w*r);
    
    % Define equality and inequality quadratic constraints
    subject to
    for j=1:nIncon
        trace(Incon(j).Q*X) + Incon(j).c <= 0;
    end
    
    for j=1:nEqcon
        trace(Eqcon(j).Q*X) + Eqcon(j).c == 0;
    end
  
    
   % 约束X为正半定矩阵
    X == semidefinite(nVar);
    
  % 对X的第二大特征值施加半定约束
    % 约束r乘以单位矩阵减去EVi对应的特征向量与X的乘积为半定矩阵
    r*eye(nVar-1) - EVi(:,1:nVar-1)'*X*EVi(:,1:nVar-1) == semidefinite(nVar-1);%即式子>=0
    
    cvx_end
    
    % Record r value obtained at each step
    rk(kk) = r;
      
    % Find corresponding eigenvectors at each iteration step
    [EVi,~] = eig(X);%~表示忽略输出
    
    % Print out iteration index, objective value, weighting factor and r at
    % current step
    fprintf('kk=%d,trace(Q0*X) = %f,r = %f,w = %f\n',kk,trace(Q0*X),r,w);
    
    % When 'r' which represents the second largest eigenvlue of 'X' is
    % significantly small, we assume the unknown matrix 'X' is a rank one
    % matrix and the stopping criteria is satisfied
       % 当r值足够小，认为X矩阵接近秩1，满足停止条件
       if r<minr&&r>0
           minr=r;
           minX=X;
       end
           
    if (r<options.delta&&r>0)
        output.flag = 1;
        break;
    end
  
    if kk<options.max
        clear X r
    end
end
while (r<0)
     clear X r
     kk=kk+1;
      w = w*options.wt;
    cvx_solver SeDuMi;
    cvx_begin
    variable X(nVar,nVar);
    variable r nonnegative;
    minimize (trace(Q0*X) + w*r);
    subject to
    for j=1:nIncon
        trace(Incon(j).Q*X) + Incon(j).c <= 0;
    end
    for j=1:nEqcon
        trace(Eqcon(j).Q*X) + Eqcon(j).c == 0;
    end
    X == semidefinite(nVar);
    r*eye(nVar-1) - EVi(:,1:nVar-1)'*X*EVi(:,1:nVar-1) == semidefinite(nVar-1);%即式子>=0
    cvx_end
    rk(kk) = r;
    [EVi,~] = eig(X);%~表示忽略输出
    fprintf('kk=%d,trace(Q0*X) = %f,r = %f,w = %f\n',kk,trace(Q0*X),r,w);
     if r<minr&&r>0
           minr=r;
           minX=X;
     end
end   

    % Find the computation time
output.time = toc;

% 根据特征值和特征向量找到IRM算法的解
% x是最大特征值的平方根乘以对应的最大特征向量
[EVf,EVa] = eig(minX);
output.x = sqrt(max(max(EVa)))*EVf(:,end);%默认从小到大排序

% 输出目标函数的值
output.Jf = trace(Q0*minX);
% 输出迭代的步数
output.step = kk;
% 输出每次迭代的r值
output.r = rk;

end