clear;close all;
tic();
x0 = 0; y0 = 0; % 起点
xf = 6; yf = 6; % 终点
waypoints = [ 4,5 ;9,6; 9,10 ;10,10; 5,9]; % 经停点
pri_points=[10,1;8,5];%优先点
k=size(pri_points,1);
param.searchFeild = [-3,12,-3,12];   % 搜索区域
param.threshold = 1;   % d_{min}
param.theta = 5*pi/6;    % 论文中theta取150度
param.resolution = 10;   % 决定在路径上多远取几个点，以判断路径是否避开障碍
param.maxNodes = 5000;  %算法在停止之前会生成的最大节点数，最大迭代次数
param.step_size = 1;  %新节点向采样点生长时的最大步长
param.neighbourhood = 1; % 设置neighbourhood的半径r
param.random_seed =40;
param.u_max = 0.2;    %最大曲率  
param.obstacles=[2,1,2,3;
  3,3,7,2;
  2,2,2,7;
  2,2,8,8;];
m=size(param.obstacles,1);%障碍数
a = [2, 3, 2,2 ]; % 长轴
b = [1, 3, 2, 2]; % 短轴
c = [2, 7, 2, 8]; % 椭圆中心x坐标
d = [3, 2, 7, 8]; % 椭圆中心y坐标

global mindistance;
mindistance=100;
 global  minpath_x;
  global  minpath_y;
    minpath_x=[];
  minpath_y=[];
% 所有点集合
points = [x0, y0; waypoints; xf, yf];
n = size(points, 1); % 点的总数
global distMatrix ;
global xMatrix;
global yMatrix;
distMatrix = zeros(n, n);
xMatrix = [];
yMatrix = [];
% 求解TSP问题（起点固定为1，终点固定为最后一个点）
idx_start = 1; % 起点索引
idx_end = n; % 终点索引
% 使用遗传算法求解TSP问题
fitnessFunc =  @(order)tspFitness(order,points, idx_start, idx_end,param);
nWaypoints = n - 2; % 必经点数（不包括起点和终点）
% 遗传算法参数设置
options = optimoptions('ga', 'Display', 'iter', 'UseParallel', false);
options.PopulationSize = 50;    % 设置种群大小
options.MaxGenerations = 200;   % 设置最大代数
options.CrossoverFraction = 0.8; % 设置交叉概率
%  options.MigrationFraction = 0.01;    % 设置突变概率，默认0.2
options.Display = 'iter';       % 显示每代信息
%    options.CrossoverFcn=@crossover;
% options=gaoptimset('CrossoverFcn',@crossoverscattered);
%   options.intcon = 1:nWaypoints; % 指定离散变量
% IntCon = 1:nWaypoints;

% 采用全排列的个体表示（确保路径中点顺序唯一）
 order = ga(fitnessFunc, nWaypoints, [], [], [], [], ...
    ones(1, nWaypoints), nWaypoints * ones(1, nWaypoints) ,[],1:nWaypoints,options);

% 将结果转换为实际路径
optimalOrder = [idx_start, order + 1, idx_end]; % 起点 → 必经点 → 终点
optimalPath = points(optimalOrder, :);
% 显示最优路径点序列
disp('最优点访问顺序:');
disp(optimalOrder);

% 绘制路径
figure;
plot(minpath_x, minpath_y, '--bo', 'LineWidth', 2);
hold on;

% 标记起点、必经点和终点
plot(x0, y0, 'ko', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'g'); % 起点
plot(optimalPath(2:end-1, 1), optimalPath(2:end-1, 2), 'ko', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'y'); % 必经点
plot(xf, yf, 'ko', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'r'); % 终点

% 绘制障碍物
theta = linspace(0, 2*pi, 100);
for i = 1:m
    xp = c(i) + a(i) * cos(theta);
    yp = d(i) + b(i) * sin(theta);
    plot(xp, yp, '.r');
end
t=toc();
disp(['t=',num2str(t)])
% --- TSP 适应度函数 ---
function totalDistance = tspFitness(order,points, idx_start, idx_end,param)
   global distMatrix ;
   global xMatrix;
global yMatrix;
% 检查路径是否包含重复点
    if length(unique(order)) ~= length(order)
        totalDistance = inf; % 如果路径包含重复点，返回极大的适应度值
        return;
    end
    % 将顺序加上起点和终点
    fullOrder = [idx_start, order + 1, idx_end]; % 加入起点和终点
    current_points=points(fullOrder,:);
    num_points=size(current_points,1);
     totalDistance = 0;
     all_x = points(1,1); % 用于保存所有路径点的x坐标
all_y = points(1,2); % 用于保存所有路径点的y坐标
    for i = 1:num_points-1
     x_start = current_points(i, 1);
         y_start = current_points(i, 2);
         x_end = current_points(i+1, 1);
         y_end = current_points(i+1, 2);
      if distMatrix(fullOrder(i),fullOrder(i+1))~=0
             path_length=distMatrix(fullOrder(i),fullOrder(i+1)); 
             x_path=[];
             y_path=[];
             for j=1:size(xMatrix,3)
                 if xMatrix(fullOrder(i),fullOrder(i+1),j)==0 && yMatrix(fullOrder(i),fullOrder(i+1),j)==0&&j~=1
                     break;
                 else
              x_path(j)=xMatrix(fullOrder(i),fullOrder(i+1),j);
               y_path(j)=yMatrix(fullOrder(i),fullOrder(i+1),j);
               
                 end
             end
            
      else
          if ~check_obstacle(x_start, y_start, x_end, y_end,100,param.obstacles)
        % 如果没有障碍物，直接连接这两个点
        x_path = [x_start, x_end];
        y_path = [y_start, y_end];
          else
         p_start=[x_start;y_start];
                 p_end=[x_end; y_end];
                result = RRTstar12(param, p_start, p_end);  % 正确的调用
                 x_path = result.refinedP(1,:);%转置取消
              y_path = result.refinedP(2,:);     
          end
         
      path_length = 0;
      for j = 1:length(x_path) - 1
        path_length = path_length + norm([x_path(j+1), y_path(j+1)] - [x_path(j), y_path(j)]);
      end
     distMatrix(fullOrder(i),fullOrder(i+1))=path_length;
%     distMatrix(fullOrder(i+1),fullOrder(i))=path_length;
    for j=1:length(x_path)
     xMatrix(fullOrder(i),fullOrder(i+1),j)=x_path(j);
    yMatrix(fullOrder(i),fullOrder(i+1),j)=y_path(j);
    end
      end
       all_x = [all_x, x_path(2:end)];
    all_y = [all_y, y_path(2:end)];  
        totalDistance = totalDistance + path_length; % 计算路径长度
    end
    global mindistance;
  global  minpath_x;
  global  minpath_y;
    if mindistance>totalDistance
        mindistance=totalDistance;
        minpath_x=all_x ;
       minpath_y = all_y;
    end
    
end
