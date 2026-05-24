clear; close all;
% 初始化参数
tic();
x0 = 0; y0 = 0; % 起点

%case1:不返回起点
% xf = 10; yf = 10; % 终点
% waypoints = [4, 5; 9, 6; 9, 10; 6, 6; 5, 9]; % 必经点
% param.obstacles=[2,1,2,3;
%   3,3,7,2;
%   2,2,2,7;
%   2,2,8,8;];

% %case1:返回起点
% xf = 0; yf = 0; % 终点
% waypoints = [2,8;9,9;8,2;]; % 必经点
% param.obstacles=[4,4,5,5;];
 
% %case2:
% xf = 0; yf = 0; % 终点
% waypoints = [ 4,5 ;9,6; 9,10 ;6,6; 5,9;10,3;6,1]; % 必经点
% param.obstacles=[2,1,2,3;
%   2,2,7,3;
%   2,2,2,7;
%   2,2,8,8;];

% % case3：
% xf = 0; yf = 0; % 终点
% waypoints = [ 3,5 ;9,6; 9,10 ;6,6.5; 5,9;10,3;6,1]; % 必经点
% param.obstacles=[1,1.5,2,3;
%   1.5,1,7,3;
%   1,1.5,2,7;
%   1.5,1,8,8;
%   1,1,3,7;
%   1,2,7,4;
%   0.5,1.5,4,0;
%   1,1,5,5;
%   0.5,0.5,5,7;
%   1,1,7,9;
%   ];

%case4:
xf = 0; yf = 0; % 终点
waypoints = [ 2,7;2,2;4,4;6,2;9,2;9,6;7,9;3.5,1;10,10 ]; % 必经点
param.obstacles=[1,1,1,2.5;
    1,1,0.5,4.5;
    1,1,1,8;
    0.5,0.5,2.5,1;
    1,1,5,1;
    1,1,4,2;
    1,1,3.5,5;
    1,1,5.5,9;
    1,1,7.5,2;
    2,2,7,6;
    1,1,9.5,1;
    1,1,9.5,4;
    1,1,9,9;
    ];

% %case5:
% xf = 0; yf = 0; % 终点
% waypoints = [ 1,6;4,8;10,10;9,2;4,1.5;6,4 ]; % 必经点 
% param.obstacles=[2,2,1.5,4;
%     1,1,2,7.5;
%     1,1,3,1.5;
%     1,1,5,3;
%     1.5,1.5,5,6.5;
%     1,1,6.5,1;
%     1,1,8,2;
%     1,1,7.5,5;
%     2,2,8.5,8;
%     ];
    


param.searchFeild = [-3,12,-3,12];   % 搜索区域
param.threshold = 1;   % d_{min}
param.theta = 5*pi/6;    % 论文中theta取150度
param.resolution = 10;   % 决定在路径上多远取几个点，以判断路径是否避开障碍
param.maxNodes = 1000;  %算法在停止之前会生成的最大节点数，最大迭代次数
param.step_size = 1;  %新节点向采样点生长时的最大步长
param.neighbourhood = 1; % 设置neighbourhood的半径r
param.random_seed =40;
param.u_max = 0.2;    %最大曲率  

num_obs=size(param.obstacles,1);%障碍数
for i=1:num_obs
a(i)= param.obstacles(i,1); % 长轴
b(i) = param.obstacles(i,2); % 短轴
c (i)= param.obstacles(i,3); % 椭圆中心x坐标
d (i)= param.obstacles(i,4); % 椭圆中心y坐标
end
global mindistance;
mindistance=100;
 global  minpath_x;
  global  minpath_y;
    minpath_x=[];
  minpath_y=[];
global distMatrix ;
global xMatrix;
global yMatrix;
xMatrix = [];
yMatrix = [];
% 将起点和终点加到 points 中
points = [x0, y0; waypoints; xf, yf];
n = size(points, 1); % 总点数（包括起点和终点）
idx_start = 1; % 起点索引
idx_end = n; % 终点索引
distMatrix = zeros(n, n);
% 遗传算法参数
pop_size = 50; % 种群大小
max_gen = 500;  % 最大代数
mutation_rate = 0.2; % 变异率
crossover_rate=0.8;%交叉率
tournament_size = 5; % 锦标赛选择大小
elite_size = 1; % 精英保留数量
best_fitness=1000;%最优路径长度
overnum=0;%提前结束迭代计数器
% 初始化种群：随机生成路径
pop = zeros(pop_size, n);
for i = 1:pop_size
    pop(i, :) = [1, randperm(n-2) + 1, n]; % 随机生成一个路径，1为起点，n为终点
end
% 遗传算法主循环
for gen = 1:max_gen    
    % 计算每个个体的适应度（路径长度）
    fitness = zeros(pop_size, 1);
    for i=1:pop_size
    fitness(i)=tspFitness(pop(i,:),points,param);   
    end
    % 适应度归一化，越大越好
%     fitness = 1 ./ (1 + fitness);
    % 选择操作（轮盘赌选择）锦标赛
%       new_pop = tournament_selection(pop, fitness, pop_size,tournament_size);%可能损失基因多样性
   % 交叉操作（部分匹配交叉PMX）
    new_pop = crossover2(pop, pop_size, n,crossover_rate);
    
    % 变异操作（交换变异）
    new_pop = mutation(new_pop, mutation_rate, n);
    %适应度不再变化时提前停止迭代
    if best_fitness==min(fitness)
    overnum=overnum+1;
    else
    overnum=0;
    end
    if overnum==50
        break;
    end
    % 精英保留
    for j=1:elite_size
    [best_fitness, best_idx] = min(fitness);
    new_pop(j, :) = pop(best_idx, :);
fitness(j)=fitness(best_idx);
    end
    % 更新种群
    pop = new_pop;
    % 输出当前代数和最优路径长度
%     if mod(gen, 10) == 0
        disp(['Generation: ', num2str(gen), ' Best fitness: ', num2str(best_fitness)]);
%     end
    
end

% 获取最优路径
best_path = pop(best_idx, :);
optimalPath = points(best_path, :);
best_path_length=best_fitness;
% 输出最优路径和最短距离
disp('最短路径:');
disp(points(best_path, :));
% disp(['最短距离: ', num2str(best_fitness)]);
disp(['最短距离: ', num2str(best_path_length)]);
% 可视化最优路径
figure;
% plot(minpath_x, minpath_y, '--bo', 'LineWidth', 2);
plot(minpath_x, minpath_y, '--', 'LineWidth', 2);
hold on;
% 标记起点、必经点和终点
plot(x0, y0, 'ko', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'g'); % 起点
plot(optimalPath(2:end-1, 1), optimalPath(2:end-1, 2), 'ko', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'y'); % 必经点
plot(xf, yf, 'ko', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'r'); % 终点
for i = 2:size(optimalPath, 1)-1
    text(optimalPath(i, 1) + 0.2, optimalPath(i, 2), ['Waypoint', num2str(i-1)], 'Color', 'black', 'HorizontalAlignment', 'left');
end
% 绘制障碍物
theta = linspace(0, 2*pi, 200);
for i = 1:num_obs
    xp = c(i) + a(i) * cos(theta);
    yp = d(i) + b(i) * sin(theta);
    plot(xp, yp, '.r');
end
t=toc();
disp(['t=',num2str(t)]);
function new_pop = crossover2(parents, pop_size, n,crossover_rate)
    new_pop = zeros(pop_size, n);
    for i = 1:2:pop_size
        p1 = parents(i, :);
        p2 = parents(i+1, :);
        % 随机选择交叉点，确保 crossover_point2 大于 crossover_point1
        crossover_point = randi([2, n-2]); % 随机选择一个交叉点
        % 创建掩码区域的拷贝
        child1 = p1;
        child2 = p2;
         if rand<crossover_rate
        % 交换交叉点之间的基因
        temp = child1(crossover_point:n-1);
        child1(crossover_point:n-1) = child2(crossover_point:n-1);
        child2(crossover_point:n-1) = temp;
        % 确保剩余基因不重复（使用PMX处理非交叉部分）
        for j = crossover_point:-1:2
            if ismember(child1(j), child1(j+1:n-1)) % 如果这个基因已经在交叉区域出现过
                % 替换为未出现在交叉区域的基因
                missing_gene = p2(find(~ismember(p2, child1(1:n))));
                child1(j) = missing_gene(1);
            end
            if ismember(child2(j), child2(j+1:n-1)) % 同样处理child2
                missing_gene = p1(find(~ismember(p1, child2(1:n))));
                child2(j) = missing_gene(1);
            end
        end
        end
        % 将交叉结果放入新种群
        new_pop(i, :) = child1;
        new_pop(i+1, :) = child2;
         
    end
end
% 锦标赛选择
function selected = tournament_selection(pop, fitness, pop_size, tournament_size)
    selected = zeros(pop_size, size(pop, 2));
    for i = 1:pop_size
        tournament_idx = randperm(pop_size, tournament_size);%产生1到pop_size中tournament_size个随机数
        [~, best_idx] = min(fitness(tournament_idx));
        selected(i, :) = pop(tournament_idx(best_idx), :);
    end
end

% 变异操作
function new_pop = mutation(pop, mutation_rate, n)
    new_pop = pop;
    for i = 1:size(pop, 1)
        if rand < mutation_rate
            % 随机选择两个位置并交换
            mutation_points = randperm(n-1, 2);
            if mutation_points(1)==1
                mutation_points(1)=2;
            end
            if mutation_points(2)==1
                mutation_points(2)=2;
            end
            temp = new_pop(i, mutation_points(1));
            new_pop(i, mutation_points(1)) = new_pop(i, mutation_points(2));
            new_pop(i, mutation_points(2)) = temp;
        end
    end
end
function totalDistance = tspFitness(order,points,param)
   global distMatrix ;
   global xMatrix;
global yMatrix;
% 检查路径是否包含重复点
%     if length(unique(order)) ~= length(order)
%         totalDistance = inf; % 如果路径包含重复点，返回极大的适应度值
%         return;
%     end
    % 将顺序加上起点和终点
    fullOrder =  order; % 加入起点和终点
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
                 if xMatrix(fullOrder(i),fullOrder(i+1),j)==0 && yMatrix(fullOrder(i),fullOrder(i+1),j)==0&&j~=1&&j~=2
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
