clear; close all;
% 初始化参数
tic();

% %case1
% startpoints =[ 0,0;10,10]; % 旅行商起点
% waypoints = [ 9, 6; 9, 10; 5.5, 9;4,2;0.5,5;0.5,1.5;2,8.5;9.5,4;10,7;7,10;3,4.8;6,6.5;4,6;0,4]; % 必经点
% param.obstacles=[2,2,2,3;
%   2,2,7,2;
%   2,1.5,2,7;
%   2,2,8,8;
%   1,1,4.5,5;
%   ];

% %case2
% startpoints =[ 0,0;5,10;10,0]; % 旅行商起点
% waypoints = [3.2,2;4,8.5; 6.5,7; 9.5,2;6,2;4,6.5;2,0.5;7,5;5,3.5;7.5,1;6,8.5;0.5,3;2.5,4;8.5,4;5.5,5;9,6.5;]; % 必经点
% param.obstacles=[1,1,1,1.5;
%   1.5,1,7,3;
%   1,1.5,2,7;
%   1.5,1,8,8;
%   1,1,3,7;
%   1,0.5,7,4;
%   0.5,1.5,5,1;
%   1,1,4,5;
%   0.5,0.5,5,7;
%   0.5,1,7,9;
%   1,0.5,5,9;
%   1,0.5,9,1;
%   1,1,2,3;
%   1,0.5,8,6;
%   ];



%case3
startpoints =[ 0,5;5,0;5,10;10,5]; % 旅行商起点
waypoints = [2,0;3,1;5,3;2,4;0.5,2;0,8;2,9;8,9.5;6,0;6,4;9,5;10,3;1,6;5,6;7.5,1;9.5,7.5;6.5,7;4,4.5;]; % 必经点
param.obstacles=[1,1,2,2;
  1,2,9,2;
  1,2,6,2;
  1,1,7,5;
  1,1,2,7;
  1,1,9,9;
  1,1,8,8;
  0.5,1,1,9;
  1,1,4,6;
  0.5,0.5,2,5;
  0.5,0.5,6,8.5;
  0.5,0.5,1,1;
  1,0.5,3.5,4;
  0.5,0.5,4,2;
  0.4,0.5,1,4;
  1,1,3.5,8.5;
  ];%长轴短轴xy

% %case4
% startpoints =[ 0,0;0,10;10,0;10,10]; % 旅行商起点
% waypoints = [ 2,7;3,3;5.5,3.5;7,0.5;9,7;7,9;2,1;4,6;3,9;7,3; 2,5;0.5,7.5;10,2;0.5,1.7;1,3.2]; % 必经点
% param.obstacles=[1,0.5,1,2.5;
%     1,1,0.5,4.5;
%     1,1,1,9;
%     0.5,0.5,1,1;
%     1,1,5,1;
%     1,1,4,2;
%     1,1,3.5,5;
%     1,1,5.5,9;
%     1,1,7.5,2;
%     2,2,7,6;
%     1,1,9.5,1;
%     1,1,9.5,4;
%     1,1,9,9;
%     1,1,3,7.5;
%     ];




param.searchFeild = [-3,12,-3,12];   % 搜索区域
param.threshold = 0.2;   % d_{min}
param.theta = 5*pi/6;    % 论文中theta取150度
param.resolution = 10;   % 决定在路径上多远取几个点，以判断路径是否避开障碍
param.maxNodes = 1000;  %算法在停止之前会生成的最大节点数，最大迭代次数
param.step_size = 0.3;  %新节点向采样点生长时的最大步长
param.neighbourhood = 1; % 设置neighbourhood的半径r
param.random_seed =40;
param.u_max = 0.2;    %最大曲率  

m=size(param.obstacles,1);%障碍数
n_travelers = size(startpoints,1); % 旅行商数量
alpha=0.5;
beta=0.5;
for i=1:m
a(i)= param.obstacles(i,1); % 长轴
b(i) = param.obstacles(i,2); % 短轴
c (i)= param.obstacles(i,3); % 椭圆中心x坐标
d (i)= param.obstacles(i,4); % 椭圆中心y坐标
end
global distMatrix ;
global xMatrix;
global yMatrix;
xMatrix = [];
yMatrix = [];
% 将起点和终点加到 waypoints 中
points = [startpoints;waypoints];
n = size(points, 1); % 总点数（包括起点和终点）
near_zone=4;%为无人机临近点范围
idx_start = 1; % 起点索引
idx_end = n; % 终点索引
distMatrix = zeros(n, n);
oudist_matrix = zeros(n, n);
for i = 1:n%欧几里得距离
    for j = 1:n
        oudist_matrix(i, j) = sqrt((points(i, 1) - points(j, 1))^2 + ...
                                 (points(i, 2) - points(j, 2))^2);
    end
end
% 遗传算法参数
pop_size = 50; % 种群大小
max_gen = 500;  % 最大代数
mutation_rate = 0.3; % 变异率
crossover_rate=0.9;%交叉率
tournament_size = 2; % 锦标赛选择大小
elite_size = 1; % 精英保留数量
best_fitness=1000;%最优路径长度
overnum=0;%提前结束迭代计数器
% 初始化种群：随机生成路径
pop = zeros(pop_size, n);
for i = 1:pop_size
    firsttraveler=ceil(rand()*n_travelers);%第一个旅行商
    order=randperm(n);
    firstidx=find( order==firsttraveler);%剔除冗余的旅行商
    order(firstidx)=[];
    pop(i, :) = [firsttraveler,order]; % 随机生成一个路径
end
% 遗传算法主循环
for gen = 1:max_gen    
    % 计算每个个体的适应度（路径长度）
    fitness = zeros(pop_size, 1);
    for i=1:pop_size
     fitness(i) = all_total_distance(pop(i, :),points,param , oudist_matrix, n_travelers,near_zone,alpha,beta);
    end
    % 适应度归一化，越大越好
%     fitness = 1 ./ (1 + fitness);
    % 选择操作（轮盘赌选择）
       new_pop = tournament_selection(pop, fitness, pop_size,tournament_size);%可能损失基因多样性
   % 交叉操作（部分匹配交叉PMX）
    new_pop = crossover2(new_pop, pop_size, n,crossover_rate);
    
    % 变异操作（交换变异）
    new_pop = mutation(new_pop, mutation_rate, n);
    %适应度不再变化时提前停止迭代
    if best_fitness==min(fitness)
    overnum=overnum+1;
    else
    overnum=0;
    end
    if overnum==100
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
    if mod(gen, 10) == 0
        disp(['Generation: ', num2str(gen), ' Best fitness: ', num2str(best_fitness)]);
    end
    
end

% 获取最优路径
best_path = pop(best_idx, :);
optimalPath = points(best_path, :);
best_path_length=best_fitness;
% 输出最优路径和最短距离
disp('最短路径:');
disp(points(best_path, :));
disp(best_path);
% disp(['最短距离: ', num2str(best_fitness)]);
disp(['最短距离: ', num2str(best_path_length)]);
% 可视化最优路径
figure;
for t=1:n_travelers
    traveleridx=find(best_path<=n_travelers,n_travelers);
    traveler_path=travelers_path_func(best_path, oudist_matrix, n_travelers,near_zone,traveleridx,t);
    current_points=points(traveler_path,:);
    num_points=size(current_points,1);
     all_x = current_points(1,1); % 用于保存所有路径点的x坐标
all_y = current_points(1,2); % 用于保存所有路径点的y坐标
    for i=1:num_points-1
        x_start = current_points(i, 1);
         y_start = current_points(i, 2);
         x_end = current_points(i+1, 1);
         y_end = current_points(i+1, 2);
             x_path=[];%清空上一次的路径
             y_path=[];
       for j=1:size(xMatrix,3)
                 if xMatrix(traveler_path(i),traveler_path(i+1),j)==0 && yMatrix(traveler_path(i),traveler_path(i+1),j)==0&&j~=1
                     break;
                 else
              x_path(j)=xMatrix(traveler_path(i),traveler_path(i+1),j);
               y_path(j)=yMatrix(traveler_path(i),traveler_path(i+1),j);
                 end
       end
       if current_points(1,1)==0&&current_points(1,2)==0&&i==num_points-1%路径回到起点（0，0）
           x_path(j+1)=0;
           y_path(j+1)=0;
       end
       all_x = [all_x, x_path(2:end)];
    all_y = [all_y, y_path(2:end)];  
    end
    plot(all_x,  all_y, '--', 'LineWidth', 2); 
    hold on;
end
hold on;
plot(waypoints(:,1),waypoints(:,2), 'ko', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'g'); % 经停点
plot(startpoints(:,1),startpoints(:,2), 'ko', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'r'); % 起点
for i = 1:size(waypoints, 1)
    text(waypoints(i, 1) + 0.2, waypoints(i, 2), ['', num2str(i+n_travelers)], 'Color', 'black', 'HorizontalAlignment', 'left');
end
for i = 1:size(startpoints, 1)
    text(startpoints(i,1) + 0.2, startpoints(i,2), ['', num2str(i)], 'Color', 'black', 'HorizontalAlignment', 'left');
end
% 绘制障碍物
theta = linspace(0, 2*pi, 200);
for i = 1:m
    xp = c(i) + a(i) * cos(theta);
    yp = d(i) + b(i) * sin(theta);
    plot(xp, yp, '.r');
end
t=toc();
disp(['t=',num2str(t)]);
function new_pop = crossover2(parents, pop_size, n,crossover_rate)
   new_pop = parents;
    for i = 1:2:pop_size
        p1 = parents(i, :);
        p2 = parents(i+1, :);
        % 随机选择交叉点
        crossover_point = randi([1, n]); % 随机选择一个交叉点
        % 创建掩码区域的拷贝
        child1 = p1;
        child2 = p2;
        if rand()<crossover_rate
        % 交换交叉点之间的基因
        temp = child1(crossover_point:end);
        child1(crossover_point:end) = child2(crossover_point:end);
        child2(crossover_point:end) = temp;
        % 确保剩余基因不重复（使用PMX处理非交叉部分）
        for j=crossover_point:n
            if ismember(child1(j),child1(1:j-1))
                missing_gene = p2(find(~ismember(p2, child1(1:n))));
                child1(j) = missing_gene(1);
            end
            if ismember(child2(j),child2(1:j-1))
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
function totalDistance = tspFitness(order,points,param)%单个无人机路径长度
   global distMatrix ;
   global xMatrix;
global yMatrix;
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
                 if xMatrix(fullOrder(i),fullOrder(i+1),j)==0 && yMatrix(fullOrder(i),fullOrder(i+1),j)==0&&j~=1&&x_path(1)~=0&&y_path(1)~=0
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
end
function len = all_total_distance(path,points,param ,oudist_matrix, n_travelers,near_zone,alpha,beta)
    len = 0;
     traveleridx=find(path<=n_travelers,n_travelers);%无人机的下标
     travelers_paths=zeros(n_travelers,length(path)+1);%保存所有无人机的路径
     max_len=0;
    for t = 1:n_travelers
       traveler_path=travelers_path_func(path, oudist_matrix, n_travelers,near_zone,traveleridx,t);
       for i=1:length(traveler_path)
       travelers_paths(t,i)=traveler_path(i);
       end
       len1=tspFitness(traveler_path,points,param);%当前无人机路径长度
       if len1>max_len
           max_len=len1;
       end   
      len = len +  len1;% 计算路径长度，每个无人机路径长度和
    end
    len=alpha*len+beta*max_len;%综合路径长度
    
    %多无人机防碰撞
    for t=1:n_travelers-1
        if len==inf%存在无人机发送碰撞
            break;
        end
    ii=1;
    while travelers_paths(t,ii)%将点存入x1x2，检测路径是否与其他无人机路径碰撞
        x1(ii)=points(travelers_paths(t,ii),1);
        y1(ii)=points(travelers_paths(t,ii),2);
      ii=  ii+1;
    end
    for i=1:length(x1)-1
        x2(i)=x1(i+1);
        y2(i)=y1(i+1);
    end
    x1(end)=[];
    y1(end)=[];
    
    for i=t+1:n_travelers%其他无人机路径
        jj=1;
        while travelers_paths(i,jj)
        x3(jj)=points(travelers_paths(i,jj),1);
        y3(jj)=points(travelers_paths(i,jj),2);
        jj=jj+1;
        end
    for i=1:length(x3)-1
        x4(i)=x3(i+1);
        y4(i)=y3(i+1);
    end
    x3(end)=[];
    y3(end)=[];
    if chack(x1,y1,x2,y2,x3,y3,x4,y4)
        len=inf;
        break;
    end
    end
    end
end