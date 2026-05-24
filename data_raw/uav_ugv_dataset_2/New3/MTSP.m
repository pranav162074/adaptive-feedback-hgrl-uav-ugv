clear; close all;
% 初始化参数
startpoints =[ 0,0;0,10;10,0;10,10]; % 旅行商起点
waypoints = [1,0.5;2,0;3,1;5,3;2,4;1,3;0,2;0,8;2,9;8,10;6,0;6,4;9,5;10,3;2,7;6,8;8,7;7,3;];%必经点
% 所有点集合，包括三个旅行商的起点
points = [startpoints;waypoints];
n = size(points, 1); % 总点数（包括起点和终点）

% 计算点之间的距离矩阵
dist_matrix = zeros(n, n);
for i = 1:n
    for j = 1:n
        dist_matrix(i, j) = sqrt((points(i, 1) - points(j, 1))^2 + ...
                                 (points(i, 2) - points(j, 2))^2);
    end
end
% 遗传算法参数
pop_size = 50; % 种群大小
max_gen = 500;  % 最大代数
mutation_rate = 0.2; % 变异率
tournament_size = 5; % 锦标赛选择大小
elite_size = 2; % 精英保留数量
n_travelers = size(startpoints,1); % 旅行商数量
n_waypoints = size(waypoints, 1); % 必经点数量
alpha=0.5;
beta=0.5;

% 初始化种群：随机生成路径
pop = zeros(pop_size, n);
for i = 1:pop_size
    firsttraveler=ceil(rand()*n_travelers);
    order=randperm(n);
    firstidx=find( order==firsttraveler);
    order(firstidx)=[];
    pop(i, :) = [firsttraveler,order]; % 随机生成一个路径
end

% 遗传算法主循环
for gen = 1:max_gen
    % 计算每个个体的适应度（路径长度）
    fitness = zeros(pop_size, 1);
    for i = 1:pop_size
        fitness(i) = total_distance(pop(i, :), dist_matrix, n_travelers,alpha,beta);
    end
   
    % 选择操作（轮盘赌选择）
     new_pop = tournament_selection(pop, fitness, pop_size, tournament_size);
  
    % 交叉操作
  
      new_pop = crossover(new_pop, pop_size, n);
    % 变异操作
     new_pop = mutation(new_pop, mutation_rate, n);

    % 精英保留
    for j = 1:elite_size
        [best_fitness, best_idx] = min(fitness);
        new_pop(j, :) = pop(best_idx, :);
        fitness(j) = fitness(best_idx);
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

% 输出最优路径和最短距离
disp('最短路径:');
disp(points(best_path, :));
disp(['最短距离: ', num2str(best_fitness)]);

% 可视化最优路径
figure;
plot(points(:, 1), points(:, 2), 'ro', 'MarkerSize', 8); % 所有点
hold on;
plot(startpoints(:,1),startpoints(:,2), 'ko', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'r'); % 起点
   % 为每个旅行商分配路径
 traveleridx=find(best_path<=n_travelers,n_travelers);
    for t = 1:n_travelers
        if t==n_travelers
             traveler_path=[best_path(traveleridx(t):end),best_path(traveleridx(t))];
             for i = 1:(length(traveler_path) - 1)
                 plot(points(traveler_path(i:i+1), 1), points(traveler_path(i:i+1), 2), 'b-', 'LineWidth', 2);
             end
        else
                traveler_path = [best_path(traveleridx(t):(traveleridx(t+1)-1)),best_path(traveleridx(t))];
                for i = 1:(length(traveler_path) - 1)
                 plot(points(traveler_path(i:i+1), 1), points(traveler_path(i:i+1), 2), 'b-', 'LineWidth', 2);
                end
        end
    end
text(points(:, 1), points(:, 2), string(1:n), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
grid on;
title('最优路径');
xlabel('X');
ylabel('Y');

% 计算路径总长度的函数
function len = total_distance(path, dist_matrix, n_travelers,alpha,beta)
    len = 0;
     traveleridx=find(path<=n_travelers,n_travelers);
     max_len=0;
    for t = 1:n_travelers
        % 为每个旅行商分配路径
        if t==n_travelers
             traveler_path=[path(traveleridx(t):end),path(traveleridx(t))];
        else
                traveler_path = [path(traveleridx(t):(traveleridx(t+1)-1)),path(traveleridx(t))];
        end
        len1=0;
        for j=1:length(traveler_path)-1
            len1 = len1+ dist_matrix(traveler_path(j), traveler_path(j+1)); % 计算路径长度
        end
        if len1>max_len
           max_len=len1; 
        end
        len=len+len1;
    end
    len=alpha*len+beta*max_len;
end

function new_pop = crossover(parents, pop_size, n)
    new_pop = parents;
    for i = 1:2:pop_size
        p1 = parents(i, :);
        p2 = parents(i+1, :);
        
        % 随机选择交叉点，确保 crossover_point2 大于 crossover_point1
        crossover_point = randi([1, n]); % 随机选择一个交叉点
        % 创建掩码区域的拷贝
        child1 = p1;
        child2 = p2;
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

% 部分匹配交叉 (PMX)



% 变异操作
function new_pop = mutation(pop, mutation_rate, n)
    new_pop = pop;
    for i = 1:size(pop, 1)
        if rand < mutation_rate
            % 随机选择两个位置并交换
            mutation_points = randperm(n, 2);
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