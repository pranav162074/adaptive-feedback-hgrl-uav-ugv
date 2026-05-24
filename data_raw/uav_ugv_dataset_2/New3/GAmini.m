clear; close all;
% 初始化参数
x0 = 0; y0 = 0; % 起点
xf = 10; yf = 10; % 终点
waypoints = [4, 5; 9, 6; 9, 10; 6, 6; 5, 9]; % 必经点
% 将起点和终点加到 waypoints 中
points = [x0, y0; waypoints; xf, yf];
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
mutation_rate = 0.01; % 变异率
tournament_size = 5; % 锦标赛选择大小
elite_size = 2; % 精英保留数量

% 初始化种群：随机生成路径
pop = zeros(pop_size, n);
for i = 1:pop_size
    pop(i, :) = [1, randperm(n-2) + 1, n]; % 随机生成一个路径，1为起点，n为终点
end

% 遗传算法主循环
for gen = 1:max_gen
    % 计算每个个体的适应度（路径长度）
    fitness = zeros(pop_size, 1);
    for i = 1:pop_size
        fitness(i) = path_length(pop(i, :), dist_matrix); % 计算路径长度
    end
    % 选择操作（轮盘赌选择）
    selected_parents = tournament_selection(pop, fitness, pop_size, tournament_size);

    % 交叉操作（部分匹配交叉PMX）
    new_pop = crossover(selected_parents, pop_size, n);
    
    % 变异操作（交换变异）
    new_pop = mutation(new_pop, mutation_rate, n);

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

% 输出最优路径和最短距离
disp('最短路径:');
disp(points(best_path, :));
disp(['最短距离: ', num2str(best_fitness)]);

% 可视化最优路径
figure;
plot(points(:, 1), points(:, 2), 'ro', 'MarkerSize', 8); % 所有点
hold on;
for i = 1:(length(best_path) - 1)
    plot(points(best_path(i:i+1), 1), points(best_path(i:i+1), 2), 'b-', 'LineWidth', 2);
end
text(points(:, 1), points(:, 2), string(1:n), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
grid on;
title('最优路径');
xlabel('X');
ylabel('Y');

% 计算路径长度的函数
function len = path_length(path, dist_matrix)
    len = 0;
    for i = 1:(length(path) - 1)
        len = len + dist_matrix(path(i), path(i+1));
    end
end
function new_pop = crossover(parents, pop_size, n)
    new_pop = zeros(pop_size, n);
    for i = 1:2:pop_size
        p1 = parents(i, :);
        p2 = parents(i+1, :);
        
        % 随机选择交叉点，确保 crossover_point2 大于 crossover_point1
        crossover_point = randi([2, n-2]); % 随机选择一个交叉点
        % 创建掩码区域的拷贝
        child1 = p1;
        child2 = p2;
        
        
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
