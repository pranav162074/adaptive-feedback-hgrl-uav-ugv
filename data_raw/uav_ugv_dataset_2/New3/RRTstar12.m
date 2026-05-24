function result = RRTstar12(param, p_start, p_goal)
%RRT探索区域
if p_start(1,1)>p_goal(1,1)
    if p_start(2,1)>p_goal(2,1)
        param.searchFeild=[p_goal(1,1)-2,p_start(1,1)+2,p_goal(2,1)-2,p_start(2,1)+2];
    else
        param.searchFeild=[p_goal(1,1)-2,p_start(1,1)+2,p_start(2,1)-2,p_goal(2,1)+2];
    end
else
        if p_start(2,1)>p_goal(2,1)
        param.searchFeild=[p_start(1,1)-2,p_goal(1,1)+2,p_goal(2,1)-2,p_start(2,1)+2];
    else
        param.searchFeild=[p_start(1,1)-2,p_goal(1,1)+2,p_start(2,1)-2,p_goal(2,1)+2];
        end
end
% param : 问题的参数 
%   1) threshold : 停止条件（目标和当前节点之间的距离）
%   2) maxNodes : RRT 树的最大节点数 
%   3) neighborhood : 用于寻找邻居节点的距离限制
%   4) obstacle : 障碍物 #限制
%   5) step_size : 无人机一次能够移动的最大距离（必须等于邻域大小） #限制
%   6) random_seed : 用于控制随机数生成
%                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
% 变量命名 : 在描述节点时，如果名称中带有 'node'，则表示该节点的坐标，否则表示节点在 RRT 树中的索引
% rrt 结构 : 1) p : 坐标, 2) iPrev : 父节点索引, 3) cost : 路径的总代价（欧几里得距离）
% 障碍物只能在端点处检测到，而不能沿着两个点之间的直线检测到
field1 = 'p';      % position
field2 = 'iPrev';  % parent父节点索引
field3 = 'cost';%距离
field4 = 'goalReached';
%定义字符串变量，边钰结构体中调用
rng(param.random_seed);%随机数生成
%tic;%开始计时
tic_rrt = tic();
start();

function start()
    rrt(1) = struct(field1, p_start, field2, 0, field3, 0, field4, 0);%初始化rrt树，创建结构体数组rrt
    N = param.maxNodes; % iterations N是最大迭代次数既最大节点数目
    j = 1;%迭代次数计数器，跟踪当前迭代进度

     while ~rrt(end).goalReached && j <= N %当goalreached为真或到达最大迭代次数时停止
%     while ~rrt(end).goalReached 
        sample_node = getSample(param.searchFeild);%在搜索区域内选取节点坐标
        nearest_node_ind = findNearest(rrt, sample_node);%最近节点
        new_node = steering(rrt(nearest_node_ind).p, sample_node);%通过 steering 函数将最近节点向目标节点方向延伸，生成新的节点
        if ObsFree(rrt(nearest_node_ind).p, new_node, param.resolution) == 1 %碰撞检测，若返回值为1，将新节点加入rrt树中
            neighbors_ind = getNeighbors(rrt, new_node);%寻找新节点的临近节点
            parent_node_ind = nearest_node_ind;
            if ~isempty(neighbors_ind) % new_node 附近至少有一个现有的 RRT 树节点
                parent_node_ind = chooseParent(rrt, neighbors_ind, nearest_node_ind, new_node);
            end
            rrt = insertNode(rrt, parent_node_ind, new_node);%如果通过碰撞检测，加入rrt树中
            %if norm(sample_node-p_goal) <= param.threshold %计算样本节点距离目标接待您的距离，若小于threshold则到达
            if norm(new_node-p_goal) <= param.threshold %计算样本节点距离目标接待您的距离，若小于threshold则到达
                rrt = setReachGoal(rrt);
            end
        end
        j = j + 1;
    end
    time_rrt = toc(tic_rrt);
    result.time_rrt = time_rrt;
    result.pa = setPath(rrt);%返回路径到pa字段中
    result.rrt = rrt;
    tic_rrtstar = tic();
    p = fliplr(result.pa);  % flip matrix result.pa left to right
    refinedP = refinePath(p, param.u_max);% get the improved path
    result.refinedP = refinedP;

    time_rrtstar = toc(tic_rrtstar);
    
    % 存储 RRT* 优化路径的时间
    result.time_rrtstar = time_rrtstar;
    

end



% Other functions (refinePath, steeringEval, circleCenter, pathOpt, getSampleP, ellipse, setReachGoal, setPath, getFinalResult, ObsFree, isObstacleFree, steering, reWire, insertNode, chooseParent, getCostFromRoot, getNeighbors, getSample, findNearest) remain the same as in the original code, without commented parts.
function refinedP = refinePath(p,u_max)
        % 验证方向角变化速度是否在规定的最大变化速度之内
        % 按照文献Two Approaches for Path Planning of Unmanned Aerial 
        % Vehicles with Avoidance Zones中的理论，steeringEval()函数为真时，
        % 退出以下while循环，但是"|| length(p)>200"这个条件时，一直不能退出循环
        % 希望以后有人能完善一下，或者谁不定作者理论有问题呢？
        refinedP = [];%初始化一个空的优化后路径
        max_refine_num=100;
            current_refine_num=1;
        while isempty(refinedP) %若redinedp为空则循环
            p = pathOpt(p); %调用了 pathOpt 函数对路径 p 进行优化
            current_refine_num=current_refine_num+1;
            if steeringEval(p,u_max) || length(p)>50 ||current_refine_num==max_refine_num%调用 steeringEval 函数检查路径是否满足转向速度的限制，如果满足转向速度限制，或者路径长度超过200个点则推出循环
                refinedP = p; %将优化后的路径p赋值给refinedp
            end
        end
        
    end

    function state = steeringEval(p,u_max) %评估路径转向性能的函数
        % p的格式i：[p1，p2，p3]_{2x3}，其中p1,p2,p3是相邻的边的三个端点
        state = 1; %将状态 state 设置为1，表示路径的转向性能符合要求
        for i = 1:size(p,2)-2 %获取p的列数并遍历
            pf= cross([p(:,i);0]-[p(:,i+1);0],[p(:,i);0]-[p(:,i+2);0]); %计算路径上相邻点构成的两个向量的叉乘，以判断是否共线
            if all(pf == 0) % 如果三点共线，跳出当前循环
                continue;
            else
                R = circleCenter([p(:,i);0],[p(:,i+1);0],[p(:,i+2);0]);
            end
            K = 1 / R;   % K是曲率
            u = K; %用u表示路径转向性能
            if u > u_max
                state = 0;
                break;
            end
        end
    end

    function r = circleCenter(p1, p2, p3)
        % CircleCenter(p1, p2, p3) 根据三个空间点，计算出其圆心，再求得R
        % p1,p2,p3:三个空间点在网上找的程序
        % 圆的法向量
        pf= cross(p1-p2, p1-p3);   
        % 两条线段的中点，之后需要求中垂线
        p12 = (p1 + p2)/2;
        p23 = (p2 + p3)/2;
        % 求两条线的中垂线
        p12f = cross(pf, p1-p2);
        p23f = cross(pf, p2-p3);
        % 求在中垂线上投影的大小
        ds = ( (p12(2)-p23(2))*p12f(1) - (p12(1)-p23(1))*p12f(2) ) /...
        	( p23f(2)*p12f(1) - p12f(2)*p23f(1) );
        % 得出距离
        centre = p23 + p23f .* ds;
        r = norm(centre-p1);
    end
%a=0;
%b=0;
    function pa = pathOpt(p) %优化路径p
        pa = [];
        maxAttempts = 500; % 设置最大尝试次数
        attempt = 0;
        while isempty(pa)&&attempt < maxAttempts %循环直到pa不为空
            attempt = attempt + 1;
            % 改进由RRT*算法生成的路径P
            [num1,p1] = getSampleP(p); %获取两个采样点 p1 和 p2，并获取它们的索引 num1 和 num2
            [num2,p2] = getSampleP(p);
            while num1 == num2  %当num1等于num2时，重新采样
                [num1,p1] = getSampleP(p);
                [num2,p2] = getSampleP(p);
            end
          %  a=num1;
           % b=num2;
            if ObsFree(p1,p2,100) %判断p1和p2之间有无障碍物
                if num1 < num2 %矩阵切片操作来实现路径的合并
                    pa = [p(:,1:num1),p1,p2,p(:,num2+1:end)];
                else
                    pa = [p(:,1:num2),p2,p1,p(:,num1+1:end)];
                end
            end
        end
        if attempt == maxAttempts
        warning('Maximum number of attempts reached without finding an improved path.');
        pa =p;
        end
    end

    function [num, point] = getSampleP(p)
        %在RRT*算法生成的路径P上随机取一点point
        length = size(p, 2);  %获取组成路径的点数（包括终点和起点），p的列数
        num = unidrnd(length - 1);  %生成一个范围为 1 到 length - 1 的随机整数
        point = p(:,num) + (p(:,num+1) - p(:,num))*rand(1); %在线段上随机选取一点
    end


    function rrt=setReachGoal(rrt) %到达终点
        rrt(end).goalReached = 1;
    end
    

    function Path = setPath(rrt) %绘制rrtstar生成的路径
       
        [cost,i] = getFinalResult(rrt);
        result.cost = cost;
        result.rrt = [rrt.p];
      
        Path = [p_goal,rrt(i).p];
        while i ~= 0 %当i为0时，退出循环，从最优路径的最后一个节点开始，逐步回溯到起始节点
            p11 = rrt(i).p;
           
            i = rrt(i).iPrev; %更新 i 的值为当前节点的父节点索引
            if i ~= 0
                p22 = rrt(i).p;  
               
                Path = [Path,p22];
            end 
        end  
 %       result.time_taken = toc; %使用了 toc 函数来计算从调用 tic 开始到当前位置的时间间隔
         
         
    end

    function [value,min_node_ind] = getFinalResult(rrt)
        goal_ind = find([rrt.goalReached]==1);
        if ~(isempty(goal_ind))
            disp('Goal has been reached!');
            rrt_goal = rrt(goal_ind);
            value = min([rrt_goal.cost]); %取最短路径的节点
            min_node_ind = find([rrt.cost]==value);
            if length(min_node_ind)>1
                min_node_ind = min_node_ind(1);
            end
        else
%             disp('Goal has not been reached!');
            for i =1:length(rrt)
                norm_rrt(i) = norm(p_goal-rrt(i).p); %算每个节点到目标点 p_goal 的欧氏距离
            end
            [~,min_node_ind]= min(norm_rrt); 
            value = rrt(min_node_ind).cost; %找到最近的节点，并将cost赋值给value
        end
    end

    function free = ObsFree(node1,node2,n) %n>=2,代表将线段分的段数，检查node1和2之间分段数n下是否触碰障碍
        free = 1;
        dx = (node2(1)-node1(1)) / n; %计算每段线段的步长 dx 和 dy
        dy = (node2(2)-node1(2)) / n;
        for i = 1:n
            if ~isObstacleFree([node1(1)+i*dx, node1(2)+i*dy]) %n段中，第i段的坐标，判断其是否触碰障碍
                free = 0;
                break
            end
        end
    end

    % if it is obstacle-free, return 1.
    % otherwise, return 0
    function free=isObstacleFree(node_free)   %4
        free = 1;
        for i = 1: length(param.obstacles(:,1)) %第一列数目，既障碍物数目
            obs = param.obstacles(i,:); %提取第i个障碍物信息给obs
            nx = node_free(1); %将待检验的横纵坐标赋值
            ny = node_free(2);
            ha = (nx-obs(3))^2 / obs(1)^2 + (ny-obs(4))^2 / obs(2)^2; %计算了节点到障碍物边界的距离与障碍物长轴和短轴的比值之和，obs(3) 和 obs(4) 是椭圆中心的 x 和 y 坐标，obs(1) 和 obs(2) 分别是椭圆的长轴和短轴长度
            if (ha < 1) %如果 ha 小于等于 1，意味着节点 (nx, ny) 落在了椭圆障碍物内部或边界上
                free = 0;
            end
        end 
    end
    
    function new_node=steering(nearest_node, random_node)   %3 从最近节点（nearest_node）沿着方向向随机节点（random_node）移动一定距离
       dist = norm(random_node-nearest_node);
       ratio_distance = param.step_size/dist; %计算一个比例因子，用于确定移动的步长
       if ratio_distance < 1
           x = (1-ratio_distance).* nearest_node(1)+ratio_distance .* random_node(1);
           y = (1-ratio_distance).* nearest_node(2)+ratio_distance .* random_node(2);
           new_node = [x;y];
       else %如果比例因子大于等于 1，说明最近节点到随机节点的距离已经比给定的步长要短，这时无需插值，直接将随机节点作为新节点
           new_node = random_node;
       end
    end
    
    function rrt = reWire(rrt, neighbors, parent, new) %8 重新连接 RRT 中的节点，优化路径探索更多可能
        for i=1:length(neighbors)
            cost = rrt(new).cost + norm(rrt(neighbors(i)).p - rrt(new).p);
            
            if (cost<rrt(neighbors(i)).cost)
                rrt(neighbors(i)).iPrev = new;
                rrt(neighbors(i)).cost = cost;
            end
        end
    end
    

    function rrt = insertNode(rrt, parent, new_node)   %7 向 RRT 中插入新节点
        rrt(end+1) = struct(field1, new_node, field2, parent, field3,...
            rrt(parent).cost + norm(rrt(parent).p-new_node), field4, 0);
    end
    
    function parent = chooseParent(rrt, neighbors, nearest, new_node)  %6 用于选择新节点的父节点
        min_cost = getCostFromRoot(rrt, nearest, new_node); %计算从最近节点到新节点的代价
        parent = nearest; %将最近的节点设置为新节点的初始父节点
        for i=1:length(neighbors) %遍历所有与新节点相邻的节点
            if ObsFree(rrt(neighbors(i)).p, new_node, param.resolution) == 1 
            cost = getCostFromRoot(rrt, neighbors(i), new_node);
            else
                cost=inf;
            end
            if (cost<min_cost) %如果当前相邻节点到新节点的代价小于最小代价
               min_cost = cost; %更新距离
               parent = neighbors(i); %将当前相邻节点设置为新节点的父节点
            end
        end
    end
    
    function cost = getCostFromRoot(rrt, parent, child_node)    %6.1
         
       cost =  rrt(parent).cost + norm(child_node - rrt(parent).p); %计算了从根节点到给定子节点的代价 rrt(parent).cost获取父节点到根节点的代价。在 RRT 树中，每个节点都记录了到根节点的路径代价
       
    end

    function neighbors = getNeighbors(rrt, node)    %5 获取与给定节点在一定邻域内的相邻节点
        neighbors = [];
        for i = 1:length(rrt)
            dist = norm(rrt(i).p-node); %计算给定节点与当前节点之间的距离
            if (dist<=param.neighbourhood) %如果距离小于等于给定的邻域大小（参数 param.neighbourhood），则说明当前节点在给定节点的邻域内
               neighbors = [neighbors i]; %将当前节点的索引添加到相邻节点数组中
            end
        end        
    end
    
    function node = getSample(sfeild)   %1 从指定搜索范围内获取随机样本的函数
        ax = sfeild(1);  % ax是x的搜索范围的下界
        bx = sfeild(2);  % bx是x的搜索范围的上界
        ay = sfeild(3);  % ay是y的搜索范围的下界
        by = sfeild(4);  % by是y的搜索范围的上界
        node=[0;0]; %初始化一个名为 node 的变量，它是一个二维向量，表示将要返回的随机采样点的坐标
        free = 0; %用于表示当前获取的样本是否在障碍物自由区域内。初始值为 0，表示不自由
        while ~free
            node(1) = (bx-ax) * rand(1) + ax; %这一行生成一个随机数，表示 x 轴上的坐标值，范围在 ax 和 bx 之间
            node(2) = (by-ay) * rand(1) + ay;
            if isObstacleFree(node) %检查生成的随机样本是否位于障碍物自由区域内
                free = 1;
            end
        end
    end
    
    
    function indx = findNearest(rrt, n)   %2 一组点集合中找到距离目标点最近的点，n是目标点
        mindist = norm(rrt(1).p - n); %计算第一个点和目标点之间的距离
        indx = 1;
        for i = 2:length(rrt)
            dist = norm(rrt(i).p - n);
            if (dist<mindist)
               mindist = dist;
               indx = i;
            end
        end
    end 

end