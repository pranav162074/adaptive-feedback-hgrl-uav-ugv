 function free=isObstacleFree(node_free,obstacles)   %4
        free = 0;
        for i = 1: length(obstacles(:,1)) %第一列数目，既障碍物数目
            obs = obstacles(i,:); %提取第i个障碍物信息给obs
            nx = node_free(1); %将待检验的横纵坐标赋值
            ny = node_free(2);
            ha = (nx-obs(3))^2 / obs(1)^2 + (ny-obs(4))^2 / obs(2)^2; %计算了节点到障碍物边界的距离与障碍物长轴和短轴的比值之和，obs(3) 和 obs(4) 是椭圆中心的 x 和 y 坐标，obs(1) 和 obs(2) 分别是椭圆的长轴和短轴长度
            if (ha < 1) %如果 ha 小于等于 1，意味着节点 (nx, ny) 落在了椭圆障碍物内部
                free = 1;
            end
        end 
    end