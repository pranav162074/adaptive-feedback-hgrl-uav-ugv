function [sorted_waypoints, total_distance] = greedy_order(x0,y0, xf,yf,waypoints)
    % 实现简单的贪婪算法，选择离当前点最近的必经点
    start_point=[x0,y0];
    end_point=[xf,yf];
    remaining_points = waypoints;
    current_point = start_point;
   % sorted_waypoints = [];
  sorted_waypoints = start_point;
    total_distance = 0;
    
    while ~isempty(remaining_points)
        % 计算距离
        distances = vecnorm(remaining_points - current_point, 2, 2);
        [min_dist, idx] = min(distances);
        total_distance = total_distance + min_dist;
        sorted_waypoints = [sorted_waypoints; remaining_points(idx, :)];
        current_point = remaining_points(idx, :);
        remaining_points(idx, :) = [];
    end
    
    % 最后加入终点
    sorted_waypoints = [sorted_waypoints; end_point];
    total_distance = total_distance + norm(end_point - current_point);
end