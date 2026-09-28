function Fx = LQRControl(states)
    
    global K_LQR
    %% desired commands:
    X_des = [4;0;0;0];

    %% control law
    Fx = K_LQR*(X_des - states);

    %% 输出限幅 ±20N
    Fx = max(min(Fx, 20), -20);   % ← 加这一行

end
