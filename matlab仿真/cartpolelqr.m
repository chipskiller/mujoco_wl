function K = cartpolelqr
global M m l g
%% state space matrices
A = [0 0 1 0;
     0 0 0 1;
     0 m*g/M 0 0;
     0 (m*g + M*g)/(M*l) 0 0]
B = [0;0;1/M;1/(M*l)]
C = eye(4);
D = 0;

%% build ss system:
cartpole = ss(A,B,C,D);

%%LQR
Q = diag([1000 20 0 500]);%x q dx dq
R = 200;%fx
K = lqr(cartpole,Q,R)

end