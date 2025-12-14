% ================= SATELLITE NMPC – EXERCISE 1 =================
clear; clc; close all;

% Parameters
Ts   = 0.2;                 % Sampling time (as in statement)
Tf   = 20*60;               % Max simulation time (20 minutes)
Nsim = floor(Tf/Ts);

nx = 12;                    % [R(:); omega]
nu = 3;                     % torque

N  = 10;                    % NMPC horizon

J = diag([125.734 216.211 234.055]);

% Initial condition
eul_ref = deg2rad((rand(3,1)-ones(3,1)*0.5)*360);   % desired orientation
R0 = eul2rotm(eul_ref', 'ZYX');    % MATLAB has some specific functions XD
w0 = rand(3,1);
x  = [R0(:); w0];

% Reference (Euler angles -> Rotation matrix)
eul_ref = deg2rad([30; -70; 132]);   % desired orientation
Rref = eul2rotm(eul_ref', 'ZYX');    % MATLAB has some specific functions XD
xref = [Rref(:); zeros(3,1)];

% Cost matrices
Q = diag([50*ones(9,1); 5*ones(3,1)]); % care more about position than velocity
P = Q;
R = 0.01*eye(nu);

Qbar = blkdiag(kron(eye(N),Q),P);
Rbar = kron(eye(N),R);

% Optimization options
opts = optimoptions('fmincon',...
    'Display','none',...
    'Algorithm','sqp',...
    'MaxIterations',200);

% Logs
xlog = zeros(nx,Nsim);
ulog = zeros(nu,Nsim);

% ================= MAIN SIMULATION LOOP =================
for k = 1:Nsim
    xlog(:,k) = x;

    % Solve NMPC
    u = attController(x, xref, N, nx, nu,Ts, Qbar, Rbar, opts, J);

    ulog(:,k) = u;

    % Integrate dynamics
    [~,y] = ode45(@(t,y) satelliteDynamics(y,u,J),[0 Ts],x);
    x = y(end,:)';

    % Stop condition (0.1 deg)
    Re = reshape(x(1:9),3,3)'*Rref;
    ang_err = acos((trace(Re)-1)/2);
    if rad2deg(abs(ang_err)) < 0.1
        fprintf('Converged at t = %.2f s\n',k*Ts);
        break;
    end
end

%% ================= PLOTS =================
figure;
plot(rad2deg(vecnorm(ulog)));
title('Control effort (deg)');
grid on;

figure;
for i = 1:1:k
    clf;
    plotSatelliteAxes(reshape(xlog(1:9,i),3,3));
    title(sprintf('t = %.1f s',i*Ts));
    axis([-2 2 -2 2 -2 2]);
    drawnow;
end

% ================= FUNCTIONS =================
function u = attController(x0, xref, N, nx, nu, Ts, Qbar, Rbar, opts, J)

    % Decision variables: [X; U]
    XU0 = zeros((N+1)*nx + N*nu,1);

    cost = @(XU) (XU - [repmat(xref,N+1,1); zeros(N*nu,1)])' ...
                  * blkdiag(Qbar,Rbar) ...
                  * (XU - [repmat(xref,N+1,1); zeros(N*nu,1)]);

    nonlcon = @(XU) dynamicsConstraint(XU, x0, N, nx, nu, Ts, J);

    XU = fmincon(cost,XU0,[],[],[],[],[],[],nonlcon,opts);

    u = XU((N+1)*nx + (1:nu));
end

function [c,ceq] = dynamicsConstraint(XU, x0, N, nx, nu, Ts, J)

    c = [];
    ceq = [];

    X = reshape(XU(1:(N+1)*nx),nx,N+1);
    U = reshape(XU((N+1)*nx+1:end),nu,N);

    ceq = [ceq; X(:,1) - x0];

    for k = 1:N
        xnext = X(:,k) + Ts*satelliteDynamics(X(:,k), U(:,k), J);
        ceq = [ceq; X(:,k+1) - xnext];
    end
end

function xdot = satelliteDynamics(x,u,J)

    S = @(w)[  0   -w(3)  w(2);
              w(3)   0  -w(1);
             -w(2) w(1)   0 ];

    R = reshape(x(1:9),3,3);
    w = x(10:12);

    Rdot = R*S(w);
    wdot = J \ (-S(w)*J*w + u);

    xdot = [Rdot(:); wdot];
end

function plotSatelliteAxes(R)
    O = [0 0 0];
    hold on; grid on; axis equal;
    quiver3(O(1),O(2),O(3),R(1,1),R(2,1),R(3,1),'r','LineWidth',2);
    quiver3(O(1),O(2),O(3),R(1,2),R(2,2),R(3,2),'g','LineWidth',2);
    quiver3(O(1),O(2),O(3),R(1,3),R(2,3),R(3,3),'b','LineWidth',2);
    xlabel('X'); ylabel('Y'); zlabel('Z');
    view(3);
end
