% Nonlinear network aging: Optimal control solver using TOMLAB/POPT
toms t

I_vec = [0, 0.15];%[0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25];
alpha_vec = [1:0.4:20];%[1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]; 
r_vec = [0:0.002:0.1];%[0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1];
f_vec = [0:0.002:0.1];%[0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1];

% Parameters
f = 0.025;
n = 100;
alpha = 10;
r = 0.01;
gamma = 0.975;

for alpha = alpha_vec
    T1_vec = [];
    T2_vec = [];
    for I = I_vec
        p = tomPhase('p', t, 0, 1000, 501);
        setPhase(p);
        tomStates x1 % this is phi
        tomControls u1 % this is r
        setPhase(p);
        % Initial guess
        if I == 0
            x0 = {icollocate({x1 == 1});
                collocate({u1 == 0})};% Box constraints
        else
            x0 = {icollocate({x1 == x1opt});
                collocate({u1 == u1opt})};% Use prev soln as initial guess
        end
        cbox = {0 <= icollocate(x1) <= 1;
        0 <= collocate(u1)  <= r};

        % Boundary constraints
        cbnd = initial({x1 == 1});

        % ODEs and path constraints
        if I == 0
            ceq = collocate({dot(x1) == -get_feff(I,x1,f,n)*x1 + get_h(I,x1,n)*u1*(1-x1)});
        else
            ceq = collocate({dot(x1) == -approx_feff(I,x1,f,n)*x1 + approx_h(I,x1,n)*u1*(1-x1)});
        end

            % Objective
        objective = integrate(exp(log(gamma)*t)*(alpha*u1-x1));
        %objective = integrate((alpha*u1-x1));

        options = struct;
        options.name = 'Linear Problem Bang';
        solution = ezsolve(objective, {cbox, cbnd, ceq}, x0, options);
        t_plt  = subs(collocate(t),solution);
        x1_plt = subs(collocate(x1),solution);
        u1_plt = subs(collocate(u1),solution);

        subplot(2,1,1)
        plot(t_plt,x1_plt,'*-');
        legend('x1');
        title('Linear Problem Bang state variables')

        subplot(2,1,2)
        plot(t_plt,u1_plt,'+-');
        legend('u1');
        title('Linear Problem Bang control');

        % Retrieve solution
        x1opt = subs(x1, solution);
        u1opt = subs(u1, solution);

        % Save T1 and T2 values
        best_val = 0;
        T1 = 0;
        T2 = 0;
        for t1=1:length(u1_plt)
            for t2=1:length(u1_plt)
                obj_val = mean(u1_plt(t1:t2))-mean(u1_plt(1:t1))-mean(u1_plt(t2:length(u1_plt)));
                if obj_val > best_val
                    best_val = obj_val;
                    T1 = t_plt(t1);
                    T2 = t_plt(t2);
                end
            end
        end
                
        %T1 = find( abs(u1_plt)>=0.005, 1);
        %T2 = find( abs(u1_plt)>=0.005, 1, 'last');

        T1_vec(end+1) = T1;
        T2_vec(end+1) = T2;
    end

    % Save Results to file (for specified alpha value)
    savematrix = [I_vec; T1_vec; T2_vec];
    csvwrite(['nonlin_alpha_',num2str(alpha),'.csv'], savematrix)
    %csvwrite(['nonlin_r_',num2str(r),'.csv'], savematrix)
    %csvwrite(['nonlin_f_',num2str(f),'.csv'], savematrix)
end
    
% Functions
function feff = get_feff(I, x1, f, n)
m = get_m(I,x1,n);
feff = f/(1-(I*n)*m/x1*(1-f));
end

function m = get_m(I, x1, n)
m = nchoosek(n,round(I*n)) * x1^(I*n) * (1-x1)^(n-(I*n));
end

function feff = approx_feff(I, x1, f, n)
m = approx_m(I,x1,n);
k = I*n;
%feff = (-2+f*(k-1)+2*k*m*(1-f)+sqrt(8*f*(k-1)+(2+f*(1-k)+2*(f-1)*k*m)^2))/(2*(k-1));
feff = 2*f/(1+(f-1)*k*m+(1+(f-1)*k*m*(2+2*f-2*f*k+(f-1)*k*m))^(1/2)); % 2nd order
%feff = f/(1-(I*n)*m/x1*(1-f)); 1st order
end

function m = approx_m(I, x1, n)
%m = nchoosek(n,round(I*n)) * x1^(I*n) * (1-x1)^(n-(I*n));
m = get_binom_approx(I*n, x1, n);
end

function h = approx_h(I, x1, n)
h = 0.5-0.5*erf((I*n-n*x1)/(sqrt(n*x1*(1-x1))));
end

function h = get_h(I, x1, n)
h = 0;
for i = 1:round(I*n)
     %h = h + nchoosek(n,i) * x1^i * (1-x1)^(n-i);
     h = h + get_binom_approx(i, x1, n);
end;
h = 1-h;
end

function num = get_binom_approx(k, x1, n)
num = 1/sqrt(2*pi*n*x1*(1-x1)) * exp(-(k-n*x1)^2/(2*n*x1*(1-x1)));
end