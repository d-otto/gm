% The code implements the three-stage glacier model of Roe and Baker,
% JGlac, 2014 (RB14). You provide a time series of climate anomalies (either accumlation
% and melt-season temperature *or* annual-mean mass balance anomalies. The
% code then calculates the time series of the glacier response for the
% specified glacier parameters.

% The standard set of parameters are those used in 
% RB14 (see also Roe, JGlac, 2011), chosen to be typical of small Alpine
% glaciers around Mt. Baker, WA.

% Feel free to use for anything. Note that there are analytic solutions for
% the basic statistical properties of the three-stage model (given in RB14), 
% so you don't need to run this a bajillion times to propagate uncertainties.

% Have fun! Gerard (groe@uw.edu)

clear all

% define time array.
tf = 2000;          % length of integration [yrs]
ts = 0;             % starting year
dt = 1;             % time step [keep at 1yr always]
nts = (tf-ts)/dt+1; % number of time steps
t = ts:dt:tf;       % array of times [yr]

%% glacier model parameters
mu = 0.65;      % melt factor [m yr^-1 K^-1]
Atot = 3.95e6;  % total area of the glacier [m^2]
ATgt0 = 3.4e6;  % area of the glacier where some melting occurs [m^2]
Aabl = 1.95e6;  % ablation area [m^2] 
w = 500;        % characteristic width of the glacier tongue [m]. 
dt = 1;         % incremental time step [yr]
hbar = 44.4186; % characteristic ice thickness near the terminus [m]
gamma = 6.5e-3; % assumed surface lapse rate [K m^-1] 
tanphi = 0.4;   % assumed basal slope [no units]

% natural climate variability - for temperature and precipitation forcing
sigP = 1.0;     % std. dev. of accumulation variability [m yr^-1]
sigT = 0.8;     % std. dev. of melt-season temperature variability [m yr^-1]
% natural climate variability - for mass balance forcing
%sigb = 1.5;     % std. dev. of annual-mean mass balance [m yr^-1]


%% linear model coefficients, combined from above parameters
%% play with their values by choosing different numbers...
alpha = mu*ATgt0*dt/(w*hbar);
beta = Atot*dt/(w*hbar);

% glacier memory [ys]
% this is the glacier response time  (i.e., memory) based on the above glacier geometry
% if you like, just pick a different time scale to see what happens. 
% Or also, use the simple, tau = hbar/b_term, if you know the terminus
% balance rate from, e.g., observations
tau = w*hbar/(mu*gamma*tanphi*Aabl);

% coefficient needed in the model integrations
% keep fixed - they are intrinsic to 3-stage model
eps = 1/sqrt(3);
phi = 1-dt/(eps*tau);

% note at this point you could create your own climate time series, using
% random forcing, trends, oscillations etc.
% Tp = array of melt-season temperature anomalise
% Pp = array of accumlation anomalies
% bp = array of mass balance anomalies
    Tp = sigT*randn(nts,1);
    Pp = sigP*randn(nts,1);
%   bp = sigb*randn(nts,1);


L3s = zeros(nts,1); % create array of length anomalies

%% integrate the 3 stage model equations forward in time
for i = 5:nts
    L3s(i) = 3*phi*L3s(i-1)-3*phi^2*L3s(i-2)+1*phi^3*L3s(i-3)...
     + dt^3*tau/(eps*tau)^3 * (beta*Pp(i-3) - alpha*Tp(i-3));
% if you want to use mass balance anomalies instead comment out the 2 lines
% above, and uncomment the 2 lines below
% L3s(i) = 3*phi*L3s(i-1)-3*phi^2*L3s(i-2)+1*phi^3*L3s(i-3)...
%        + dt^3*tau/(eps*tau)^3 * (beta*bp(i-3));

end

%% if you have the matlab filter function, you can use the following,
%% which is a little more elegant...
r3s = [1-1/(eps*tau) 1-1/(eps*tau) 1-1/(eps*tau)];
a3s = poly(r3s);c3s =[0 0 1 0];
Cwk = dt^3/(eps^3*tau^2)*(beta*Pp - alpha*Tp); %climate forcing
%Again use below this is for bp forcing
%Cwk = dt^3/(eps^3*tau^2)*beta*bp; 
y3s = filter(c3s,a3s,Cwk);
%%
% plot the three stage model output
figure(1); clf
subplot(2,1,1)
plot(t,L3s,'k','linewidth',2)
grid on; hold on
xlabel('Time (yrs)','fontsize',14)
ylabel('Length anomaly (m)','fontsize',14)
title('Full integration','fontsize',14)
axis([0 nts -1000 +1000]) % change this axis limits if needed

subplot(2,1,2)
plot(t,L3s,'k','linewidth',2)
grid on; hold on
xlabel('Time (yrs)','fontsize',14)
ylabel('Length anomaly (m)','fontsize',14)
title('500-yr slice','fontsize',14)
axis([0 500 -1000 +1000]) % change this axis limits if needed




