% How well do the equilibrium responses match?
% How well does the timescale match?
% What is the distribution of trends?
% How does the run length work? Does it follow a Poisson process?
% Is the dynamical model consistent with a Gaussian pdf?

%FIX THE EQUATIONS HERE _ THIS IS JUST DONE SLOPPILY_ For NOW....

clear all; 

%figure(1)
%-----------------
%define parameters
%-----------------
rho=910; %kg/m^3
g=9.81; %m/s^2
n=3;

%mu = 0.65; %melt rate in m /yr /degC
mu = 0.65; %melt rate in m /yr /degC
gamma = 6.5e-3; %lapse rate  % NOTE SHOULD PUT SOME THOUGHT INTO HOW WELL THIS IS KNOWN

% climate forcing
% sigT = 0.0; %degC
% sigP = 0.0; %m/yr
 sigT = 0.9; %degC
 sigP = 0.7; %m/yr
   
% values taken to match Anderson et al 06
fd = 1.9e-24; % Deformation parameter Pa^-3 s^-1 % default was 1.9e-24;
fs = 5.7e-20; % Sliding parameter Pa^-3 s^-1 m^2 % default was 5.7e-20; Anderson is 4.5e-20
%fs = 0;

fd = fd*pi*1e7; fs = fs*pi*1e7; % convert from seconds to years

xmx = 20000; % the domain size in m
delx = 100; %grid spacing in m
%delx = 200; %grid spacing in m
nxs = round(xmx/delx) + 1; %number of grid points

%delt = 0.0125; % time step in yrs
delt = 0.0125/4; % time step in yrs suitable for 200m
ts = 0; % starting time
tf = 10000; % final time
disp(['tf = ' num2str(tf)])
nts=floor((tf-ts)/delt) +1; %number of time steps ('round' used because need nts as integer)
nyrs = tf-ts;
x = 0:delx:xmx; % x array

%-----------------
% define functions
%------------------

% Load Oerlemans 86 Nigardsbreen geometry into 
load Oerlemans86_geom.mat

% interpolate onto model grid
zb = interp1(x_dat,zb_dat,x);
wb = interp1(x_dat,wb_dat,x);
theta = interp1(x_dat,atand(theta_dat),x);
zs = interp1(x_dat,zs_dat,x);

dzbdx=gradient(zb,x);				% slope of glacer bed geometries.

% interpolate these values out to 20km, but this should not be taken
% seriously
idx = find(isnan(dzbdx));
wb(idx) = wb(idx(1)-1);
dzbdx(idx) = dzbdx(idx(1)-1);
theta(idx) = theta(idx(1)-1);
zb(idx) = zb(idx(1)-1) + dzbdx(idx).*(x(idx)-x(idx(1)-1));
zs(idx) = zb(idx);

h_dat = zs-zb; % thickness based on Anderson data
h_dat(idx)=0;
idx = find(h_dat<10);
h_dat(idx) = 0;

%load initial glacier profile
load glac_init_nigardsbreen_PTmassbal_11km.mat
h0  = interp1(x_init,h_init,x,'linear');

% uncomment this line, and set nts = 1, to compare with Anderson's topography;
%h0 = h_dat;

Qp=zeros(size(x)); % Qp equals j+1/2 flux
Qm=zeros(size(x)); % Qm equals j-1/2 flux
h=zeros(size(x)); % zero out height array
dhdt=zeros(size(x)); % zero out thickness rate of change array
h(:) = h0; % initialize height array

% pick climate forcing
Tp = sigT*randn(nyrs+1,1); Pp = sigP*randn(nyrs+1,1);  % initialize climate forcing
%load ClimateRandom.mat % load the same set of 10,000 random numbers for clearer comparisons
%Trand = Tfac*Trand; Prand = Pfac*Prand;

% equilibrium climate for intial mass balance
P = ones(size(x))*(3.0);
T_wk    = (14.24)*ones(size(x)) - gamma*(zb+h);
melt = max(0,mu*T_wk);b = P-melt;
         
% equivalent linear model parameter 
i1 = find(h>0, 1, 'last' );
i2 = find(b>0, 1, 'last' );

% ablation area
Aabl = trapz(x(i2:i1),wb(i2:i1));
% total area
Atot = trapz(x(1:i1),wb(1:i1));

% area above melting
i3 = find(T_wk<0, 1, 'last' ); 
if isempty(i3)==1
    disp(['******************'])
    disp(['Problem with ATgt0'])
    disp(['******************'])
    ATgt0 = Atot;
else
    ATgt0 = trapz(x(i3:i1),wb(i3:i1));
end

hbar = mean(h(1:i1));
%hbar = 150;
w = mean(wb(i1-10:i1+10)); % width is average 10 grid points either side of terminus
tanphi = abs(mean(dzbdx(i1-5:i1+5))); % basal slope, same average as above

%linear model coefficients
% linear model coefficients
dt = 1;
tau = w*hbar/(mu*gamma*tanphi*Aabl);
alpha = mu*ATgt0*dt/(w*hbar);
beta = Atot*dt/(w*hbar);

%linear prediction of variance 
varC = (alpha^2*sigT^2+beta^2*sigP^2); % variance in climate forcing
varLp = tau/(2*dt)*varC; % variance in Lp

%%
%-----------------------------------------
% begin loop over time
%-----------------------------------------
yr = 0;
idx_out = 0;

deltout = 1;
nouts = round(nts/1);
edge_out = zeros(length(nyrs));t_out = zeros(length(nyrs));
accum_tot = zeros(length(nyrs));
melt_tot = zeros(length(nyrs));
mass_bal = zeros(length(nyrs));
sfc_area = zeros(length(nyrs));
ela = zeros(length(nyrs));

for i=1:nts

   t = delt*(i-1); % time in years        

% if it is a new year, do some things
% calculate new width profile
% define climate every year
    if (t == floor(t))
        yr = yr+1;
         disp(['yr = ' num2str(yr)])
   
 % add in valley shape factor
         ws = wb + 2*h.*tand(theta); % assumes 25 valley slopes(?)
         wtilde = wb + h.*tand(theta);
         dwtildedx = gradient(wtilde,x);
         
         %disp(['yr = ' num2str(yr)])
         P = ones(size(x))*(3.0+Pp(yr));
         accum = P;
         %14.32 works for L = 9km
         % 13.975 works for L = 12
         % 13.85 works for L = 13
         % 14.1125 works for L = 11
         % 14.24 works for L = 10
         % 13.7 works for L = 14
         T_wk    = (14.1125+Tp(yr))*ones(size(x)) - gamma*(zb+h);
         melt = max(0,mu*T_wk);
         b = P-melt;   
   
    end
    
%%
    
%-----------------------------------------
% begin loop over space
%-----------------------------------------
   for j=1:nxs-1  % this is a kloudge -fix sometime
       
       if j==1

           h_ave =(h(1) + h(2))/2;
           dhdx = (h(2) - h(1))/delx;
           dzdx = (dzbdx(1) + dzbdx(2))/2;

            Qp(1) = -(dhdx+dzdx).^3 * h_ave.^4*(rho*g)^3*(fd*h_ave+fs/h_ave); % flux at plus half grid point
            Qm(1) = 0; % flux at minus half grid point
            dhdt(1) = b(1) ...
                   - wtilde(1)/ws(1) * Qp(1)/(delx/2)...
                   - (Qp(1)+Qm(1))/(2*ws(1)) * dwtildedx(1);
           
       elseif h(j)==0 && h(j-1)>0 % glacier toe condition
        
           Qp(j) = 0;
           h_ave = h(j-1)/2;
           dhdx = -h(j-1)/delx;			% correction inserted ght nov-24-04
           dzdx = (dzbdx(j-1) + dzbdx(j))/2;
   
           Qm(j) = -(rho*g)^3*h_ave.^4*(dhdx + dzdx).^3*(fd*h_ave+fs/h_ave);
           dhdt(j) = b(j)...
               + wtilde(j)/ws(j) * Qm(j)/delx...
               - (Qp(j)+Qm(j))/(2*ws(j)) * dwtildedx(j);
           edge = j; 					%index of glacier toe - used for fancy plotting
     
       elseif h(j)<=0 && h(j-1)<=0 % beyond glacier toe - no glacier flux
  
           dhdt(j) = b(j);
           Qp(j) = 0; Qm(j) = 0;
           
       else  % within the glacier
 
           h_ave = (h(j+1) + h(j))/2;
           dhdx = (h(j+1) - h(j))/delx;		% correction inserted ght nov-24-04
           dzdx = (dzbdx(j) + dzbdx(j+1))/2;
         
           Qp(j) = -(rho*g)^3*h_ave.^4*(dhdx + dzdx).^3*(fd*h_ave+fs/h_ave);
           h_ave = (h(j-1) + h(j))/2;

           dhdx = (h(j) - h(j-1))/delx;
           dzdx = (dzbdx(j) + dzbdx(j-1))/2;

           Qm(j) = -(rho*g)^3*h_ave.^4*(dhdx + dzdx).^3*(fd*h_ave+fs/h_ave);
           dhdt(j) = b(j)...
               - wtilde(j)/ws(j) * (Qp(j) - Qm(j))/delx...
               - (Qp(j)+Qm(j))/(2*ws(j)) * dwtildedx(j);
       end
       
       dhdt(nxs) = 0; % enforce no change at boundary

%----------------------------------------
% end loop over space
%----------------------------------------
   end
    h = max(0 , h + dhdt*delt);

%----------------------------
% plot glacier every so often
% save h , and edge
%----------------------------
    if t/1 == floor(t/1)
        idx_out = idx_out+1;
        
        edge_out(idx_out) = edge; 
        t_out(idx_out) = t;
        edge_out(idx_out) = delx*find(h>10, 1, 'last' );
        
        %calculate mass balance, accum, melt, tot,
        idx = find(h>0);
        accum_wk = trapz(x(idx),accum(idx).*ws(idx));
        melt_wk = trapz(x(idx),melt(idx).*ws(idx));
       
        % divide by surface area
        sfc_area(idx_out) = trapz(x(idx),ws(idx));
        accum_tot(idx_out) = accum_wk/sfc_area(idx_out);
        melt_tot(idx_out) = melt_wk/sfc_area(idx_out);
        
        %tot_mass_bal
        mass_bal(idx_out) = accum_tot(idx_out) - melt_tot(idx_out);
        
       % ela(idx_out) = interp1(accum-melt,zb+h,0,'cubic');
        
        idx = find(zs>400,1,'last'); % this is a little fix b/c the force ends up not being monotonic increase b/c zb is a little weird near the toe
%        ela(idx_out) = interp1(accum(1:idx)-melt(1:idx),zb(1:idx)+h(1:idx),[0],'linear','extrap');
        
        %Tsumm(idx_out) = 0.25*sum(T_mnth(6:9));
        
        time = ['t = ' num2str(t) ' yrs'];
    end
    
    if t/20 == floor(t/20)
        
        figure(1); hold off
%         set(gcf,'Units','normalized')
         subplot('position',[0.1 0.40 0.8 0.50]) 
         x1 = [x(1:edge) fliplr(x(1:edge))]; z1 = [zb(1:edge) fliplr(zb(1:edge)+h(1:edge))];
         patch(x1/1000,z1/1000,'c')
         plot(x/1000,zb/1000,'k','linewidth',2)
         % silly plotting to make glacier look good
         x1 = [x(1:edge) fliplr(x(1:edge))]; z1 = [zb(1:edge) fliplr(zb(1:edge)+h(1:edge))]  ;
         patch(x1/1000,z1/1000,'c')
         axis([0 15 0.2 2.0]);
         xlabel('Distance (km)','fontsize',14); ylabel('Elevation (km)','fontsize',14)
         
         h3 = text('position',[0.77, 0.92],'string',time,'units','normalized','fontsize',16);
      
         grid on
         hold off    
                 
         disp('outputting')
%        hout(idx_out,:) = h;
         edge_out(idx_out) = edge; 
         t_out(idx_out) = t;
         edge_out(idx_out) = delx*find(h>10, 1, 'last' );
        
         figure(1)
         subplot('position',[0.1 0.1 0.8 0.2])
         plot(t_out,edge_out/1000,'k','linewidth',2); axis([0 10000 7 14]); hold off
         xlabel('Time (years)','fontsize',14); ylabel('Terminus (km)','fontsize',14)
         grid on
         drawnow
         
          %fname = ['MovieOut/' num2str(idx_out) '.png'];
          %print('-dpng', fname)   
        
    end
    
% calculate driving stress, deformation velocity and total velocity to compare with Oerelemans
%Us = zeros(size(x));Ud = zeros(size(x));tau = zeros(size(x));
%rho = 918;g = 9.8;
%tau = rho*g*h.*gradient(zb+h,x);

%idx = find(h>0);
%Us(idx) = fs*tau(idx).^3./h(idx);
%Ud = fd.*h.*tau.^3;
%-----------------------------------------
% end loop over time
%-----------------------------------------
end

