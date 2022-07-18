% This code solves the shallow-ice equations for ice flow down a 
% flowline with prescribed bed geometry. See Roe, JGlac, 2011, or 
% many, many other places for equations.

% You can provide a time series of climate (accumlation and melt-season 
% temperature). The code then calculates integrates the shallow-ice 
% equations.

% The standard set of parameters are chosen(ish) to be typical of small Alpine
% glaciers around Mt. Baker, WA. The flowline assumes constant width (1D),
% so the glacier is longer than typical around Mt Baker in order to
% emulate a realistic accumulation area. 

% This version of the model was just just for fun, so the parameters are
% different from those in Roe, JGalc, 11. 

% The code uses an explicit numberical scheme, so is not particularly
% stable. If it blows up, try and shorter time step (delt), or a coarser
% spatial resolution (delx). I am a crappy coder, so there are no
% guarantees it is free of bugs.

% I have tuned the mean glacier length primarily by specifying average 
% melt-season temperature at z=0, and mean-annual accumulation, Pbar.

% Right now the code loads in an initial glacier profile from the file 
% "glac_init_baker.mat". If you change parameters, you may want to
% integrate the glacier in time with climate anomalies set to zero until
% the glacier attains equilibrium, save that profile, and use it as the
% initial conditions for your new integrations (if that makes sense).

% You can change the basal geometry, glacier size, climate, flow
% parameters,


% Feel free to use for anything, and to change anything. Have fun! 
% Gerard (groe@uw.edu - although negligible technical support will be offered!). 

clear all; close all

figure(1)
%-----------------
%define parameters
%-----------------
rho=910; %kg/m^3
g=9.81; %m/s^2
n=3;
A=2.4e-24; %Pa^-3 s^-1
K=2*A*(rho*g)^n/(n+2);
K=K*pi*1e7; % make units m^2/yr
K=K;

xmx = 20000; % the domain size in m
delx = 200; %grid spacing in m
%delx = 50; %grid spacing in m
nxs = round(xmx/delx) + 1; %number of grid points

%delt = 0.00125; % time step in yrs. needed if delx = 50
delt = 0.025; % time step in yrs
ts = 0; % starting time
tf = 10000; % final time
nts=floor((tf-ts)/delt) +1; %number of time steps ('round' used because need nts as integer)

%----------------------------
% climate parameters
%----------------------------
Tbar = 20.5;    % average temperature at z=0 [^oC]
sigT = 0.9;     % standard deviation of temperature [^oC]

Pbar = 3.0;     % average value of accumulation [m yr^-1]
sigP = 1.0;     % standard deviation of accumulation [m yr^-1]

gamma = 6.5e-3; % lapse rate  [K m^-1]
mu = 0.5;       % melt factor for ablation [m yr^-1 K^-1]

x = [0:delx:xmx]; % x array
%-----------------
% define functions
%-----------------
b=(3-(4/5000)*x); %mass balance in m/yr

%---------------------------------
% different glacier bed geometries
%---------------------------------
%zb=2000-.1*x; %bed profile in m
%zb =  2000.*exp(-x/4000);
del = 15e3/log(3);
zb =  3000.*exp(-x/del);
%zb = 2000-(.1*x) + 200*(1-exp(-x/4000));
%zb = 1000*ones(size(x));

dzbdx=gradient(zb,x);				% slope of glacer bed geometries.
%Q=(2*x(j)-(4/5000)*(x(j).^2)/2)*(1e-7/pi); 	%mass flux m^2/s

% Not needed if you load your own initial profile in.
%------------------------------------------
% % find zeros of flux to find glacier length
% %------------------------------------------
% % Note oerlemans value is c=10
% c=1; %equals 2tau/(rho g) plastic ice sheet constant from paterson
% idx=find(cumsum(b)<0); % indices where flux is less than zero
% idx=min(idx);
% %L=x(idx); %approximate length of glacier
% L = 20e3;
% 
% %------------------------------------------
% % plastic glacier initial profile
% %------------------------------------------
% h0=zeros(size(x)); % zero out initial height array
% h0(1:idx-1)=sqrt(c*(L-x(1:idx-1))); %plastic ice profile as initial try

%----------------------------------------------------------------------------------------------
% alternative starting geometry for a glacierusing results from a previus integration
% stored in glac_init.mat
% probably only worth it for mass balance parameterizations similar to the default one we tried
%----------------------------------------------------------------------------------------------
load glac_init_baker.mat
h0  = interp1(x_init,h_init,x,'pchip');

Qp=zeros(size(x)); % Qp equals j+1/2 flux
Qm=zeros(size(x)); % Qm equals j-1/2 flux
h=zeros(size(x)); % zero out height array
dhdt=zeros(size(x)); % zero out thickness rate of change array
h(:) = h0; % initialize height array

%aviobj = avifile('glacmovie.avi','fps',10,'compression','indeo3','quality',20);

nyrs = tf-ts;
Pout=zeros(nyrs,1);Tout=zeros(nyrs,1);
%-----------------------------------------
% begin loop over time
%-----------------------------------------
yr = 0;
idx_out = 0;

deltout = 5;
nouts = round(nts/5);
edge_out = zeros(length(nouts));t_out = zeros(length(nouts));

for i=1:nts

   t = delt*(i-1); % time in years 
    %t(i)
    
% define climate if it is the start of a new year
    if (t == floor(t))
        yr = yr+1;
         disp(['yr = ' num2str(yr)])
         P = ones(size(x))*(Pbar+sigP*randn(1));
         T_wk    = (Tbar+sigT*randn(1))*ones(size(x)) - gamma*zb;
         Pout(yr) = P(1); Tout(yr)=T_wk(1);
%          if yr<500;
%              P = ones(size(x))*(3.0);
%              T_wk    = (17.5)*ones(size(x)) - 6.5e-3*zb;
%          else
%              P = ones(size(x))*(3.0);
%              T_wk    = (20.5)*ones(size(x)) - 6.5e-3*zb;
%          end
        melt = max(0,mu*T_wk);
        b = P-melt;
%        b(yr,:)=(3-(4/5000)*x); %mass balance in m/yr
    end
    
%     b(i) = P(i) - melt(i);
%   if t(i) >= 250 
%       idx = find(b>0);
%       b(idx) = 4*b0(idx);
%   end
    
    
%-----------------------------------------
% begin loop over space
%-----------------------------------------
   for j=1:nxs-1  % this is a kloudge -fix sometime
       
       if j==1
%           disp(['condn 1: ' num2str(j)])
           h_ave =(h(1) + h(2))/2;
           dhdx = (h(2) - h(1))/delx;
           dzdx = (dzbdx(1) + dzbdx(2))/2;
           Qp(1) = -K*(dhdx+dzdx).^3 * h_ave.^5; % flux at plus half grid point
           Qm(1) = 0; % flux at minus half grid point
           dhdt(1) = b(1) - Qp(1)/(delx/2);
       elseif h(j)==0 && h(j-1)>0 % glacier toe condition
 %          disp(['condn 2: ' num2str(j)])          
           Qp(j) = 0;
           h_ave = h(j-1)/2;
           dhdx = -h(j-1)/delx;			% correction inserted ght nov-24-04
           dzdx = (dzbdx(j-1) + dzbdx(j))/2;
           Qm(j) = -K*(dhdx + dzdx).^3 * h_ave.^5;
           dhdt(j) = b(j) + Qm(j)/delx;
           edge = j; 					%index of glacier toe - used for fancy plotting
       elseif h(j)<=0 && h(j-1)<=0 % beyond glacier toe - no glacier flux
  %         disp(['condn 3: ' num2str(j)])
           dhdt(j) = b(j);
           Qp(j) = 0; Qm(j) = 0;
       else  % within the glacier
   %        disp(['condn 4: ' num2str(j)])
           h_ave = (h(j+1) + h(j))/2;
           dhdx = (h(j+1) - h(j))/delx;		% correction inserted ght nov-24-04
           dzdx = (dzbdx(j) + dzbdx(j+1))/2;
           Qp(j) = -K*(dhdx+dzdx).^3 * h_ave.^5;
           h_ave = (h(j-1) + h(j))/2;
%           dhdx = (h(i,j-1) - h(i,j))/delx;
           dhdx = (h(j) - h(j-1))/delx;
           dzdx = (dzbdx(j) + dzbdx(j-1))/2;
           Qm(j) = -K*(dhdx+dzdx).^3 * h_ave.^5;
           dhdt(j) = b(j) - (Qp(j) - Qm(j))/delx;
       end
       
       dhdt(nxs) = 0; % enforce no change at boundary

%----------------------------------------
% end loop over space
%----------------------------------------
   end
    h = max(0 , h + dhdt*delt);

%----------------------------
% plot glacier every so often
% save h , and terminus position
%----------------------------
    if t/20 == floor(t/20)
        idx_out = idx_out+1;
        
        edge_out(idx_out) = edge; 
        t_out(idx_out) = t;
        edge_out(idx_out) = delx*max(find(h>10));
         
        time = ['t = ' num2str(t) ' yrs'];
                
        
        figure(1); hold off
%         set(gcf,'Units','normalized')
         subplot('position',[0.1 0.40 0.8 0.50]) 
         x1 = [x(1:edge) fliplr(x(1:edge))]; z1 = [zb(1:edge) fliplr(zb(1:edge)+h(1:edge))];
         patch(x1/1000,z1/1000,'c')
         plot(x/1000,zb/1000,'k','linewidth',2)
         % silly plotting to make glacier look good
         x1 = [x(1:edge) fliplr(x(1:edge))]; z1 = [zb(1:edge) fliplr(zb(1:edge)+h(1:edge))]  ;
         patch(x1/1000,z1/1000,'c')
         axis([0 10 1.3 3.2]);
         xlabel('Distance (km)','fontsize',14); ylabel('Elevation (km)','fontsize',14)
         
         h3 = text('position',[8.0, 3.3],'string',time,'units','normalized','fontsize',16);
      
         grid on
         hold off    
         
        
         disp('outputting')
%        hout(idx_out,:) = h;
         edge_out(idx_out) = edge; 
         t_out(idx_out) = t;
         edge_out(idx_out) = delx*max(find(h>10));
        
         figure(1)
         subplot('position',[0.1 0.1 0.8 0.2])
         plot(t_out,edge_out/1000,'k','linewidth',2); axis([0 10000 7 10]); hold off
         xlabel('Time (years)','fontsize',14); ylabel('Terminus (km)','fontsize',14)
         grid on
         
         drawnow
         
         % print out figures. Will slow you down, but you can make movies
         % fname = ['MovieOut/' num2str(idx_out) '.jpg'];
         % print('-djpeg', fname)
         %frame = getframe(gcf);
         %aviobj = addframe(aviobj,frame);
   
        
    end

%-----------------------------------------
% end loop over time
%-----------------------------------------
end

%aviobj = close(aviobj);
