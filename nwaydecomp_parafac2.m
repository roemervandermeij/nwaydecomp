function [comp,P,ssqres,expvar,scaling,tuckcongr] = nwaydecomp_parafac2(dat,ncomp,specmodes,varargin)

% NWAYDECOMP_PARAFAC2 is a low-level function for nd_nwaydecomposition and is used
% to perform nway-decomposition using the PARAFAC2 model. 
% The PRAFAC2 model is modified such that (specified) parameter vectors can be real-valued 
% with complex-valued input. This is described for the PARAFAC model in the publication
% below (please cite when used):
%
%    van der Meij, R., Kahana, M. J., and Maris, E. (2012). Phase-amplitude coupling in 
%        human ECoG is spatially distributed and phase diverse. Journal of Neuroscience, 
%        32(1), 111-123.
%
% If the input array contains more than 3 dimensions, then the original PARAFAC2 model
% is expanded such that the extra dimensions are considered as 'outer modes' of the
% cross-product (see below). The 'outer mode'-dimensions are considered as regular modes 
% for computing their loading vectors, but are concatenated during estimation of matrix P.
% The other option is to consider the extra dimensions as 'estimating modes', which means
% matrix P is computed for each combination of elements of each of the 'estimating modes'.
% This option is not implemented, but equally possible. We use this approach in 
% both SPACE models (see nwaydecomp_spacefsp/spacetime). 
%
%
% Use as
%   [comp,ssqres,expvar,scaling,tuckcongr] = nwaydecomp_parafac2(dat,ncomp,...)
%
% Input:
%   dat         = array containing data to be decomposed, NaNs must be used to fill the array in the incomplete mode (only minimal filling)
%   ncomp       = number indicating number of components
%   specmodes   = vector with length equal to ndims(dat) with 1, 2 and 3
%                 indicating special modes: 1: outer mode of inner-product                     (i.e. the utility modes)   (nmode-2 modes must be this)
%                                           2: inner mode of inner-product                     (i.e. the incomplete mode) (only one allowed)
%                                           3: mode over which inner-products will be computed (i.e. the estimating mode) (only one allowed)
%
% Output:
%   comp       = cell-array containing component loadings per mode (all loading vectors have frobenius norm = 1)
%   P          = cell-array containing P for every incomplete mode
%   ssqres     = sums of squares of the residuals
%   expvar     = percentage explained variance of the complete model
%   scaling    = scaling coefficients belonging to the first mode (magnitude scaling) and the first complex mode (phase shift)
%                all component loading vectors have norm = 1 and, if complex, have magnitude weighted mean phase of 0 in the output
%                the scaling coefficients that have been removed are put in scaling, which is a 1x1 cell-arry containing a ncomp*1 vector
%                if no complex modes are present and a 1x2 cell-array containing an extra ncomp*1 vector if they are
%   tuckcongr  = tuckers congruence coefficents between components, high values mean some components are highly correlated, which is a sign of
%                a degenerate model
%
%
%
% Additional options should be specified in key-value pairs and can be
%   'compmodes'   = vector with length equal to ndims(dat) with 0 or 1
%                   indicating whether component parameters should be complex or not (note, the incomplete mode cannot be real-valued for complex input)
%   'nitt'        = maximum number of iterations (default = 2500)
%   'convcrit'    = convergence criterion (default = 1e-6)
%   'startval'    = previously computed start-values
%   'dispprefix'  = prefix added to all disp-calls, handy when function is used in many loops after each other
%
%
%
% This function is inspired by the N-way toolbox (http://www.models.kvl.dk/source/nwaytoolbox)
% and the Triple SPICE project (http://www.ece.umn.edu/users/nikos/public_html/3SPICE/code.html)
% A fantastic resource on PARAFAC/2 is written by Rasmus Bro http://www.models.kvl.dk/users/rasmus/ 
% (Multi-way Analysis book or monograph)
%
%
%
%   TO DO: core-consistency check (need Tucker3 for that)
%   TO DO: revamp fit computation (QRs...)
%   TO DO: implement linear search as in the SPACE models
%   TO DO: merge some of the subfunctions of the PARAFAC models into externals
%   TO DO: merge PARAFAC2 and PARAFAC2CP
%
%
%
%


%
% Copyright (C) 2012-2014, Roemer van der Meij, roemervandermeij AT gmail DOT com
%
% This file is part of Nwaydecomp, see https://github.com/roemervandermeij/nwaydecomp
%
%    Nwaydecomp is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    Nwaydecomp is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with Nwaydecomp. If not, see <http://www.gnu.org/licenses/>.




% Get the optional input arguments
keyvalcheck(varargin, 'optional', {'compmodes','specmodes','nitt','convcrit','startval','dispprefix'});
compmodes   = keyval('compmodes', varargin);  
nitt        = keyval('nitt', varargin);        if isempty(nitt),         nitt         = 2500;                  end
convcrit    = keyval('convcrit', varargin);    if isempty(convcrit),     convcrit     = 1e-6;                  end
startval    = keyval('startval', varargin);
dispprefix  = keyval('dispprefix', varargin);  if isempty(dispprefix),   dispprefix   = [];                    end

% load data from disk if string is given
if ischar(dat)
  filevars = whos('-file', dat);
  if numel(filevars)>1 || ~any(strcmp(filevars.class,{'single','double'}))
    error('filename contains either a single variable or not an array of singles or doubles')
  end
  datvarname = filevars.name;
  filecontent = load(dat);
  dat = filecontent.(datvarname);
  clear filecontent
end

% set compflg and nmode
compflg = ~isreal(dat);
nmode = ndims(dat);
modeindex = 1:nmode;
if isempty(compmodes), 
  compmodes = ones(1,ndims(dat)) .* compflg;
end

% throw an error of specmodes is not proper
if (sum(specmodes==3)~=1) || (sum(specmodes==2)~=1) || (sum(specmodes==1)~=(ndims(dat)-2)) || isempty(specmodes) || (numel(specmodes)~=ndims(dat)) || (sum(specmodes)~=(5+(ndims(dat)-2)))
  error('improper specmodes')
end

% throw errors for real/complex input with complex/real component matrices
if ~compflg && sum(compmodes) ~= 0
  error('complex input needed for some component matrices to be complex')
elseif compflg && sum(compmodes) == 0
  error('real-valued input needed for all component matrices to be real-valued')
end

% throw error when compmodes is not the same size as data
if nmode ~= length(compmodes)
  error('length of compmodes needs to be equal to ndims(dat)')
end

% throw an error for compmodes being improper
if compflg && compmodes(specmodes==2)==0
  error('incomplete mode cannot be real-valued for complex-valued input')
end

% ensure working with double precision
if ~strcmp(class(dat),'double')
  dat = double(dat);
end

% reorder dimensions so incomplmode is always end-1 and estimmode is always end
% this makes data-handling much easier, and is trivial
% (the only lines of code that are sensitive to this are the lines that permute data))
permordorg = [find(specmodes==1) find(specmodes==2) find(specmodes==3)];
dat        = permute(dat,permordorg);
specmodes  = specmodes(permordorg);
compmodes  = compmodes(permordorg);
if ~isempty(startval)
  startval = startval(permordorg);
end

% set variables related to special modes
utilitymodes = find(specmodes==1);
incomplmode  = find(specmodes==2);
estimmode    = find(specmodes==3);

% set general size of modes, determine sizes of incomplete mode and set smodey
smode = size(dat);
smodeincompl = NaN(1,smode(estimmode));
permorder    = [incomplmode estimmode utilitymodes];
incompldat   = permute(dat,permorder);
incompldat   = incompldat(:,:); % select only the incomplmode and estimmode
for iestimval = 1:smode(estimmode)
  smodeincompl(iestimval) = sum(~isnan(incompldat(:,iestimval)));
end
smodey = smode;
smodey(incomplmode) = ncomp;

% throw errors for improper NaNs
if max(smodeincompl)~=smode(incomplmode)
  error('incomplete mode may only be filled with NaNs to the maximum size of the mode, none more')
end
for imode = [utilitymodes estimmode]
  permorder   = [imode setdiff(modeindex,imode)];
  reshapesize = [smode(imode) prod(smode(setdiff(modeindex,imode)))];
  tmpdat      = reshape(permute(dat,permorder),reshapesize);
  tmpdat      = squeeze(tmpdat(:,1));
  if any(isnan(tmpdat))
    error('only the incomplete mode can contain NaNs')
  end
end

% display input data
dimstring = num2str(smode(1));
for imode = 2:nmode
  dimstring = [dimstring 'x' num2str(smode(imode))];
end
disp([dispprefix 'data is complex array with dimensions ' dimstring])
disp([dispprefix 'a PARAFAC2-model with ' num2str(ncomp) ' components will be estimated '])
disp([dispprefix 'maximum number of iterations = ' num2str(nitt)])
disp([dispprefix 'convergence criterion = ' num2str(convcrit)])

% concat real and imag parts of unfolded dats for real-valued minimization of real-valued modes
disp([dispprefix 'component loading matrices of ' num2str(sum(compmodes)) ' modes will be complex' ])
% has to be done inside the algorithm

% unfold dat for computation of P before ALS
if min(smodeincompl)>prod(smode(utilitymodes))
  choleskflg = 1;
  smodeincomplorg = smodeincompl;
else
  choleskflg = 0;
end
datorig = dat;
% create datforq
datforq = cell(1,smode(estimmode));
for iestimval = 1:smode(estimmode)
  % first, permute and unfold so estimmode is at the front, and can be indexed
  permorder   = [estimmode setdiff(modeindex,estimmode)];
  reshapesize = [smode(estimmode) prod(smode(setdiff(modeindex,estimmode)))];
  tmpdat      = reshape(permute(datorig,permorder),reshapesize);
  % select part of dat for the current estimval, and reshape again to full array
  tmpdat      = tmpdat(iestimval,:);
  reshapesize = smode(setdiff(modeindex,estimmode));
  tmpdat      = reshape(tmpdat,reshapesize);
  % now unfold data from the perspective of incomplmode
  permorder   = [incomplmode setdiff(find(specmodes~=3),incomplmode)];
  reshapesize = [smode(incomplmode) prod(smode(setdiff(find(specmodes~=3),incomplmode)))];
  tmpdat      = reshape(permute(tmpdat,permorder),reshapesize);
  % remove NaNs using smodeincompl
  tmpdat   = tmpdat(1:smodeincompl(iestimval),:); % NaNs should be the same over all utility modes
  
  % compute cholesky decomposition if incomplete mode is bigger than utility modes
  if choleskflg
    if ~(rank(tmpdat'*tmpdat) < size(tmpdat'*tmpdat,1))
      tmpdat = chol(tmpdat'*tmpdat);
    else
      choleskflg = 0;
    end
  end
  
  % save in datforq
  datforq{iestimval} = tmpdat.'; % data-handling related transpose
end
if choleskflg
  disp([dispprefix 'incomplete mode bigger than combined utility modes and outer product is of full rank: using Cholesky decomposition to speed up computation'])
end

% create datmain
reshapesize = [smode(1) prod(smode(setdiff(modeindex,1)))];
datmain     = reshape(datorig,reshapesize); % to be used in computing error term, expvar, etc

% produce random start values from -1 to 1, if no start-values were provided
if isempty(startval)
  comp = cell(1,nmode);
  % do it for all but incomplmode
  for imode = [utilitymodes estimmode]
    compsize = [smode(imode) ncomp];
    if compmodes(imode)==1
      comp{imode} = complex((rand(compsize)*2)-1,(rand(compsize)*2)-1);
    elseif compmodes(imode)==0
      comp{imode} = rand(compsize);
    end
  end
  % do it for the incomplmode
  if compmodes(incomplmode)==1
    comp{incomplmode} = complex((rand(ncomp,ncomp)*2)-1,(rand(ncomp,ncomp)*2)-1);
  elseif compmodes(incomplmode)==0
    comp{incomplmode} = rand(ncomp,ncomp);
  end
else
  comp = startval;
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%   PARAFAC2 ALS  START  %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% The below code calculates an ALS-based estimate of the PARAFAC2 model based on the following structure:
% ((x) = Khatri-rao-bro product)
%
% Model (4-way) (specmodes = [1 2 3 1]):
% X = A*(D(x)C(x)(Pk*Bk))' + E
% Where K is index into C, Bk has size ncomp*ncomp, and Pk has size J*ncomp, where J
% is the k-specific size of second array dimension ('B')
%
% First, calculate Pk:
% Qk = Xk'* A*(D(x)Bk)' (where Xk is the k-specific unfolding of dat, being J*IL)
% Pk = Qk*(Qk'*Qk)^-.5
%
% Then, compute Yk
% Yk = Xk*Pk ((where Xk is transpose of the k-specific unfolding of dat, being IL*J)
%
% Afterwards, update matrices A, B, C and D using a regular PARAFAC-step on Y
%
% Estimate A by minimizing X - AZ'  (frobenius norm squared, x = unfolded)
% where Z = D(x)C(x)B
% ---> A = Y*Z * inv(Z'Z)
%
%
%
%
%
%


% Set some  important variables
ssqdat     = nansum(nansum(abs(datmain).^2));
ssqres     = ssqdat;
prevssqres = 2 * ssqres;
itt        = 0;


% start main while loop of algorithm (updating component matrices)
disp([dispprefix 'starting ALS algorithm using QR-based fit estimation']) % FIXME
while (abs((ssqres - prevssqres) / prevssqres) > convcrit) && (itt < nitt) % needs eps failsafe
  
  
  % Count itt
  itt = itt + 1;
  
  % Set previous stuff
  prevssqres  = ssqres;
  if itt>1
    prevP       = P;
    prevprevcomp = prevcomp;
  end
  prevcomp    = comp;
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Calculate P by using Q, and assemblye Y based on Xk*Pk
  % calculate Pk by Qk
  P = cell(1,smode(estimmode));
  Y = cell(1,smode(estimmode));
  for iestimval = 1:smode(estimmode)
    % prepare dat
    currdatforq = datforq{iestimval};
    
    % calculate partial model for q
    compforq    = comp;
    compforq{estimmode} = compforq{estimmode}(iestimval,:);
    modelforq   = calcmodel_parafac(compforq); % this can be done using regular parafac model formulation
    % reshape modelforq into array
    reshapesize = smodey(specmodes~=3);
    modelforq   = reshape(modelforq,reshapesize);
    % permute so incomplete dimension is first, and unfold
    permorder   = [incomplmode setdiff(find(specmodes~=3),incomplmode)];
    reshapesize = [ncomp prod(smode(setdiff(find(specmodes~=3),incomplmode)))];
    modelforq   = reshape(permute(modelforq,permorder),reshapesize);
    % transpose (regular transpose, as this is data-handling related)
    modelforq   = modelforq.';
    
    % calculate Qk
    Qk = currdatforq' * modelforq;
    % calculate Pk
    Pk = Qk*psqrt(Qk'*Qk);
    P{iestimval} = Pk;
    
    % calculate Yk
    Yk = datforq{iestimval} * Pk;
    
    % transpose, permute and reshape Yk and save it
    Yk           = Yk.'; % incomplmode is now first dimension (data-handling related transpose)
    reshapesize  = [ncomp smode(utilitymodes)];
    Yk           = reshape(Yk,reshapesize);
    permorder    = [incomplmode setdiff(find(specmodes~=3),incomplmode)];
    Yk           = ipermute(Yk,permorder);
    reshapesize  = [smodey(find(specmodes~=3,1)) prod(smodey(setdiff(find(specmodes~=3),find(specmodes~=3,1))))];
    Yk           = reshape(Yk,reshapesize);
    Y{iestimval} = Yk;
  end
  % assemble Y
  Y           = horzcat(Y{:});
  reshapesize = [smodey(specmodes~=3) smode(estimmode)];
  Y           = reshape(Y,reshapesize);
  permorder   = [setdiff(modeindex,estimmode) estimmode];
  Y           = ipermute(Y,permorder);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % prepare for a regular PARAFAC step on Y
  % unfold Ydat's for computation during ALS
  Ydat     = cell(1,nmode);
  Ysmode   = smode;
  Ysmode(incomplmode) = ncomp;
  for imode = 1:nmode
    permorder   = [imode setdiff(modeindex,imode)];
    reshapesize = [Ysmode(imode) prod(Ysmode(setdiff(modeindex,imode)))];
    Ydat{imode}  = reshape(permute(Y,permorder),reshapesize);
  end
  
  % concat real and imag parts of Ydat for real-valued minimization for real-valued modes
  if compflg % only cat when input was complex
    for imode = 1:nmode
      if compmodes(imode)==0
        Ydat{imode} = cat(2,real(Ydat{imode}),imag(Ydat{imode}));
      end
    end
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  
  
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Update all component matrices per mode using a regular PARAFAC step
  for imode = 1:nmode
    currmode = imode; % current mode
    remmodes = setdiff(1:nmode,currmode); % remaining modes
    
     % Calculate Z by a series of nested khatri-rao-bro products (using kr)
      Z = kr(comp(remmodes(end:-1:1)));
      
      
      % Old code, kept here for future use 
      %       % Calculate Z by building tempZ over nested khatri-rao-bro products
      %       % Z is calculated by a nested series of khatri-rao-bro products
      %       % E.g. the Z for the first mode with 4 modes: Z = krbcomp(comp{4},krbcomp(comp{3},comp{2}));
      %       nremmodes = length(remmodes); % number of remaining modes
      %       nkrbcomps = nremmodes-1; % number of khatri-rao-bro products to calculate
      %       tempZ     = comp{remmodes(1)}; % second element of first krbcomp is always first remaining mode
      %       for ikrbcomp = 1:nkrbcomps % loop over nested calculations of khatri-rao-bro productsda
      %         tempZ = krbcomp(comp{remmodes(ikrbcomp+1)},tempZ); 
      %       end
      %       Z = tempZ;
      %       % Update the component matrix for the current mode
      %       comp{imode} = dat{imode} * conj(Z) * inv(Z'*Z).'; % .' is equal to conj in this case
	
    % concatenate Z if it shouldn't be complex (and it is)
    if compflg % only cat when input was complex
      if compmodes(currmode)==0 && ~isreal(Z)
        Z = cat(1,real(Z),imag(Z));
      end
    end
    
    % Update the component matrix for the current mode
    comp{currmode} = Ydat{currmode} * conj(Z) * inv(Z'*Z).';
    %comp{currmode} = Ydat{currmode} * conj(Z) / conj(Z'*Z);
  end % end of looping over modes
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % normalize components, for scaling, phase and permutation indeterminancy
  comp = normalizecomp(comp,compmodes,0,dispprefix);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  
    
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   % Compute fit using a QR shortcut
%   % compute QR decomp of all modes but incompl
%   rcomp = cell(1,nmode);
%   % perform QR-decomposition of all component-matrices
%   for imode = find(specmodes~=2)
%     [dum rcomp{imode} ] = qr(comp{imode},0);
%   end
%   % do the same for B multiplied with conj(P)
%   for iestimval = 1:smode(estimmode)
%     [dum rcomp{incomplmode}{iestimval}] = qr(conj(P{iestimval}) * comp{incomplmode},0);
%   end
%   model     = calcmodel_parafac2qr(rcomp);
  model     = calcmodel_parafac2(comp,P,utilitymodes,estimmode,smode,smodeincompl); % subfunction for calculating model
  ssqmodel  = nansum(nansum(abs(model).^2));
  ssqres    = ssqdat - ssqmodel;
  expvar    = 100 - ((ssqres / ssqdat) * 100);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  
  % Display results of current iteration
  disp([dispprefix 'iteration ' num2str(itt) ' - expvar: ' num2str(expvar,'%-2.1f')   '%  ssqres: ' num2str(ssqres)  '  ssqmodel: ' num2str(ssqmodel)])
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Algorithm stops
  % calculate final explained variance, not based on QR-decomposition
  if ~(abs((ssqres - prevssqres) / prevssqres) > convcrit) || (itt == nitt)
    % compute real P if cholesky decomposition was used
    if choleskflg
      P = cell(1,smode(estimmode));
      for iestimval = 1:smode(estimmode)
        % prepare dat
        % first, permute and unfold so estimmode is at the front, and can be indexed
        permorder   = [estimmode setdiff(modeindex,estimmode)];
        reshapesize = [smode(estimmode) prod(smode(setdiff(modeindex,estimmode)))];
        currdatforq = reshape(permute(datorig,permorder),reshapesize);
        % select part of dat for the current estimval, and reshape again to full array
        currdatforq = currdatforq(iestimval,:);
        reshapesize = smode(setdiff(modeindex,estimmode));
        currdatforq = reshape(currdatforq,reshapesize);
        % now unfold data from the perspective of incomplmode
        permorder   = [incomplmode setdiff(find(specmodes~=3),incomplmode)];
        reshapesize = [smode(incomplmode) prod(smode(setdiff(find(specmodes~=3),incomplmode)))];
        currdatforq = reshape(permute(currdatforq,permorder),reshapesize);
        % remove NaNs using smodeincompl and regular transpose (data-handling related)
        currdatforq = currdatforq(1:smodeincomplorg(iestimval),:); % NaNs should be the same over all utility modes
        currdatforq = currdatforq.';
        
        % calculate partial model for q
        compforq    = comp;
        compforq{estimmode} = compforq{estimmode}(iestimval,:);
        modelforq   = calcmodel_parafac(compforq); % this can be done using regular parafac model formulation
        % reshape modelforq into array
        reshapesize = smodey(specmodes~=3);
        modelforq   = reshape(modelforq,reshapesize);
        % permute so incomplete dimension is first, and unfold
        permorder   = [incomplmode setdiff(find(specmodes~=3),incomplmode)];
        reshapesize = [ncomp prod(smode(setdiff(find(specmodes~=3),incomplmode)))];
        modelforq   = reshape(permute(modelforq,permorder),reshapesize);
        % transpose (regular transpose, as this is data-handling related)
        modelforq   = modelforq.';
        
        % calculate Qk
        Qk = currdatforq' * modelforq;
        % calculate Pk
        Pk = Qk*psqrt(Qk'*Qk);
        P{iestimval} = Pk;
      end
    end
    model     = calcmodel_parafac2(comp,P,utilitymodes,estimmode,smode,smodeincompl); % subfunction for calculating model
    ssqres    = nansum(nansum(abs(datmain - model).^2));
%     qrexpvar  = expvar;
    expvar    = 100 - ((ssqres / ssqdat) * 100);
%     if abs(qrexpvar - expvar) > 1
%       error(['problem with QR-based acceleration: expvar = ' num2str(expvar,'%-2.4f') '%  QR-based expvar = ' num2str(qrexpvar,'%-2.4f') '%'])
%     end
  end
  if ~(abs((ssqres - prevssqres) / prevssqres) > convcrit)
    disp([dispprefix 'convergence criterion of ' num2str(convcrit) ' reached in ' num2str(itt) ' iterations'])
    disp([dispprefix 'explained variance by model: ' num2str(expvar,'%-2.1f') '%'])
  elseif (itt == nitt)
    disp([dispprefix 'maximum number of iterations = ' num2str(itt) ' reached'])
    disp([dispprefix 'explained variance by model: ' num2str(expvar,'%-2.1f') '%'])
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  
end % end main while loop of algorithm (updating component matrices)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%   PARAFAC2 ALS   END    %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%   POSTPROCESSING START  %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate Core Consistency Diagnostic if desired
if nargout > 6
  error('experimental')
%   % this is an approximation of the t3core for the PARAFAC model, but made for every K, and then averaged
%   % loop over the estimmode
%   t3core = NaN([smode(estimmode) prod(ones(1,nmode)*ncomp)]);
%   for iestimval = 1:smode(estimmode)
%     
%     % first, caculate the kronecker products of all loading matrices
%     % calculate kroncomp by building kroncomp over nested kronecker products
%     % E.g. t with 4 modes: kronprod = kron(D,kron(C,kron(B,A)))
%     % perform steps based on the number of modes
%     if nmode>4
%       tempkron = comp{1};
%       % krbcomps over utility modes
%       for inkroncomp = 1:nmodes-3
%         tempkron = kron(comp{inkroncomp+1},tempkron);
%       end
%       % first-to-last kronecker product
%       tempkron = kron(conj(P{iestimval})*comp{end-1},tempkron);
%       % last kronecker product
%       kroncomp = kron(comp{end}(iestimval,:),tempkron);
%     elseif nmode==4
%       % first kronecker product
%       tempkron = kron(comp{2},comp{1});
%       % first-to-last kronecker product
%       tempkron = kron(conj(P{iestimval})*comp{3},tempkron);
%       % last kronecker product
%       kroncomp = kron(comp{4}(iestimval,:),tempkron);
%     elseif nmode==3
%       % first-to-last kronecker product
%       tempkron = kron(conj(P{iestimval})*comp{2},comp{1});
%       % last kronecker product
%       kroncomp = kron(comp{3}(iestimval,:),tempkron);
%     end
%     
%     % vec the data
%     datvec = datforq{iestimval}(:);
%     % calculate tucker3 core array
%     % if complex, make it real by concatenating the real and imagenary parts
%     if compflg
%       t3coretmp = cat(1,real(kroncomp),imag(kroncomp)) \ cat(1,real(datvec),imag(datvec));
%     else
%       t3coretmp = kroncomp \ datvec;
%     end
%     t3core(iestimval,:) = t3coretmp;
%   end % estimval
%   % average over estimmode and reshape
%   t3core = squeeze(mean(t3core,1));
%   t3core = reshape(t3core,ones(1,nmode)*ncomp);
end


% reverse permute performed at the start (also for all the original variables that will not be used, for bookkeeping and testing)
invpermordorg(permordorg) = 1:nmode;
dat       = permute(dat,invpermordorg);
specmodes = specmodes(invpermordorg);
compmodes = compmodes(invpermordorg);
comp      = comp(invpermordorg);



% normalize components, for scaling, phase and permutation indeterminancy
comp = normalizecomp(comp,compmodes,1,dispprefix);

% Compute ssq per component and sort
ssqcomp = diag(comp{1}'*comp{1});
[ssqcomp,sortorder] = sort(ssqcomp,'descend');
for imode = 1:nmode % sort per mode
  comp{imode} = comp{imode}(:,sortorder);
end
disp([dispprefix 'components have been ordered according to magnitude'])


% Continue postprocessing
% set norm of mode 1 per loading vector to 1 and save scaling coeffient
for icomp = 1:ncomp
  % set mode1
  mode1     = comp{1}(:,icomp);
  mode1norm = norm(mode1,'fro');
  mode1     = mode1 ./ mode1norm;
  comp{1}(:,icomp)  = mode1;
  scaling{1}(icomp) = mode1norm;
end
disp([dispprefix 'first mode magnitude scaling coefficient removed'])
% set average magnitude weighted phase to 0 for the first complex loading vector per component and save scaling coefficient
compmodeindex = find(compmodes==1);
if sum(compmodes)~=0 % only do this if there is a complex mode
  for icomp = 1:ncomp
    % set mode1
    mode1      = comp{compmodeindex(1)}(:,icomp); % compmodeindex determined above
    meanangle  = angle(mean(mode1)); % angle of the mean weighs by magnitude
    phaseshift = exp(-1i*meanangle);
    mode1      = mode1 .* phaseshift;
    comp{compmodeindex(1)}(:,icomp) = mode1;
    scaling{2}(icomp) = phaseshift;
  end
  disp([dispprefix 'first complex mode has been phase shifted so average magnitude-weighted-phase = 0'])
end
disp([dispprefix 'post-processing finished'])


% Calculate Tucker's Congruency coefficient
tuckcongr = ones(ncomp,ncomp);
for imode = 1:nmode
  innprod = comp{imode}' * comp{imode};
  tuckcongr = tuckcongr .* innprod;
end
% remove upper triangle and diagonal, make vector, and take abs
tuckcongr = tril(tuckcongr,-1);
tuckcongr(tuckcongr==0) = [];
tuckcongr = abs(tuckcongr);
if ncomp==1 % if ncomp is one, tuckers congruence cannot be calculated
  tuckcongr = NaN;
end
% display warning if too high
if max(tuckcongr) >= 0.85
  disp([dispprefix 'Warning: some components are highly correlated, model might be degenerate'])
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%   POSTPROCESSING END    %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%













%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%      SUBFUNCTIONS START HERE     %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%  Subfunction normalizecomp   %%%%%%%%%%%%
function comp = normalizecomp(comp,compmodes,dispflg,dispprefix)
% set stuff for subfunction internal use
ncomp = size(comp{1},2);
nmode = numel(comp);

% Magnitude postprocessing of components (make frobenius norm per loading vector norm = 1)
% make sure all explained variance is contained in first mode
for icomp = 1:ncomp
  % set mode1, which get's all the variance
  mode1 = comp{1}(:,icomp);
  % loop over the other modes
  for imode = 2:nmode
    currmode     = comp{imode}(:,icomp); % current mode of current compoment
    currmodenorm = norm(currmode,'fro');
    mode1        = mode1    .* currmodenorm;
    currmode     = currmode ./ currmodenorm;
    comp{imode}(:,icomp) = currmode;
  end
  % set mode1 back into original format
  comp{1}(:,icomp) = mode1;
end
if dispflg
  disp([dispprefix 'components have been magnitude normalized in all but the first mode'])
end

% Phase postprocessing of complex components (make average magnitude weigthed phase per complex loading vector 0)
% make sure the scaling of phases is contained in first complex mode
compmodeindex = find(compmodes==1);
if sum(compmodes)>=2 % only do this if there is more than one complex mode
  for icomp = 1:ncomp
    % set mode1 (the first complex mode), which get's the phase scaling
    mode1 = comp{compmodeindex(1)}(:,icomp);
    % loop over the other modes
    for imode = compmodeindex(2:end)
      currmode   = comp{imode}(:,icomp); % current mode of current compoment
      meanangle  = angle(mean(currmode)); % angle of the mean weighs by magnitude
      phaseshift = exp(-1i*meanangle);
      mode1      = mode1    ./ phaseshift;
      currmode   = currmode .* phaseshift;
      comp{imode}(:,icomp) = currmode;
    end
    % set mode1 back into original format
    comp{compmodeindex(1)}(:,icomp) = mode1;
  end
  if dispflg
    disp([dispprefix 'components have been phase shifted so average magnitude-weighted-phase = 0 with respect to the first complex mode'])
  end
end


% Apply sign convention, all real-valued modes are set to be 'mainly' positive, on a per component basis
% If a loading vector in a mode has a negative max(abs(value)) then the entire loading vector is scaled by -1,
% the respective loading vector in the first mode is then also scaled by -1 (inv(-1) to be precise)
% set mode1 which get's all the scaling
mode1         = comp{1};
signsmode1    = ones(1,ncomp);
% loop over the other modes
for imode = 2:nmode
  if isreal(comp{imode}) % don't scale the complex modes, it's irrelevant
    currmode      = comp{imode};
    signscurrmode = sign(sum(currmode,1));
    signsmode1    = signsmode1 .* signscurrmode; % update signs to be used at mode1
    comp{imode}   = currmode * diag(signscurrmode);
  end
end
% Apply all scaling to mode1
comp{1} = mode1 * diag(signsmode1);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%  Subfunction calcmodel_parafac   %%%%%%%%%%%%
function [model] = calcmodel_parafac(comp)

% Calculate model by building it over a series of nested khatri-rao-products
% E.g. the model for 4 modes: model = comp{1} * krbcomp(comp{4},krbcomp(comp{3},comp{2})).';

model = comp{1} * kr(comp(end:-1:2)).';

% Old code, kept here for future use
% % set several variables
% nmode    = length(comp); % number of modes
% nkrbcomps = nmode-2; % number of khatri-rao-bro products to calculate
% tempmodel = comp{2}; % second element of first krbcomp is always second mode
% 
% % start series of khatri-rao-bro products
% for ikrbcomp = 1:nkrbcomps %
%   tempmodel = krbcomp(comp{ikrbcomp+2},tempmodel); %
% end
% 
% % last step in calculating model
% model = comp{1} * tempmodel.';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%  Subfunction calcmodel_parafac2   %%%%%%%%%%%%
function [model] = calcmodel_parafac2(comp,P,utilitymodes,estimmode,smode,smodeincompl)

% Calculate model by building it over a series of nested khatri-rao-products
% E.g. the model for 4 modes: model = comp{1} * krbcomp(comp{4},krbcomp(comp{3},comp{2})).';
% In the incomplmode, calculate Pk * comp{i} prior to kbr-products, by looping over estimmmode

% set stuff
nmodes    = numel(comp); % number of modes

% NOTE: below krbcomp has been replaced by kr, which SHOULD be fine... (error will be present if not), can be optimized without loop
% start series of khatri-rao-bro products, looping over estimmode
modelk = cell(1,smode(estimmode));
for iestimval = 1:smode(estimmode)
  
  % perform steps based on the number of modes
  if nmodes>4 % if nmodes is 3 then the last krbcomp is the only one
    tempmodel = comp{2}; % second element of first krbcomp is always second mode
    % krbcomps over utility modes
    for ikrbcomp = 1:nmodes-4
      tempmodel = kr(comp{ikrbcomp+2},tempmodel);
    end
    % first-to-last khatri-rao-bro product
    tempmodel = kr(conj(P{iestimval})*comp{end-1},tempmodel);
    
    % last khatri-rao-bro product
    tempmodel = kr(comp{end}(iestimval,:),tempmodel);
    
  elseif nmodes==4
    % first-to-last khatri-rao-bro product
    tempmodel = kr(conj(P{iestimval})*comp{3},comp{2});
    
    % last khatri-rao-bro product
    tempmodel = kr(comp{4}(iestimval,:),tempmodel);
    
  elseif nmodes==3
    % last (and only) khatri-rao-bro product
    tempmodel = kr(comp{3}(iestimval,:),conj(P{iestimval})*comp{2});
  end
  
  
  % last step in calculating model
  model = comp{1} * tempmodel.';
  
  % add NaNs by adding them to permuted last dimension
  % reshape so that incomplmode is at the beginning, so NaNs van be added
  reshapesize = [smode(1) smode(setdiff(utilitymodes,1)) smodeincompl(iestimval)];
  model       = reshape(model,reshapesize);
  permorder   = [nmodes-1 1:(nmodes-2)]; % last mode right now is always incomplmode
  reshapesize = [smodeincompl(iestimval) prod(smode(utilitymodes))];
  model       = reshape(permute(model,permorder),reshapesize);
  % add NaNs
  model       = cat(1,model, nan(max(smodeincompl)-smodeincompl(iestimval),size(model,2)));
  reshapesize = [max(smodeincompl) smode(utilitymodes)];
  model       = reshape(model,reshapesize);
  permorder   = [nmodes-1 1:(nmodes-2)];
  model       = ipermute(model,permorder);
  reshapesize = [smode(1) prod([smode(setdiff(utilitymodes,1)) max(smodeincompl)])];
  model       = reshape(model,reshapesize);
  
  % save modelk
  modelk{iestimval} = model;
end

% assemble model
model       = horzcat(modelk{:});



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%  Subfunction calcmodel_parafac2qr   %%%%%%%%%
function [model] = calcmodel_parafac2qr(comp)

% Calculate model by building it over a series of nested khatri-rao-products

% set stuff
nmodes    = numel(comp); % number of modes
sestim    = size(comp{end},1);

% NOTE: below krbcomp has been replaced by kr, which SHOULD be fine... (error will be present if not), can be optimized without loop
% start series of khatri-rao-bro products, looping over estimmode
modelk = cell(1,sestim);
for iestimval = 1:sestim
  
  % perform steps based on the number of modes
  if nmodes>4 % if nmodes is 3 then the last krbcomp is the only one
    tempmodel = comp{2}; % second element of first krbcomp is always second mode
    % krbcomps over utility modes
    for ikrbcomp = 1:nmodes-4
      tempmodel = kr(comp{ikrbcomp+2},tempmodel);
    end
    % first-to-last khatri-rao-bro product
    tempmodel = kr(comp{end-1}{iestimval},tempmodel);
    
    % last khatri-rao-bro product
    tempmodel = kr(comp{end}(iestimval,:),tempmodel);
    
  elseif nmodes==4
    % first-to-last khatri-rao-bro product
    tempmodel = kr(comp{3}{iestimval},comp{2});
    
    % last khatri-rao-bro product
    tempmodel = kr(comp{4}(iestimval,:),tempmodel);
    
  elseif nmodes==3
    % last (and only) khatri-rao-bro product
    tempmodel = kr(comp{3}(iestimval,:),comp{2}{iestimval});
  end
  
  
  % last step in calculating model
  model = comp{1} * tempmodel.';
 
  % save modelk
  modelk{iestimval} = model;
end

% assemble model
model       = horzcat(modelk{:});



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%  Subfunction psqrt   %%%%%%%%%%%%%%%%%%%%%%%%
% calculate a the square root even if rank problems
function X = psqrt(A,tol)

   % Produces A^(-.5) even if rank-problems

   [U,S,V] = svd(A,0);
   if min(size(S)) == 1
     S = S(1);
   else
     S = diag(S);
   end
   if (nargin == 1)
     tol = max(size(A)) * S(1) * eps;
   end
   r = sum(S > tol);
   if (r == 0)
     X = zeros(size(A'));
   else
     S = diag(ones(r,1)./sqrt(S(1:r)));
     X = V(:,1:r)*S*U(:,1:r)';
  end






function sandboxed_testingplayground

%
nans = nan(10,15,10,25);
nans(:,:,:,1) = complex(rand(10,15,10),rand(10,15,10));
dat = cat(3,complex(rand(10,15,10,25),rand(10,15,10,25)), nans);
specmodes = [1 1 2 3];
ncomp = 2;
nwaydecomp_parafac2(dat,ncomp,specmodes,'compmodes',[1 1 1 1])

%
nans = nan(10,15,30);
nans(:,:,1) = complex(rand(10,15),rand(10,15));
dat = cat(2,complex(rand(10,15,30),rand(10,15,30)), nans);
specmodes = [1 2 3];
ncomp = 2;
nwaydecomp_parafac2(dat,ncomp,specmodes,'compmodes',[1 1 1])


%
dat = NaN(10,15,20,25);
for ix = 1:10
  for iy = 1:15
    for iz = 1:20
      for ia = 1:25
        dat(ix,iy,iz,ia) = ix*1e9+iy*1e6+iz*1e3+ia;
      end
    end
  end
end
nwaydecomp_parafac2(dat,2,[1 1 2 3])


%
dat = complex(rand(15,50,25),rand(15,50,25));
specmodes = [1 2 3];
ncomp = 2;
nwaydecomp_parafac2(dat,ncomp,specmodes,'compmodes',[1 1 1])
nwaydecomp_parafac(dat,ncomp,'compmodes',[1 1 1])


%
dat = rand(10,15,20,25);
specmodes = [1 2 3 1];
ncomp = 2;
nwaydecomp_parafac2(dat,ncomp,specmodes)
nwaydecomp_parafac(dat,ncomp)





I = 10;
J = 15;
K = 20;
L = 25;
F = 4;
A = rand(I,F);
B = rand(F,F);
C = rand(K,F);
D = rand(L,F);
Pk = orth(rand(J,F));
Bfull = Pk*B;
P = NaN(J,F,K);
Bfull = NaN(J,F,K);
for i = 1:K
  P(:,:,i) = orth(rand(J,F));
  Bfull(:,:,i) = P(:,:,i)*B;
end





% % 4-way Pk
% X = A*krbcomp(D,krbcomp(C,B)).';
% X = reshape(X,[I,J,K,L]);
% Pk = cell(1,K);
% for i = 1:K
%   Xunfold = squeeze(X(:,:,i,:));
%   Xunfold = reshape(Xunfold,[I,J*L]);
%   Qk      = Xunfold.' * (A*(krbcomp(D,krbcomp(C(i,:),B))).');
%   Pk{i}   = (Qk*(Qk.'*Qk))^-.5;
% end



%%%%
% 4-way estimation of complex Pk and PARAFAC ALS step
clear
clc
I = 10;
J = 15;
K = 20;
L = 25;
F = 3;
A = complex(rand(I,F),rand(I,F));
B = complex(rand(J,F),rand(J,F));
C = complex(rand(F,F),rand(F,F));
D = complex(rand(L,F),rand(L,F));
Pl = orth(complex(rand(K,F),rand(K,F)));
Cfull = Pl*C;
X = A*krbcomp(D,krbcomp(Cfull,B)).';
X = reshape(X,[I,J,K,L]);
Y = reshape(A*krbcomp(D,krbcomp(C,B)).',[I,J,F,L]);
Pl2 = cell(1,L);
for i = 1:L
  Xunfold = squeeze(X(:,:,:,i)); % order is I, J, K
  Xunfold = permute(Xunfold,[3 1 2]); % order should be K, I, J
  Xunfold = reshape(Xunfold,[K I*J]); % unfolded with K after I
  Qkpart  = A*(krbcomp(D(i,:),krbcomp(C,B))).'; % K is now F, so the order is I * JF (and not I * KF)
  Qkpart  = reshape(Qkpart,[I,J,F]); % order is I, J, F
  Qkpart  = permute(Qkpart,[3 1 2]); % order should be F, I, J (so after reshape and transpose the form is IJ*F)
  Qkpart  = reshape(Qkpart,[F I*J]);
  Qkpart  = Qkpart.'; % transpose so dimensions are: IK*F
  Qk      = conj(Xunfold) * Qkpart; % TRANSPOSE IS NOW CONJ HERE BECAUSE OF PERMUTE
  Pl2{i}   = ((Qk*(Qk'*Qk)^-.5));
  %   [U,S,V] = svd(Xunfold' * (A*(krbcomp(C(i,:),B)).'),0);
  %   Pk2{i}  = U*V';
end
% reassemble X into Xfromk
Xfroml = cell(1,L);
for i = 1:L
  Xfroml{i} = A*krbcomp(D(i,:),krbcomp(conj(Pl2{i})*C,B)).';
end
Xfroml = horzcat(Xfroml{:});
Xfroml = reshape(Xfroml,[I,J,K,L]);
Yfroml = cell(1,L);
for i = 1:L
  Xunfold = squeeze(X(:,:,:,i)); % order is I, J, K
  Xunfold = permute(Xunfold,[3 1 2]); % order should be K, I, J
  Xunfold = reshape(Xunfold,[K I*J]); % unfolded with L after I
  Yl      = (Xunfold.' * Pl2{i}).';
  Yl      = reshape(Yl,[F,I,J]);
  Yl      = permute(Yl,[2 3 1]);
  Yl      = reshape(Yl,[I,J*F]);
  Yfroml{i} = Yl;
end
Yfroml = horzcat(Yfroml{:});
%Yfromk = reshape(Yfromk,[I,F,K]);
disp(' ')
disp(['Xdiff = ' num2str(mean(mean(mean(mean(Xfroml-X)))))])
disp(['Ydiff = ' num2str(mean(mean(mean(mean(reshape(Yfroml,[I,J,F,L])-Y)))))])
% one ALS step A,B,C
Aals = Yfroml * pinv(krbcomp(D,krbcomp(C,B))).';
disp(['Adiff = ' num2str(mean(mean(Aals-A)))])
Bals = reshape(permute(reshape(Yfroml,[I,J,F,L]),[2 1 3 4]),[J,I*F*L]) * pinv(krbcomp(D,krbcomp(C,Aals))).';
disp(['Bdiff = ' num2str(mean(mean(Bals-B)))])
Cals = reshape(permute(reshape(Yfroml,[I,J,F,L]),[3 1 2 4]),[F,I*J*L]) * pinv(krbcomp(D,krbcomp(Bals,Aals))).';
disp(['Cdiff = ' num2str(mean(mean(Cals-C)))])
Dals = reshape(permute(reshape(Yfroml,[I,J,F,L]),[4 1 2 3]),[L,I*J*F]) * pinv(krbcomp(Cals,krbcomp(Bals,Aals))).';
disp(['Cdiff = ' num2str(mean(mean(Dals-D)))])





%%%%
% 3-way estimation of complex Pk and PARAFAC ALS step  - BOOM! (successful)
clear
clc
I = 10;
J = 15;
K = 20;
F = 3;
A = complex(rand(I,F),rand(I,F));
B = complex(rand(F,F),rand(F,F));
C = complex(rand(K,F),rand(K,F));
Pk = orth(complex(rand(J,F),rand(J,F)));
Bfull = Pk*B;
X = A*krbcomp(C,Bfull).';
X = reshape(X,[I,J,K]);
Y = reshape(A*krbcomp(C,B).',[I,F,K]);
Pk2 = cell(1,K);
for i = 1:K
  Xunfold = squeeze(X(:,:,i));
  Xunfold = reshape(Xunfold,[I,J]);
  Qk      = Xunfold' * (A*(krbcomp(C(i,:),B)).');
  Pk2{i}  = ((Qk*(Qk'*Qk)^-.5));
  %   [U,S,V] = svd(Xunfold' * (A*(krbcomp(C(i,:),B)).'),0);
  %   Pk2{i}  = U*V';
end
% reassemble X into Xfromk
Xfromk = cell(1,K);
for i = 1:K
  Xfromk{i} = reshape(A*krbcomp(C(i,:),conj(Pk2{i})*B).',[I,J]);
end
Xfromk = horzcat(Xfromk{:});
Xfromk = reshape(Xfromk,[I,J,K]);% assemble Y from Yk
Yfromk = cell(1,K);
for i = 1:K
  Xunfold   = squeeze(X(:,:,i));
  Xunfold   = reshape(Xunfold,[I,J]);
  Yfromk{i} = Xunfold * (Pk2{i});
end
Yfromk = horzcat(Yfromk{:});
%Yfromk = reshape(Yfromk,[I,F,K]);
disp(' ')
disp(['Xdiff = ' num2str(mean(mean(mean(Xfromk-X))))])
disp(['Ydiff = ' num2str(mean(mean(mean(reshape(Yfromk,[I,F,K])-Y))))])
% one ALS step A,B,C
Aals = Yfromk * pinv(krbcomp(C,B)).';
disp(['Adiff = ' num2str(mean(mean(Aals-A)))])
Bals = reshape(permute(reshape(Yfromk,[I,F,K]),[2,1,3]),[F,I*K]) * pinv(krbcomp(C,Aals)).';
disp(['Bdiff = ' num2str(mean(mean(Bals-B)))])
Cals = reshape(permute(reshape(Yfromk,[I,F,K]),[3,1,2]),[K,I*F]) * pinv(krbcomp(Bals,Aals)).';
disp(['Cdiff = ' num2str(mean(mean(Cals-C)))])




%%%%
% 3-way estimation of Pk - BOOM! (successful)
X = A*krbcomp(C,Bfull).';
X = reshape(X,[I,J,K]);
Y = reshape(A*krbcomp(C,B).',[I,F,K]);
Pk = cell(1,K);
for i = 1:K
  Xunfold = squeeze(X(:,:,i));
  Xunfold = reshape(Xunfold,[I,J]);
  Qk      = Xunfold.' * (A*(krbcomp(C(i,:),B)).');
  Pk{i}   = Qk*(Qk'*Qk)^-.5;
end
% reassemble X into Xfromk
Xfromk = cell(1,K);
for i = 1:K
  Xfromk{i} = reshape(A*krbcomp(C(i,:),Pk{i}*B).',[I,J]);
end
Xfromk = horzcat(Xfromk{:});
Xfromk = reshape(Xfromk,[I,J,K]);
% assemble Y from Yk
Yfromk = cell(1,K);
for i = 1:K
  Xunfold   = squeeze(X(:,:,i));
  Xunfold   = reshape(Xunfold,[I,J]);
  Yfromk{i} = Xunfold * Pk{i};
end
Yfromk = horzcat(Yfromk{:});
Yfromk = reshape(Yfromk,[I,F,K]);




%%%%
% 4-way estimation of Pk - (extra mode is utility mode) BOOM! (successful)
X = A*krbcomp(D,krbcomp(C,Bfull)).';
X = reshape(X,[I,J,K,L]);
Y = reshape(A*krbcomp(D,krbcomp(C,B)).',[I,F,K,L]);
Pk = cell(1,K);
for i = 1:K
  Xunfold = squeeze(X(:,:,i,:)); % order is I, J, L
  Xunfold = permute(Xunfold,[2 1 3]); % order should be J, I, L
  Xunfold = reshape(Xunfold,[J I*L]); % unfolded with L after I
  Qkpart  = A*(krbcomp(D,krbcomp(C(i,:),B))).'; % J is now F, so the order is I * FL (and not I * LF)
  Qkpart  = reshape(Qkpart,[I,F,L]); % order is I, F, L
  Qkpart  = permute(Qkpart,[2 1 3]); % order should be F, I, L (so after reshape and transpose the form is IL*F)
  Qkpart  = reshape(Qkpart,[F I*L]);
  Qkpart  = Qkpart.'; % transpose so dimensions are: IL*F
  Qk      = Xunfold * Qkpart; % TRANSPOSE NOT HERE NOW BECAUSE OF PERMUTE
  Pk{i}   = Qk*(Qk'*Qk)^-.5;
end
% reassemble X into Xfromk
Xfromk = cell(1,K);
for i = 1:K
  Xfromk{i} = reshape(A*krbcomp(D,krbcomp(C(i,:),Pk{i}*B)).',[I,J,L]);
end
Xfromk = horzcat(Xfromk{:});
Xfromk = reshape(Xfromk,[I,J,L,K]); % order is I, J, L, K
Xfromk = permute(Xfromk,[1 2 4 3]); % order should be I, J, K, L
% assemble Y from Yk
Yfromk = cell(1,K);
for i = 1:K
  Xunfold   = squeeze(X(:,:,i,:)); % order is I, J, L
  Xunfold   = permute(Xunfold,[2 1 3]); % order should be J, I, L
  Xunfold   = reshape(Xunfold,[J I*L]); % unfolded with L after I
  Xunfold   = Xunfold.'; % transpose, order is IL*J
  Yfromk{i} = (Xunfold * Pk{i}).'; % transpose, order is F*IL
end
Yfromk = horzcat(Yfromk{:}); % order is F*ILK
Yfromk = reshape(Yfromk,[F,I,L,K]);
Yfromk = permute(Yfromk,[2 1 4 3]); % order should be I,F,K,L
% 4-way estimation of Pk - (extra mode is estimating mode)
Pkl = cell(K,L);
for k = 1:K
  for l = 1:L
    Xunfold = squeeze(X(:,:,k,l));
    Xunfold = reshape(Xunfold,[I,J]);
    Qk      = Xunfold.' * (A*(krbcomp(D(l,:),krbcomp(C(k,:),B))).');
    Pkl{k,l}   = Qk*(Qk'*Qk)^-.5;
  end
end
% reassemble X into Xfromk
Xfromk = cell(K,L);
for k = 1:K
  for l = 1:L
    Xfromk{k,l} = reshape(A*krbcomp(D(l,:),krbcomp(C(k,:),Pk{k}*B)).',[I,J]);
  end
end
Xfromk = horzcat(Xfromk{:});
Xfromk = reshape(Xfromk,[I,J,L,K]); % order is I, J, L, K
Xfromk = permute(Xfromk,[1 2 4 3]); % order should be I, J, K, L
% assemble Y from Yk
Yfromk = cell(1,K);
for i = 1:K
  Xunfold   = squeeze(X(:,:,i,:)); % order is I, J, L
  Xunfold   = permute(Xunfold,[2 1 3]); % order should be J, I, L
  Xunfold   = reshape(Xunfold,[J I*L]); % unfolded with L after I
  Xunfold   = Xunfold.'; % transpose, order is IL*J
  Yfromk{i} = (Xunfold * Pk{i}).'; % transpose, order is F*IL
end
Yfromk = horzcat(Yfromk{:}); % order is F*ILK
Yfromk = reshape(Yfromk,[F,I,L,K]);
Yfromk = permute(Yfromk,[2 1 4 3]); % order should be I,F,K,L






% % testing equaly of Bfull and Pk*B
% Xk = cell(1,K);
% for i = 1:K
%   Xk{i} = reshape(A*krbcomp(C(i,:),Pk*B).',[I,J]);
% end
% Xfromk = horzcat(Xk{:});
% Xfromk = reshape(Xfromk,[I,J,K]);
% X = A*krbcomp(C,Bfull).';
% X = reshape(X,[I,J,K]);
%
%
% % testing 3-way reassamble for k, reshape operations and such
% Xk = cell(1,K);
% for i = 1:K
%   Xk{i} = reshape(A*krbcomp(C(i,:),Bfull).',[I,J]);
% end
% Xfromk = horzcat(Xk{:});
% Xfromk = reshape(Xfromk,[I,J,K]);
% X = A*krbcomp(C,Bfull).';
% X = reshape(X,[I,J,K]);



% test multidimensional reshape
test = NaN(2,4,6,8);
for ix = 1:2
  for iy = 1:4
    for iz = 1:6
      for ia = 1:8
        test(ix,iy,iz,ia) = ix*1000+iy*100+iz*10+ia;
      end
    end
  end
end
test2 = reshape(test,[2 4*6*8]);


% test cat over 3rd dimension
test = cell(1,3);
for i = 1:3
  test{i} = [110 120; 210 220] + i;
end
test = cat(3,test{:});





%%% test subfunc calcmodel_parafac2 and other main func stuff
clear
clc
I = 10;
J = 15;
K = 20;
F = 4;
A = complex(rand(I,F),rand(I,F));
B = complex(rand(F,F),rand(F,F));
C = complex(rand(K,F),rand(K,F));
Pk = orth(complex(rand(J,F),rand(J,F)));
Bfull = Pk*B;
Ynf = A*krbcomp(C,B)';
X = A*krbcomp(C,Bfull)';
Xfromk = cell(1,K);
for i = 1:K
  Xfromk{i} = A*krbcomp(C(i,:),Pk*B)';
end
%Xfromk = horzcat(Xfromk{:});
%Xfroml = reshape(Xfroml,[I,J,K,L]); % order is I, J, L, K
Yfromk = cell(1,K);
Pfromk = cell(1,K);
for i = 1:K
  Qk        = Xfromk{i}' * (A*krbcomp(C(i,:),B)');
  Pfromk{i} = Qk*(Qk'*Qk)^-.5;
  Yfromk{i} = Xfromk{i} * Pfromk{i};
end
Yfromk = horzcat(Yfromk{:});
% Xfromk = horzcat(Xfromk{:});
% disp('')
% disp(['Pdiff = ' num2str(mean(mean(Pfromk{1}-Pk)))])
% disp(['Xdiff = ' num2str(mean(mean(mean(Xfromk-X))))])
% disp(['Ydiff = ' num2str(mean(mean(mean(Yfromk-Ynf))))])
% disp('')
%
datforq       = Xfromk;
comp          = {A,B,C};
comporg       = comp;
ncomp         = F;
specmodes     = [1 2 3];
utilitymodes  = [1];
incomplmode   = 2;
estimmode     = 3;
smode         = [I,J,K];
smodey        = [I, F, K];
smodeincompl  = repmat(J,[1 K]);
nmode         = 3;
modeindex     = 1:nmode;
%
disp(['Pfuncdiff = ' num2str(mean(mean(P{1}-Pk)))])
disp(['Yfuncdiff = ' num2str(mean(mean(mean(reshape(Ynf,[I,F,K])-Y))))])
disp(['CompPittdiff = ' num2str(mean(mean(vertcat(comporg{:})-vertcat(comp{:}))))])
disp(['CompPittModelYdiff = ' num2str(mean(mean(Ynf-(comp{1}*krbcomp(comp{3},comp{2})'))))])






figure
compass(datmain(10,1:1000))
hold on
compass(model(10,1:1000),'r')





figure
compass(datmain(10,1:1000)-model(10,1:1000))
hold on
compass(model(10,1:1000),'r')


















