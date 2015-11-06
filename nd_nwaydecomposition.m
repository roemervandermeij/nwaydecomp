function [nwaycomp] = nd_nwaydecomposition(cfg,data)

% ND_NWAYDECOMPOSITION Decomposes an N-way array of electrophysiological data into components.
% 
% Please first read the README_FIRST.rtf, acompanying this toolbox.
%
% A numerical N-way array needs to be present, and indicated in the cfg.
%
% Supported N-way models are: SPACE-FSP, SPACE-time, PARAFAC (modified), PARAFAC2 (modified)
% This function can be used to decompose numerical arrays, which are complex-valued or real-valued, into 
% components. How a component is modeled and what it represents, as well as the required structure of the 
% numerical array, depends on the model. This functions does two important things:
% 1) it takes care of randomly initializing each model
% 2) it can be used to estimate the number of components to extract
%
% Ad1: 
% Each of the supported models needs to randomly initialized multiple times in order to avoid local minima 
% of the loss functions of each model, and to avoid degenerate solutions. Once it converges to the same 
% solution from multiple random starting points, it can be assumed the global minimum is reached.
%
% Ad2:
% The number of components to extract from the numerical array needs to be determined empirically (similar to ICA). 
% This is the case for all of the supported models. This functions can be used for this purpose using 4 different
% strategies:
% 1) split-half of the array along a dimension. This strategy allows for increasing the number of components until a criterion 
%    is no longer reached. This criterion is based on a statistic that assesses the similarity of components between 
%    split halves, which ranges between 0 and 1 (identical). The two split halves need to be given in a separate field in the data,
%    next to the full N-way array.
% 2) core-consistency diagnostic. This approach uses a statistic which can be viewed as a measures of noise being modelled 
%    and as such as an indication of whether the model with a certain number of components is still appropriate 
%    (mostly appropriate for PARAFAC). This ranges from 0 to 1 (perfect)
% 3) minium increase in explained variance. A simple procedure that increases the number of components until the new component
%    no longer results in a certain increase of % explained variance.
% 4) degeneracy. This procedure keeps on increasing the number of components until some become degenerate. This uses a statistic
%    denote as the Tucker's congruency coefficient, which ranges from 0 to 1 (fully degenerate). 
%
% The settings of the above strategies are specified in the cfg. 
%
%
% Models SPACE-time or SPACE-FSP are fully described in the following publication. 
% Please cite when either of them are used:
%    van der Meij R, Jacobs J, Maris E (2015). Uncovering phase-coupled oscillatory networks in
% 	      electrophysiological data. Human Brain Mapping
%
% The PARAFAC and PARAFAC2 models are modified such that parameter matrices can be real-valued when the input array is complex-valued. 
% For additional info on the models and the split-half procedure, see the following publication, please cite when either of 
% the models are used:
%    van der Meij, R., Kahana, M. J., and Maris, E. (2012). Phase-amplitude coupling in 
%        human ECoG is spatially distributed and phase diverse. Journal of Neuroscience, 
%        32(1), 111-123.
% 
% For additional information of the models, please also see their low-level functions. And, of course, 
% the README_FIRST.rtf acompanying this toolbox.
%
%
%
%
% Use as
%   [nwaycomp] = nd_nwaydecomposition(cfg, data)
%
% The input data can be any type of FieldTrip-style data structure, as long as the field containing the data
% to be decomposed contains a numerical array. For the SPACE models, the input needs to be Fourier coefficients.
% These Fourier coefficients can be provided in 2 ways. They can either come from (1) custom code together with dimord
% of 'chan_freq_epoch_tap' (with the Fourier coefficients following this dimensionality; nepoch can be 1), 
% or (2) they can can come as output from ft_freqanalysis (cfg.output = 'fourier' or 'powandcsd'; DPSS tapering is supported). 
% In the case of the later, the 'rpt' (trial) dimension will be used as 'epochs' in SPACE terminology. The time dimension will be 
% used as 'tapers' in SPACE terminology. If multiple tapers are present per time-point, these will be handled accordingly. 
% Additionally, if you used method = 'mtmconvol', and frequency-dependent window-lengths, it is highly recommended to supply 
% cfg.fsample, containing the sampling rate of the data in Hz.
%
%
%   cfg.model                = 'parafac', 'parafac2', 'parafac2cp', 'spacetime', 'spacefsp'
%   cfg.datparam             = string, containing field name of data to be decomposed (must be a numerical array)
%   cfg.randstart            = 'no' or number indicating amount of random starts for final decomposition (default = 50) 
%   cfg.numitt               = number of iterations to perform (default = 2500)
%   cfg.convcrit             = number, convergence criterion (default = 1e-8)
%   cfg.degencrit            = number, degeneracy criterion (default = 0.7)
%   cfg.ncomp                = number of components to extract
%   cfg.ncompest             = 'no', 'splithalf', 'corcondiag', 'minexpvarinc', or 'degeneracy' (default = 'no') (FIXME: complexicity of minexpvarinc and degeneracy should same as others)
%   cfg.ncompestrandstart    = 'no' or number indicating amount of random starts for estimating component number (default = cfg.randstart)
%   cfg.ncompeststart        = starting number of components to try to extract (default = 1) (used in splithalf/corcondiag)
%   cfg.ncompestend          = maximum number of components to try to extract (default = 50) (used in splithalf/corcondiag)
%   cfg.ncompeststep         = forward stepsize in ncomp estimation (default = 1) (backward is always 1; used in splithalf/corcondiag)
%   cfg.ncompestshdatparam   = (for 'splithalf'): string containing field-name of partitioned data. Data should be kept in 1x2 cell-array, each partition in one cell
%                              when using SPACE, one can also specify 'oddeven' as cfg.ncompestshdatparam. In this case the data will be partioned using odd/even trials/epochs
%   cfg.ncompestshcritval    = (for 'splithalf'): 1Xnparam vector, critical value to use for selecting number of components using splif half (default = 0.7 for all)
%   cfg.ncompestvarinc       = (for 'minexpvarinc'): minimal required increase in explained variance when increasing number of compononents by cfg.ncompeststep
%   cfg.ncompestcorconval    = (for 'corcondiag'): minimum value of the core consistency diagnostic for increasing the number of components, between 0 and 1 (default is 0.7)
%
%      Algorithm specific options:
%        PARAFAC/2(CP)
%   cfg.complexdims          = vector of 0's and 1's with length equal to number of dimensions in data, indicating dimensions to keep complex
%                              (default = [0 for each dimension])
%        PARAFAC2(CP)
%   cfg.specialdims          = vector with length equal to ndims(dat) with 1, 2 and 3
%                              indicating special modes: 1: outer dim of inner-product                     (i.e. the utility dims)   (ndim-2 dims must be this)
%                                                        2: inner dim of inner-product                     (i.e. the incomplete dim) (only one allowed)
%                                                        3: dim over which inner-products will be computed (i.e. the estimating dim) (only one allowed)
%        SPACEFSP/SPACETIME
%   cfg.Dmode                = string, 'identity', 'kdepcomplex', type of D to estimate/use (default = 'identity')
%
%
%
%         Output nwaycomp:
%             label: cell-array containing labels for each channel
%            dimord: string, dimord of input data
%              comp: cell-array, each component consists of 1 cell, each of these consist of 1 cell per dimension/parameter
%            expvar: number, variance explained by the model
%         tuckcongr: vector, Tucker's congruence coefficient per component
%           scaling: vector orcell-array, dep. on model, containing magnitude/phase scaling coefficients 
%               cfg:
%
%    Possible additional output fields:
%            t3core: a Tucker3 model core-array (vectorized) (not possible for all models)
%        randomstat: structure containing statistics of random estimation of final decomposition (if randomly started)
%     splithalfstat: structure containing statistics for split-half component number estimation procedure
%    corcondiagstat: structure containing statistics for corcondiag component number estimation procedure
%              freq: if present in input data
%
%
% To facilitate data-handling and distributed computing you can use
%   cfg.inputfile   =  ...
%   cfg.outputfile  =  ...
% If you specify one of these (or both) the input data will be read from a *.mat
% file on disk and/or the output data will be written to a *.mat file. These mat
% files should contain only a single variable, corresponding with the
% input/output structure.

%  Undocumented options:
% (experimental)
% cfg.distcomp.system          = 'p2p' or 'torque', distributed computing for random starts (default = [])
% cfg.distcomp.timereq         = scalar, maximum time requirement in seconds of a random start (default = 60*60*24*3 (3 days))
% cfg.distcomp.memreq          = scalar, maximum memory requirement in bytes of a random start (default is autmatically determined)
% cfg.distcomp.p2presubdel     = scalar, resubmission delay for p2p in seconds (default = 60*60*24*3 (3 days))
% cfg.distcomp.inputpathprefix = saves input data with a random name in specified path
%
%

%
% Copyright (C) 2010-2015, Roemer van der Meij, roemervandermeij AT gmail DOT com
%
% This file is part of Nwaydecomnp, see https://github.com/roemervandermeij/nwaydecomp
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

% do the general setup of the function
ft_defaults
ft_preamble init
ft_preamble provenance
ft_preamble trackconfig
ft_preamble debug
ft_preamble loadvar data 

% Set defaults
cfg.model               = ft_getopt(cfg, 'model',                  []);
cfg.datparam            = ft_getopt(cfg, 'datparam',               []);
cfg.complexdims         = ft_getopt(cfg, 'complexdims',            []);
cfg.randstart           = ft_getopt(cfg, 'randstart',              50);
cfg.numitt              = ft_getopt(cfg, 'numitt',                 2500);
cfg.convcrit            = ft_getopt(cfg, 'convcrit',               1e-8);
cfg.degencrit           = ft_getopt(cfg, 'degencrit',              0.7);
cfg.ncomp               = ft_getopt(cfg, 'ncomp',                  []);
cfg.ncompest            = ft_getopt(cfg, 'ncompest',               'no');
cfg.ncompestrandstart   = ft_getopt(cfg, 'ncompestrandstart',      cfg.randstart);
cfg.ncompeststart       = ft_getopt(cfg, 'ncompeststart',          1);
cfg.ncompestend         = ft_getopt(cfg, 'ncompestend',            50);
cfg.ncompeststep        = ft_getopt(cfg, 'ncompeststep',           1);
cfg.ncompestshdatparam  = ft_getopt(cfg, 'ncompestshdatparam',     []); 
cfg.ncompestshcritval   = ft_getopt(cfg, 'ncompestshcritval',      0.7); % expanded to all paramameters later
cfg.specialdims         = ft_getopt(cfg, 'specialdims',            []); % parafac2 specific
cfg.ncompestvarinc      = ft_getopt(cfg, 'ncompestvarinc',         []);
cfg.Dmode               = ft_getopt(cfg, 'Dmode',                  'identity'); %  spacefsp/spacetime specific
cfg.ncompestcorconval   = ft_getopt(cfg, 'ncompestcorconval',      0.7);
cfg.t3core              = ft_getopt(cfg, 't3core',                 'no');

% set distributed computing random starting defaults and throw errors
cfg.distcomp                  = ft_getopt(cfg, 'distcomp',                    []);
cfg.distcomp.system           = ft_getopt(cfg.distcomp, 'system',             []);
cfg.distcomp.memreq           = ft_getopt(cfg.distcomp, 'memreq',             []);
cfg.distcomp.timreq           = ft_getopt(cfg.distcomp, 'timreq',             60*60*24*3);
cfg.distcomp.p2presubdel      = ft_getopt(cfg.distcomp, 'p2presubdel',        60*60*24*3);
cfg.distcomp.inputpathprefix  = ft_getopt(cfg.distcomp, 'inputpathprefix',    []);
if ~isempty(cfg.distcomp.system) && ~strcmp(getenv('USER'),'roevdmei')
  error('distributed computing implementation of random starting is highly experimental, disable error at own risk')
end
if strcmp(cfg.distcomp.system,'p2p') && isempty(cfg.distcomp.p2presubdel)
  error('need to specifiy cfg.distcomp.p2presubdel')
end


% check essential cfg options
if isempty(cfg.model)
  error('please specify cfg.model')
else
  cfg.model = lower(cfg.model); % make sure its lowercase
end
if isempty(cfg.datparam)
  error('you need to specify cfg.datparam')
end


% make a specific check for presence cfg.trials and cfg.channel
if isfield(cfg,'trials') || isfield(cfg,'channel')
  error('cfg.trials and cfg.channel are not supported')
end


% Check the data provided and get dimensions of data for for error checking below (parse filename if data.(cfg.datparam) is not data)
if ~isnumeric(data.(cfg.datparam))
  filevars = whos('-file', data.(cfg.datparam));
  % check file structure
  if numel(filevars)>1
    error('data filename can only contain a single variable')
  end
  if ~any(strcmp(filevars.class,{'single','double'}))
    error('array in data filename needs to be a numeric single or double array')
  end
  ndimsdat = numel(filevars.size);
else
  if ~issingle(data.(cfg.datparam)) && ~isdouble(data.(cfg.datparam))
    error('field specified by cfg.datparam needs to contain a numeric single or double array')
  end
  ndimsdat = ndims(data.(cfg.datparam));
end

% Make sure a dimord is present (in case one uses this outside of FT)
if ~isfield(data,'dimord')
  error(['Input structure needs to contain a dimord field. '...
         'This field identifies the dimensions in the most important data-containing field. See FieldTrip wiki for further details'])
end


% Ncomp and Ncompest related errors
if (isempty(cfg.ncomp) && strcmp(cfg.ncompest,'no'))
  error('you either need to estimate the number of components or set cfg.ncomp')
end
if ~isempty(cfg.ncomp) && ~strcmp(cfg.ncompest,'no')
  warning('cfg.ncomp cannot be used when estimating number of components, cfg.ncomp is ignored')
  cfg.ncomp = [];
end
if strcmp(cfg.ncompest,'splithalf') && isempty(cfg.ncompestshdatparam)
  error('you need to specify cfg.ncompestshdatparam')
end
if strcmp(cfg.ncompest,'minexpvarinc') && isempty(cfg.ncompestvarinc)
  error('must set cfg.ncompestvarinc, the minimal required increase in explained variance')
end
if strncmp(cfg.model,'parafac2',8) && strcmp(cfg.ncompest,'corcondiag')
  error('at the moment corcondiag cannot be used when cfg.model is parafac2/cp')
end
if numel(cfg.ncompeststart) ~= 1 || numel(cfg.ncompestend) ~= 1 || numel(cfg.ncompeststep) ~= 1
  error('improper cfg.ncompestXXX')
end

% check splithalf and ncompestshcritval
if strcmp(cfg.ncompest,'splithalf')
  switch cfg.model
    case {'parafac','parafac2','parafac2cp','spacetime','spacefsp'}
      % implemented
    otherwise
      error('model not supported')
  end
end
if strcmp(cfg.ncompest,'splithalf')
  if (numel(cfg.ncompestshcritval)==1)
    if strncmp(cfg.model,'parafac',7)
      cfg.ncompestshcritval = repmat(cfg.ncompestshcritval,[1 ndimsdat]);
    elseif strcmp(cfg.model,'spacetime') || strcmp(cfg.model,'spacefsp')
      cfg.ncompestshcritval = repmat(cfg.ncompestshcritval,[1 5]);
    end
  else
    if (strncmp(cfg.model,'parafac',7) && (numel(cfg.ncompestshcritval)~=ndimsdat))... % FIXME: likley should contain check for PARAFAC2
        || (strcmp(cfg.model,'spacetime') && (numel(cfg.ncompestshcritval)~=5))...
        || (strcmp(cfg.model,'spacefsp') && (numel(cfg.ncompestshcritval)~=5))
      error('improper size of cfg.ncompestshcritval')
    end
  end
end

% Check model specfic input requirements based on dimord
if   any(strcmp(cfg.model,{'spacetime','spacefsp','spacefsp'})) && (~strcmp(data.dimord,'chan_freq_epoch_tap') || ndimsdat~=4)
  error('incorrect input for specified model')
end

% Model-specific errors
% parafac2/parafac2cp
if strncmp(cfg.model,'parafac2',8) && ndimsdat ~= numel(cfg.specialdims)
  error('length of cfg.specialdims should be equal to number of dimensions in data')
end
if strncmp(cfg.model,'parafac2',8) && ~isfield(data,'ssqdatnoncp')
  error('sneaky trick required')
end
% parafac/parafac2/parafac2cp
if strncmp(cfg.model,'parafac',7) && ndimsdat ~= numel(cfg.complexdims)
  error('length of cfg.complexdims should be equal to number of dimensions in data')
end
if strcmp(cfg.ncompest,'splithalf')
  if strncmp(cfg.model,'parafac',7) && (numel(cfg.complexdims) ~= numel(cfg.ncompestshcritval))
    error('length of cfg.ncompestshcritval should be equal to the number of dimensions in the data')
  end
end


% Handle input for SPACE models
% the below code also applies a trick to dramatically reduce memory in some case
if any(strcmp(cfg.model,{'spacefsp','spacetime'}))
  if strcmp(ft_datatype(data),'freq')
    % Handle output from ft_freqanalysis
    if ~any(strcmp(cfg.datparam,{'fourierspctrm','crsspctrm'}))
      error('cfg.datparam should be either ''fourierspctrm'' or ''crsspctrm'' when input is output from ft_freqanalysis')
    end
    if ~isnumeric(data.(cfg.datparam))
      error('specifying data field as filename is only possible with manually constructed chan_freq_epoch_tap')
    end
    % throw error based on specestmethod, this has to do with current nsample normalization of Fourier coefficients (not important for mtmfft)
    specestmethod = data.cfg.method;
    if ~any(strcmp(specestmethod,{'mtmfft','mtmconvol'}))
      error(['distortion-free scaling of frequency-dependent time-windows lengths over frequency using ft_freqanalysis with method = ' specestmethod ' is not guaranteed'])
    end
    if strcmp(specestmethod,'mtmconvol') && ~isfield(cfg.fsample)
      warning(['Your input resulted from ft_freqanalysis with method = mtmconvol. If you''re also using frequency-dependent window-lengths ' ... 
               'it is highly recommended to supply cfg.fsample, containing the sampling rate of the data in Hz, to correct for ' ...
               'frequency-dependent distortions of power'])
    end
    % failsafe error for if this ever becomes supported in ft_freqananalysis (which it shouldn't)
    if strcmp(cfg.datparam,'fourierspctrm') && (strcmp(data.cfg.keeptrials,'no') || strcmp(data.cfg.keeptrials,'no'))
      error('fourierspctrm must have been computed using keeptrials and keeptapers = yes')
    end
    % failsafe error for if this ever becomes supported in ft_freqananalysis (which it shouldn't)
    if strcmp(cfg.datparam,'crsspctrm') && strcmp(data.cfg.keeptrials,'yes') && strcmp(data.cfg.keeptrials,'no')
      error('crsspctrm must have been computed using keeptrials = yes and keeptapers = no') % this can likely be detected below if it gets implemented
    end
    % failsafe error for if this ever becomes supported in ft_freqananalysis (which it might)
    if any(any(diff(data.cumtapcnt,1,2))) 
      error(['Variable number of tapers over frequency is not supported using output from ft_freqanalysis.'...
             'In order to do this using a custom ''chan_freq_epoch_tap'' see the tutorial on rhythmic components'...
             'and the code below this error message.'])
    end
    % make sure channel cmb representation of data is full in case of crssspctrm
    if strcmp(cfg.datparam,'crssspctrm') % treat crsspctrm as an exception to fourierspctrm in the below
      data = ft_checkdata(data,'cmbrepresentation','full');
    end
    % if no trials are present, add trials as singular dimension to ensure SPACE input is 4-way
    if strncmp(data.dimord,'rpt',3)
      data.dimord = ['rpt_' data.dimord];
      data.(cfg.datparam) = permute(data.(cfg.datparam),[ndimsdat+1 1:ndimsdat]);
    end
    % set
    ntrial = size(data.cumtapcnt,1);
    nfreq  = size(data.cumtapcnt,2);
    nchan  = numel(data.label);
    ntaper = nchan; 
    dat    = complex(NaN(nchan,nfreq,ntrial,ntaper),NaN(nchan,nfreq,ntrial,ntaper));
    tapcnt = data.cumtapcnt(:,1); % explicitly only use ntap of first freq due to above 
    % construct new dat
    for itrial = 1:ntrial
      trialtapind = sum(tapcnt(1:itrial-1))+1:sum(tapcnt(1:itrial-1))+tapcnt(itrial); % ow
      for ifreq = 1:nfreq
        % first, select data and create csd
        if strcmp(cfg.datparam,'fourierspctrm')
          currfour = permute(data.(cfg.datparam)(trialtapind,:,ifreq,:),[2 1 3 4]); % will work regardless of time dim presence/absence
          currfour = currfour(:,:); % this unfolds the dimensions other than chan, will work regardless of time dim presence/absence
          % get rid of NaNs (should always be the same over channels)
          currfour(:,isnan(currfour(1,:))) = [];
          %%%%%%
          % UNDO double scaling in ft_specest_mtmconvol
          % There is currently a double scaling applied in ft_specest_mtmconvol. This will likely not hurt other analyses,
          % but causes a distortion over frequencies that SPACE is sensitive for.
          % This scaling is only dependent on frequency for fourierspctrm
          % This scaling requires cfg.fsample to be given (... :( )
          if strcmp(data.cfg.method,'mtmconvol') && isfield(cfg,'fsample')
            % reconstruct tinwinsample
            t_ftimwin     = ft_findcfg(out.cfg,'t_ftimwin');
            timwinnsample = round(t_ftimwin(ifreq) .* cfg.fsample);
            % undo additional the scaling by sqrt(2./ timwinnsample)
            currfour      = currfour ./ sqrt(2 ./ timwinnsample); 
          end
          %%%%%%
          % obtain count of time-points and tapers
          currntimetap = size(currfour,2);
          % compute the csd
          csd = currfour*currfour';
          % correct for number of time-points and tapers (if fourierspctrm, individual tapers and time-points (if mtmconvol) are not aggegrated, enforced above)
          % this step is CRUCIAL if we want to interpret the loadings of the trial profile
          csd = csd ./ currntimetap; 
        else
          currcsd = permute(data.(cfg.datparam)(itrial,:,:,ifreq,:),[2 3 5 1 4]); % will work regardless of time dim presence/absence
          % get rid of NaNs (should always be the same over channel-pairs)
          currcsd(:,:,isnan(squeeze(currcsd(1,1,:)))) = [];
          %%%%%%
          % UNDO double scaling in ft_specest_mtmconvol
          % There is currently a double scaling applied in ft_specest_mtmconvol. This will likely not hurt other analyses,
          % but causes a distortion over frequencies that SPACE is sensitive for.
          % This scaling is only dependent on frequency for fourierspctrm
          % This scaling requires cfg.fsample to be given (... :( )
          if strcmp(data.cfg.method,'mtmconvol') && isfield(cfg,'fsample')
            % reconstruct tinwinsample
            t_ftimwin     = ft_findcfg(out.cfg,'t_ftimwin');
            timwinnsample = round(t_ftimwin(ifreq) .* cfg.fsample);
            % undo additional the scaling by (sqrt(2./ timwinnsample)).^2 = (2./ timwinnsample) 
            % the scaling is performed on the level of individual taper-specific Fourier coefficients, and currcsd contains the summed cross-products of these
            currcsd       = currcsd ./ (2 ./ timwinnsample); 
          end
          %%%%%%
          % obtain count of time-points (tapers are never kept if crsspctrm, enforced above)
          currntime = size(currcsd,3);
          % obtain csd
          csd = sum(currcsd,3); 
          % correct for number of time-points (if crsspctrm, time-points (if mtmconvol) are not aggegrated, but tapers are averaged, enforced above)
          % this step is CRUCIAL if we want to interpret the loadings of the trial profile
          csd = csd ./  currntime; % this is a rather circumstantial way to do this, but is such to keep it similar to above
        end
        %%%%%%%%%
        % Reduce SPACE memory load and computation time by replacing each chan_taper matrix by the
        % Eigenvectors of its chan_chan cross-products weighted by sqrt(Eigenvalues).
        % This is possible because (1) SPACE only uses the cross-products of the chan_taper matrices
        % (i.e. the frequency- and trial-specific CSD) and (2) the Eigendecomposition of a symmetric
        % matrix A is A = VLV'.
        % As such, VL^.5 has the same cross-products as the original chan_taper matrix.
        [V L] = eig(csd);
        L     = diag(L);
        tol   = max(size(csd))*eps(max(L)); % compute tol using matlabs default
        zeroL = L<tol;
        eigweigth = V(:,~zeroL)*diag(sqrt(L(~zeroL)));
        % positive semi-definite failsafe
        if any(L<-tol) || any(~isreal(L))
          error('csd not positive semidefinite')
        end
        %%%%%%%%%
        % save in dat
        currm = size(eigweigth,2);
        dat(:,ifreq,itrial,1:currm) = eigweigth;
      end
    end
    % trim dat (actual ntaper can be lower than ntaper, which depends on the data, hence the need for trimming)
    notnan = logical(squeeze(sum(sum(~isnan(squeeze(dat(1,:,:,:))),2),1)));
    dat = dat(:,:,:,notnan);
    % clear old dat 
    data.(cfg.datparam) = [];
  elseif strcmp(data.dimord,'chan_freq_epoch_tap')
    % Handle output from custom code with dimensions 'chan_freq_epoch_tap'
    if ~isnumeric(data.(cfg.datparam))
      warning('not optimizing data because data is specified as')
    else
      % only apply trick if ntaper exceeds nchan
      if size(data.(cfg.datparam),4)>size(data.(cfg.datparam),1)
        nepoch = size(data.(cfg.datparam),3);
        nfreq  = size(data.(cfg.datparam),2);
        nchan  = size(data.(cfg.datparam),1);
        ntaper = nchan;
        dat    = complex(NaN(nchan,nfreq,nepoch,ntaper),NaN(nchan,nfreq,nepoch,ntaper));
        % construct new dat
        for iepoch = 1:nepoch
          for ifreq = 1:nfreq
            % first, select data and create csd
            currfour = permute(data.(cfg.datparam)(:,ifreq,iepoch,:),[1 4 2 3]); 
            % get rid of NaNs (should always be the same over channels)
            currfour(:,isnan(currfour(1,:))) = [];
            % compute the csd
            csd = currfour*currfour';
            %%%%%%%%%
            % Reduce SPACE memory load and computation time by replacing each chan_taper matrix by the
            % Eigenvectors of its chan_chan cross-products weighted by sqrt(Eigenvalues).
            % This is possible because (1) SPACE only uses the cross-products of the chan_taper matrices
            % (i.e. the frequency- and trial-specific CSD) and (2) the Eigendecomposition of a symmetric
            % matrix A is A = VLV'.
            % As such, VL^.5 has the same cross-products as the original chan_taper matrix.
            [V L] = eig(csd);
            L     = diag(L);
            tol   = max(size(csd))*eps(max(L)); % compute tol using matlabs default
            zeroL = L<tol;
            eigweigth = V(:,~zeroL)*diag(sqrt(L(~zeroL)));
            % positive semi-definite failsafe
            if any(L<-tol) || any(~isreal(L))
              error('csd not positive semidefinite')
            end
            %%%%%%%%%
            % save in dat
            currm = size(eigweigth,2);
            dat(:,ifreq,iepoch,1:currm) = eigweigth;
          end
        end
        % trim dat (actual ntaper can be lower than ntaper, which depends on the data, hence the need for trimming)
        notnan = logical(squeeze(sum(sum(~isnan(squeeze(dat(1,:,:,:))),2),1)));
        dat = dat(:,:,:,notnan);
        % clear old dat
        data.(cfg.datparam) = [];
      end
    end
  else
    error('Input data structure is not supported for SPACE-FSP or SPACE-time. Please see function help for supported input.')
  end
end


% Set several easy to work with variables
if ~exist(dat,'var')
  dat         = data.(cfg.datparam);
end
model         = cfg.model;
nitt          = cfg.numitt;
convcrit      = cfg.convcrit;
degencrit     = cfg.degencrit;
ncomp         = cfg.ncomp;
nrand         = cfg.randstart;
nrandestcomp  = cfg.ncompestrandstart;
estnum        = [cfg.ncompeststart cfg.ncompestend cfg.ncompeststep];
estshcritval  = cfg.ncompestshcritval;
distcomp      = cfg.distcomp;
expvarinc     = cfg.ncompestvarinc;
corconval     = cfg.ncompestcorconval;
% set model specific ones and create modelopt
switch cfg.model
  case 'parafac'
    modelopt = {'compmodes',cfg.complexdims};
  case 'parafac2'
    modelopt = {'compmodes',cfg.complexdims,'specmodes',cfg.specialdims};
  case 'parafac2cp'
    if strcmp(cfg.ncompest,'splithalf')
      % do hacks for parafac2cp
      modelopt = {'compmodes',cfg.complexdims,'ssqdatnoncp',data.ssqdatnoncp,'specmodes',cfg.specialdims,'ssqdatnoncppart1',data.ssqdatnoncppart1,'ssqdatnoncppart2',data.ssqdatnoncppart2};
    else
      modelopt = {'compmodes',cfg.complexdims,'ssqdatnoncp',data.ssqdatnoncp,'specmodes',cfg.specialdims};
    end
  case 'spacetime'
    modelopt = {'freq',data.freq,'Dmode',cfg.Dmode};
  case 'spacefsp'
    modelopt = {'Dmode',cfg.Dmode};
  otherwise
    error('model not supported')
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Component estimation
switch cfg.ncompest
  
  case 'splithalf'
    
    if strcmp(cfg.ncompestshdatparam,'oddeven')
      if ~strcmp(model,'spacefsp') && ~strcmp(model,'spacetime')
        error('cfg.ncompestshdatparam = ''oddeven'' is only supported for SPACE-time and SPACE-FSP. Please provide partioned data in a 1x2 cell-array and specify its field name in cfg.ncompestshdatparam')
      end
      if size(dat,3)==1
        error('splithalf procedure is only suitable for when the number trials/epochs is bigger than 1')
      end
      % extract partitions
      datpart1 = dat(:,:,1:2:size(dat,3),:);
      datpart2 = dat(:,:,2:2:size(dat,3),:);
    else
      % extract partitions
      datpart1 = data.(cfg.ncompestshdatparam){1};
      datpart2 = data.(cfg.ncompestshdatparam){2};
    end
    
    % perform splithalf component number estimate
    [ncomp, splithalfstat] = splithalf(model, datpart1, datpart2, nrandestcomp, estnum, estshcritval, nitt, convcrit, degencrit, distcomp, modelopt{:}); % subfunction
    
  case 'degeneracy'
    % Warn about component estimation
    warning('this is a very liberal method for component estimation')
    
    % estimate ncomp
    [ncomp] = degeneracy(model, dat, nrandestcomp, estnum, nitt, convcrit, degencrit, distcomp, modelopt{:}); % subfunction
    
  case 'minexpvarinc'
    % estimate ncomp
    [ncomp] = minexpvarinc(model, dat, nrandestcomp, estnum, nitt, convcrit, degencrit, distcomp, expvarinc, modelopt{:}); % subfunction
    
  case 'corcondiag'
    % estimate ncomp
    [ncomp, corcondiagstat] = corcondiag(model, dat, nrandestcomp, estnum, nitt, convcrit, degencrit, distcomp, corconval, modelopt{:}); % subfunction
    
    % extract startval if nrand is the same
    if (nrandestcomp==nrand) && (corcondiagstat.randomstatsucc.convcrit==convcrit) % convcrit is the same as used for randomstart below 
      startval   = corcondiagstat.randomstatsucc.startvalglobmin;
      randomstat = corcondiagstat.randomstatsucc;
    end
    
  case 'no'
    % do nothing, ncomp is set
    
  otherwise
    error('method for estimating number of components not supported')
    
end % switch ncompest
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Final decomposition
disp('performing FINAL decomposition')
% call low-level function and get start values if requested
% set up general ooptions
opt = {'nitt', nitt, 'convcrit', convcrit};
switch cfg.model
  case 'parafac'
    if isnumeric(nrand)
      if ~exist('startval','var') && ~exist('randomstat','var')
        [startval, randomstat] = randomstart(model, dat, ncomp, nrand, nitt, convcrit, degencrit, distcomp, [], modelopt{:}); % subfunction
      end
      if strcmp(cfg.t3core,'yes')
        [comp, dum, expvar, scaling, tuckcongr, t3core] = feval(['nwaydecomp_' model], dat, ncomp, 'startval', startval, opt{:}, modelopt{:});
      else
        [comp, dum, expvar, scaling, tuckcongr] = feval(['nwaydecomp_' model], dat, ncomp, 'startval', startval, opt{:}, modelopt{:});
      end
    else
      if strcmp(cfg.t3core,'yes')
        [comp, dum, expvar, scaling, tuckcongr, t3core] = feval(['nwaydecomp_' model], dat, ncomp, opt{:}, modelopt{:});
      else
        [comp, dum, expvar, scaling, tuckcongr] = feval(['nwaydecomp_' model], dat, ncomp, opt{:}, modelopt{:});
      end
    end
  case 'parafac2'
    if isnumeric(nrand)
      if ~exist('startval','var') && ~exist('randomstat','var')
        [startval, randomstat] = randomstart(model, dat, ncomp, nrand, nitt, convcrit, degencrit, distcomp, [], modelopt{:}); % subfunction
      end
      [comp, P, dum, expvar, scaling, tuckcongr] = feval(['nwaydecomp_' model], dat, ncomp, cfg.specialdims, 'startval', startval, opt{:}, modelopt{:});
    else
      [comp, P, dum, expvar, scaling, tuckcongr] = feval(['nwaydecomp_' model], dat, ncomp, cfg.specialdims, opt{:}, modelopt{:});
    end
  case 'parafac2cp'
    if isnumeric(nrand)
      if ~exist('startval','var') && ~exist('randomstat','var')
        [startval, randomstat] = randomstart(model, dat, ncomp, nrand, nitt, convcrit, degencrit, distcomp, [], modelopt{:}); % subfunction
      end
      [comp, P, dum, expvar, scaling, tuckcongr] = feval(['nwaydecomp_' model], dat, ncomp, cfg.specialdims, 'startval', startval, opt{:}, modelopt{1:4});
    else
      [comp, P, dum, expvar, scaling, tuckcongr] = feval(['nwaydecomp_' model], dat, ncomp, cfg.specialdims, 'nitt', nitt, opt{:}, modelopt{1:4});
    end
  case 'spacetime'
    if isnumeric(nrand)
      if ~exist('startval','var') && ~exist('randomstat','var')
        [startval, randomstat] = randomstart(model, dat, ncomp, nrand, nitt, convcrit, degencrit, distcomp, [], modelopt{:}); % subfunction
      end
      if strcmp(cfg.t3core,'yes')
        [comp, dum, dum, expvar, scaling, tuckcongr, t3core] = feval(['nwaydecomp_' model], dat, ncomp, data.freq, 'Dmode', cfg.Dmode, 'startval', startval, opt{:});
      else
        [comp, dum, dum, expvar, scaling, tuckcongr] = feval(['nwaydecomp_' model], dat, ncomp, data.freq, 'Dmode', cfg.Dmode, 'startval', startval, opt{:});
      end
    else
      if strcmp(cfg.t3core,'yes')
        [comp, dum, dum, expvar, scaling, tuckcongr, t3core] = feval(['nwaydecomp_' model], dat, ncomp, data.freq, 'Dmode', cfg.Dmode, opt{:});
      else
        [comp, dum, dum, expvar, scaling, tuckcongr] = feval(['nwaydecomp_' model], dat, ncomp, data.freq, 'Dmode', cfg.Dmode, opt{:});
      end
    end
  case 'spacefsp'
    if isnumeric(nrand)
      if ~exist('startval','var') && ~exist('randomstat','var')
        [startval, randomstat] = randomstart(model, dat, ncomp, nrand, nitt, convcrit, degencrit, distcomp, [], modelopt{:}); % subfunction
      end
      if strcmp(cfg.t3core,'yes')
        [comp, dum, dum, expvar, scaling, tuckcongr, t3core] = feval(['nwaydecomp_' model], dat, ncomp, 'Dmode', cfg.Dmode, 'startval', startval, opt{:});
      else
        [comp, dum, dum, expvar, scaling, tuckcongr] = feval(['nwaydecomp_' model], dat, ncomp, 'Dmode', cfg.Dmode, 'startval', startval, opt{:});
      end
    else
      if strcmp(cfg.t3core,'yes')
        [comp, dum, dum, expvar, scaling, tuckcongr, t3core] = feval(['nwaydecomp_' model], dat, ncomp, 'Dmode', cfg.Dmode, opt{:});
      else
        [comp, dum, dum, expvar, scaling, tuckcongr] = feval(['nwaydecomp_' model], dat, ncomp, 'Dmode', cfg.Dmode, opt{:});
      end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Transform comp to compoment-specific cell-arrays
outputcomp = cell(1,ncomp);
switch model
  case {'parafac','parafac2','parafac2cp'}
    % comp always consist of a loading matrix per dimension
    for icomp = 1:ncomp
      for idim = 1:numel(comp)
        outputcomp{icomp}{idim} = comp{idim}(:,icomp);
      end
    end
  case 'spacetime'
    for icomp = 1:ncomp
      % A,B,C,S
      for iparam = 1:4
        outputcomp{icomp}{iparam} = comp{iparam}(:,icomp);
      end
      % D
      if strcmp(cfg.Dmode,'kdepcomplex')
        outputcomp{icomp}{5} = comp{5}(:,:,icomp);
      else % identity
        outputcomp{icomp}{5} = comp{5}(:,icomp);
      end
    end
  case 'spacefsp'
    for icomp = 1:ncomp
      % A,B,C
      for iparam = 1:3
        outputcomp{icomp}{iparam} = comp{iparam}(:,icomp);
      end
      % L
      outputcomp{icomp}{4} = comp{4}(:,:,icomp);
      % D
      if strcmp(cfg.Dmode,'kdepcomplex')
        outputcomp{icomp}{5} = comp{5}(:,:,icomp);
      else % identity
        outputcomp{icomp}{5} = comp{5}(:,icomp);
      end
    end
end

% set dimord
switch model
  case {'parafac','parafac2','parafac2cp'}
    dimord = data.dimord;
  case 'spacetime'
    dimord = 'A_B_C_S_D';
  case 'spacefsp'
    dimord = 'A_B_C_L_D';
end

% Construct output structure
nwaycomp.label      = data.label;
nwaycomp.dimord     = dimord; 
nwaycomp.comp       = outputcomp;
if any(strcmp(model,{'parafac2','parafac2cp'}))
  nwaycomp.P        = P;
end
nwaycomp.expvar     = expvar;
nwaycomp.tuckcongr  = tuckcongr;
nwaycomp.scaling    = scaling;
if isnumeric(nrand)
  nwaycomp.randomstat = randomstat;
end
if strcmp(cfg.ncompest,'splithalf')
  nwaycomp.splithalfstat = splithalfstat;
end
if strcmp(cfg.ncompest,'corcondiag')
  nwaycomp.corcondiagstat = corcondiagstat;
end
if exist('t3core','var')
  nwaycomp.t3core = t3core;
end


% add certain fields if present in input (some might be mutually exclusive, or undesired, adding all currenlty for completeness)
% general
fieldnames = {'grad','elec','trialinfo'};
nwaycomp = copyfields(data, nwaycomp, fieldnames);
% ft_freqanalysis/connectivityanalysis/timelockanalysis
fieldnames = {'freq','time','dof','labelcmb'}; % this explicitly does not contain 'cumtapcnt','cumsumcnt', as these are controlled for
nwaycomp = copyfields(data, nwaycomp, fieldnames);
% ft_componentanalysis
fieldnames = {'topo','topolabel','unmixing'};
nwaycomp = copyfields(data, nwaycomp, fieldnames);
% ft_sourceanalysis
fieldnames = {'pos','inside','outside','leadfield','dim','tri','transform'};
nwaycomp = copyfields(data, nwaycomp, fieldnames);


% includes certain fields for backwards compatability
fieldnames = {'ampfreq','phasfreq','freq_old','label_old'};
nwaycomp = copyfields(data, nwaycomp, fieldnames);


% do the general cleanup and bookkeeping at the end of the function
ft_postamble debug
ft_postamble trackconfig
ft_postamble provenance
ft_postamble previous data
ft_postamble history nwaycomp
ft_postamble savevar nwaycomp 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Subfunction for cfg.ncompest = 'degeneracy'            %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ncomp] = degeneracy(model, dat, nrand, estnum, nitt, convcrit, degencrit, distcomp, varargin)

% get model specific options from keyval
% not necessary, not explicitly used

% Display start
disp('degen-only: performing component number estimation by only checking degeneracy at each solution')


% Estimate number of components by incremently increasing number and calculating split-half correlations
ncomp  = 0;
for incomp = 1:estnum(2)
  disp(['degen-only: number of components = ' num2str(incomp) ' of max ' num2str(estnum(2))]);
  disp('degen-only: performing decomposition');
  
  
  % Get decompositions for current incomp
  % get start values for decomposition for current incomp
  [dum, randomstat] = randomstart(model, dat, incomp, nrand, nitt, convcrit, degencrit, distcomp, ['degen-only ncomp = ' num2str(incomp) ' - '], varargin{:}); % subfunction
  
  % see if there are any non-degenerate start values and set flag if otherwise
  if length(randomstat.degeninit)==nrand
    % try again with another round
    [dum, randomstat] = randomstart(model, dat, incomp, nrand, nitt, convcrit, degencrit, distcomp, ['degen-only ncomp = ' num2str(incomp) ' - '], varargin{:}); % subfunction
    if length(randomstat.degeninit)==nrand
      degenflg = true;
    else
      degenflg = false;
    end
  else
    degenflg = false;
  end
  
  
  % Act on degeneracy of solutions
  % first do a check for degeneracy (as split half makes no sense if solutions are degenerate)
  if degenflg
    disp('degen-only: random initializations only returned likely degenerate solutions')
    disp(['degen-only: final number of components = ' num2str(ncomp)]);
    break
  end
  
  
  % If all checks are passed, update ncomp
  ncomp = incomp;
end % incomp



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Subfunction for cfg.ncompest = 'minexpvarinc'          %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ncomp] = minexpvarinc(model, dat, nrand, estnum, nitt, convcrit, degencrit, distcomp, expvarinc, varargin)

% get model specific options from keyval
% not necessary, not explicitly used

% Display start
disp('minexpvarinc: performing minimum increase in expvar component number estimation')


% Estimate number of components by incremently increasing number and calculating split-half correlations
% save current expvar
lastexpvar = 0;
ncomp  = 0;
for incomp = 1:estnum(2)
  disp(['minexpvarinc: number of components = ' num2str(incomp) ' of max ' num2str(estnum(2))]);
  disp('minexpvarinc: performing decomposition');
  
  
  % Get decompositions for current incomp
  % get start values for decomposition for current incomp
  [dum, randomstat] = randomstart(model, dat, incomp, nrand, nitt, convcrit, degencrit, distcomp, ['minexpvarinc ncomp = ' num2str(incomp) ' - '], varargin{:}); % subfunction
  currexpvar = randomstat.expvar(1);
  
  % see if there are any non-degenerate start values and set flag if otherwise
  if length(randomstat.degeninit)==nrand
    degenflg = true;
  else
    degenflg = false;
  end
  
  
  % Act on degeneracy of solutions
  % first do a check for degeneracy (as split half makes no sense if solutions are degenerate)
  if degenflg
    disp('minexpvarinc: random initializations only returned likely degenerate solutions')
    disp(['minexpvarinc: final number of components = ' num2str(ncomp)]);
    break
  end
  
  
  % Act on explained variance increase
  if incomp ~= 1
    disp(['minexpvarinc: going from ncomp = ' num2str(ncomp) ' to ' num2str(incomp) '  lead to ' num2str((currexpvar - lastexpvar),'%-2.2f') '% increase in explained variance'])
    if (currexpvar - lastexpvar) < expvarinc
      disp('minexpvarinc: increase in explained variance not sufficient')
      disp(['minexpvarinc: final number of components = ' num2str(ncomp)]);
      break
    end
  end
  
  
  % If all checks are passed, update ncomp and lastexpvar
  ncomp = incomp;
  lastexpvar = currexpvar;
end % incomp



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Subfunction for cfg.ncompest = 'corcondiag'           %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ncomp, corcondiagstat] = corcondiag(model, dat, nrand, estnum, nitt, convcrit, degencrit, distcomp, corconval, varargin)

% get model specific options from keyval
switch model
  case 'parafac'
    compmodes = keyval('compmodes', varargin);
  case 'parafac2'
    compmodes = keyval('compmodes', varargin);
    specmodes = keyval('specmodes', varargin);
  case 'parafac2cp'
    compmodes   = keyval('compmodes',   varargin);
    specmodes   = keyval('specmodes',   varargin);
    ssqdatnoncp = keyval('ssqdatnoncp', varargin);
  case 'spacetime'
    freq  = keyval('freq', varargin);
    Dmode = keyval('Dmode', varargin);
  case 'spacefsp'
    Dmode = keyval('Dmode', varargin);
  otherwise
    error('model not supported')
end

% Display start
disp('corcondiag: performing component number estimation based on core consistency diagnostic')

% Estimate number of components by comparing the t3core array the identity-array of the same size
allcomp       = [];
allt3core     = [];
allcorcondiag = [];
allrandomstat = [];
ncompsucc     = [];
incomp  = estnum(1);
succes  = false;
while ~succes % the logic used here is identical as in splithalf, they should be changed SIMULTANEOUSLY or both subfunctions should be merged with switches
  disp(['corcondiag: number of components = ' num2str(incomp) ' of max ' num2str(estnum(2))]);
  disp('corcondiag: performing decomposition');
  
  
  % Get decompositions for current incomp
  % get quickly computed start values for decomposition for current incomp
  [startval, randomstat] = randomstart(model, dat, incomp, nrand, nitt, convcrit, degencrit, distcomp, ['corcondiag ncomp = ' num2str(incomp) ' - '], varargin{:}); % subfunction
  
  
  % see if there are any non-degenerate start values and set flag if otherwise
  if length(randomstat.degeninit)==nrand
    % try again with another round
    [startval, randomstat] = randomstart(model, dat, incomp, nrand, nitt, convcrit, degencrit, distcomp, ['corcondiag ncomp = ' num2str(incomp) ' - '], varargin{:}); % subfunction
    if length(randomstat.degeninit)==nrand
      degenflg = true;
    else
      degenflg = false;
    end
  else
    degenflg = false;
  end
  
 
  % get final decompositions for current incomp
  opt = {'nitt', nitt, 'convcrit', convcrit, 'dispprefix',['corcondiag ncomp = ' num2str(incomp) ': ']};
  switch model
    case 'parafac'
      [estcomp,dum,dum,dum,dum,t3core] = feval(['nwaydecomp_' model], dat, incomp, 'startval', startval, 'compmodes', compmodes, opt{:});
      %     case 'parafac2' FIXME: parafac2/cp needs to output a t3core, and crosscompcongruence needs to be aware of specmodes (see splithalf)
      %       [estcomp,dum,dum,dum,dum,dum,t3core] = feval(['nwaydecomp_' model], dat, incomp, specmodes, 'startval', startval, 'compmodes', compmodes, opt{:});
      %     case 'parafac2cp'
      %       [estcomp,dum,dum,dum,dum,dum,t3core] = feval(['nwaydecomp_' model], dat, incomp, specmodes, 'startval', startval, 'compmodes', compmodes, opt{:}, 'ssqdatnoncp', ssqdatnoncp);
    case 'spacetime'
      [estcomp,dum,dum,dum,dum,dum,t3core] = feval(['nwaydecomp_' model], dat, incomp, freq, 'Dmode', Dmode, 'startval', startval, opt{:});
    case 'spacefsp'
      [estcomp,dum,dum,dum,dum,dum,t3core] = feval(['nwaydecomp_' model], dat, incomp, 'Dmode', Dmode, 'startval', startval, opt{:});
    otherwise
      error('model not yet supported in corecondiag component number estimation')
  end
  
  % create identity array and compute core consistency diagnostic
  if incomp==1
    corcondiag = 1;
  else
    ndimsdat = round((log(numel(t3core))/log(incomp))); % get ndimsdat out of the t3core, necessary when dat is a filename containing the data
    ident = zeros(incomp^ndimsdat,1);
    for iident = 1:incomp
      ind = iident;
      for idim = 2:ndimsdat
        ind = ind +incomp.^(idim-1) .* (iident-1);
      end
      ident(ind) = 1;
    end
    t3core = t3core(:); % vectorize
    corcondiag = 1 - (sum((ident-t3core).^2)/sum(t3core.^2));
  end
  
  % save incomp specifics
  allcomp{incomp}       = estcomp;
  allt3core{incomp}     = t3core;
  allcorcondiag{incomp} = corcondiag;
  allrandomstat{incomp} = randomstat;
  
  
  % set critfailflg
  if corcondiag < corconval
    critfailflg = true;
  else
    critfailflg = false;
  end
  
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% determine incomp succes 
  % Act on critfailflg and degenflg
  disp(['corcondiag: core consistency diagnostic at ncomp = ' num2str(incomp) ' was ' num2str(corcondiag,'%-2.2f')])
  if critfailflg || degenflg
    % When current incomp fails, decrease incomp. 
    % Then, incomp-1 has either been performed or not 
    % If not performed, succes=false and continue. Then, If incomp==0 succes='true'. 
    % If it has been performed, it can only be successful, succes=true. (error check for unsuccesful)
    
    % update fail fields
    if ~degenflg
      disp(['corcondiag: minimum core consistency diagnostic of ' num2str(corconval) ' has not been reached at ncomp = ' num2str(incomp)]);
      t3corefail     = t3core;
      corcondiagfail = corcondiag;
      randomstatfail = randomstat;
      failreason = 'core consistency diagnostic';
    elseif degenflg
      disp('corcondiag: random initializations only returned likely degenerate solutions')
      t3corefail     = []; % no t3core is computed
      corcondiagfail = []; % no t3core, so no corcondiag
      randomstatfail = randomstat;
      failreason     = 'degeneracy';
    end
    ncompsucc{incomp} = false;
    % decrease incomp by 1
    incomp = incomp-1; % the code below requires this step to be one (succ fields are always at the highest ncomp)
    
    % check for incomp == 0
    if incomp == 0
      warning('corcondiag: NO COMPONENTS REACHED CORE CONSISTENCY DIAGNOSTIC CRITERION');
      % set 'succes' status and final number of components
      succes = true;
      ncomp  = 1;
      t3coresucc     = [];
      corcondiagsucc = [];
      randomstatsucc = [];
      failreason = 'core consistency diagnostic criterion fail at ncomp = 1';
      break
    end
    
    % check for new incomp already performed and stop accordingly
    if ~isempty(ncompsucc{incomp})
      if ncompsucc{incomp}
        % set succes status and final number of components
        succes = true;
        ncomp  = incomp;
      elseif ~ncompsucc{incomp}
        % this should not be possible
        error('unexpected error in ncomp estimation')
      end
    end
    
  else
    % When current incomp succeeds, increase incomp. 
    % Then, incomp+i can either be at the maximum or not.
    % If at the maximum, succes = true.
    % If not at the maximum, increment.
    % Then, there have either been attempts at incomp+i or there have not.
    % If not, increment incomp with stepsize limited by the maximum.
    % If there have been incomp+i's, they can only have failed, and none should be present at incomp+stepsize+i
    % Check for this. 
    % Then, find the closest fail. If it is incomp+1, succes = true. If not, increment up until max.

    % update succes fields and ncompsucc
    t3coresucc     = t3core;
    corcondiagsucc = corcondiag;
    randomstatsucc = randomstat;
    ncompsucc{incomp} = true;
    
    % check whether maximum has been reached, and increment incomp intelligently otherwise
    if incomp==estnum(2)
      disp(['corcondiag: succesfully reached a priori determined maximum of ' num2str(estnum(2)) ' components']);
      t3corefail     = [];
      corcondiagfail = [];
      randomstatfail = [];
      failreason = ['corcondiag stopped: reached a priori maximum of ' num2str(estnum(2)) ' components'];
      % set succes status
      succes = true;      
      ncomp  = incomp;
      
    else % keep on incrementing
      % check for previous fails at incomp+
      if isempty(ncompsucc(incomp+1:end))
        % no previous incomp+ solutions found, increment incomp with step (and check for ncompestend)
        incomp = min(incomp + estnum(3),estnum(2));
        
      else
        % sanity check
        if any([ncompsucc{incomp+1:end}])
          % incomp+i should never be successful
          error('unexpected error in ncomp estimation')
        end
        % incomp+i found, check whether next ncomp was failing
        if ncompsucc{incomp+1}==0
          % next ncomp failed, set succes status for current
          succes = true;
          ncomp  = incomp;
        else
          % next incomp was not the fail, find closest after that (depends on estnum(3))
          found = false;
          while ~found
            if isempty(ncompsucc{incomp+1})
              incomp = incomp+1;
            else
              found = true;
            end
          end
          % correct for ncompestend (because of check for ncompestend above, the correct incomp will never have been tested)
          incomp = min(incomp,estnum(3));
        end
      end
    end
    
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%
  
end % incomp
disp(['corcondiag: final number of components = ' num2str(ncomp)]);


%
% FIXME: NO MODEL SPECIFIC COEFFICIENTS HERE YET!!!!
% FIXME: the below piece of code is shit and should be reimplemented to handle flexible non-linearly increasing not-starting-from-1 ncomp's 
% Calcute cross-ncomp congruence
% the organization is as follows:
% crossncomp{i} = all cross-comp calculations started from ncomp = i
% crossncomp{i}{j} = cross-comp calculations of ncomp = i, with ncomp = j
% crossncomp{i}{j} = I*J matrix with congruence coefficents between all components
% nparam      = numel(estcomp);
% ncrossncomp = length(allcomp);
% ncompindex  = 1:ncrossncomp; % index of all cross ncomp calculations
% crossncomp  = cell(1,ncrossncomp);
% % loop over number of comps in allcomp
% for incomp = ncompindex
%   remcomp = ncompindex;
%   remcomp(incomp) = []; % remaining ncomps to calculate over
%   % loop over cross-ncomp-calculations
%   for incross = 1:length(remcomp)
%     currncross = remcomp(incross);
%     % set n's and preset crossn
%     nseedcomp = incomp;
%     ntargcomp = currncross;
%     crossn = zeros(nparam,nseedcomp,ntargcomp);
%     currseed = allcomp{incomp};
%     currtarg = allcomp{currncross};
%     % loop over parameters
%     for iparam = 1:nparam
%       currseedparam = currseed{iparam};
%       currtargparam = currtarg{iparam};
%       % loop over components of seed
%       for icompseed = 1:nseedcomp
%         if ndims(currseedparam)==3 || (ndims(currseedparam)==2 && size(currseedparam,2)~=nseedcomp)
%           currseedcomp = currseedparam(:,:,icompseed);
%           currseedcomp = reshape(permute(currseedcomp,[1 3 2]),[size(currseedcomp,1)*size(currseedcomp,2) size(currseedcomp,3)]);
%         elseif ndims(currseedparam)<=2
%           currseedcomp = currseedparam(:,icompseed);
%         end
%         % loop over components of target
%         for icomptarg = 1:ntargcomp
%           if ndims(currtargparam)==3 || (ndims(currtargparam)==2 && size(currtargparam,2)~=ntargcomp)
%             currtargcomp = currtargparam(:,:,icomptarg);
%             currtargcomp = reshape(permute(currtargcomp,[1 3 2]),[size(currtargcomp,1)*size(currtargcomp,2) size(currtargcomp,3)]);
%           elseif ndims(currtargparam)<=2
%             currtargcomp = currtargparam(:,icomptarg);
%           end
%           if all(size(currseedcomp)==size(currtargcomp))
%             currseedcomp = currseedcomp ./ sqrt(sum(abs(currseedcomp).^2));
%             currtargcomp = currtargcomp ./ sqrt(sum(abs(currtargcomp).^2));
%             crossn(iparam,icompseed,icomptarg) = abs(currseedcomp' * currtargcomp);
%           else
%             crossn(iparam,icompseed,icomptarg) = 0;
%           end
%         end % end loop over target components
%       end  % end loop over seed components
%     end % end loop over parameters
%     % mean over parameters and mean over partitions
%     crossn = squeeze(mean(crossn,1));
%     crossncomp{incomp}{incross} = crossn;
%   end % end loop over cross-ncomp-calculations
% end % end loop over number of comps in allcomp

% create corcondiagstat
corcondiagstat.ncomp           = ncomp;
corcondiagstat.corcondiagsucc  = corcondiagsucc;
corcondiagstat.corcondiagfail  = corcondiagfail;
corcondiagstat.t3coresucc      = t3coresucc;
corcondiagstat.t3corefail      = t3corefail;
corcondiagstat.randomstatsucc  = randomstatsucc;
corcondiagstat.randomstatfail  = randomstatfail;
corcondiagstat.failreason      = failreason;
corcondiagstat.crosscompcongr  = [];
corcondiagstat.allcomp         = allcomp;
corcondiagstat.allt3core       = allt3core;
corcondiagstat.allcorcondiag   = allcorcondiag;
corcondiagstat.allrandomstat   = allrandomstat;
corcondiagstat.ncompsucc       = ncompsucc;

  
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Subfunction for cfg.ncompest = 'splithalf'             %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ncomp,splithalfstat] = splithalf(model, datpart1, datpart2, nrand, estnum, estshcritval, nitt, convcrit, degencrit, distcomp, varargin)

% get model specific options from keyval
switch model
  case 'parafac'
    compmodes = keyval('compmodes', varargin);
  case 'parafac2'
    compmodes = keyval('compmodes', varargin);
    specmodes = keyval('specmodes', varargin);
  case 'parafac2cp'
    compmodes   = keyval('compmodes',   varargin);
    specmodes   = keyval('specmodes',   varargin);
    ssqdatnoncppart1 = keyval('ssqdatnoncppart1', varargin);
    ssqdatnoncppart2 = keyval('ssqdatnoncppart2', varargin);
  case 'spacetime'
    freq  = keyval('freq', varargin);
    Dmode = keyval('Dmode', varargin);
  case 'spacefsp'
    Dmode = keyval('Dmode', varargin);
  otherwise
    error('model not supported')
end

% Display start
disp('split-half: performing split-half component number estimation')

% Estimate number of components by incremently increasing number and calculating split-half correlations
allcomp           = [];
allcompsh         = [];
allpartcombcompsh = [];
allrandomstat     = [];
incomp  = estnum(1);
succes  = false;
while ~succes % the logic used here is identical as in corcondiag, they should be changed SIMULTANEOUSLY or both subfunctions should be merged with switches
 
  disp(['split-half: number of components = ' num2str(incomp) ' of max ' num2str(estnum(2))]);
  disp('split-half: performing decomposition and computing split-half coefficients');
  
  
  % Get decompositions for current incomp
  % get start values for decomposition for current incomp
  [startval1, randomstat1] = randomstart(model, datpart1, incomp, nrand, nitt, convcrit, degencrit, distcomp, ['split-half part 1 ncomp = ' num2str(incomp) ' - '], varargin{:}); % subfunction
  [startval2, randomstat2] = randomstart(model, datpart2, incomp, nrand, nitt, convcrit, degencrit, distcomp, ['split-half part 2 ncomp = ' num2str(incomp) ' - '], varargin{:}); % subfunction
  
  % see if there are any non-degenerate start values and set flag if otherwise (and try again if it only goes for one partition)
  if length(randomstat1.degeninit)==nrand && length(randomstat2.degeninit)==nrand
    degenflg = true;
  elseif length(randomstat1.degeninit)==nrand && length(randomstat2.degeninit)~=nrand % if random inits for part1 do not contain non-degenerate solutions, but they do for part2, retry part1
    [startval1, randomstat1] = randomstart(model, datpart1, incomp, nrand, nitt, convcrit, degencrit, distcomp, ['split-half ncomp = ' num2str(incomp) ' - '], varargin{:}); % subfunction
    if length(randomstat1.degeninit)==nrand % if there are still no non-degenerates
      degenflg = true;
    end
  elseif length(randomstat1.degeninit)~=nrand && length(randomstat2.degeninit)==nrand% if random inits for part2 do not contain non-degenerate solutions, but they do for part1, retry part2
    [startval2, randomstat2] = randomstart(model, datpart2, incomp, nrand, nitt, convcrit, degencrit, distcomp, ['split-half ncomp = ' num2str(incomp) ' - '], varargin{:}); % subfunction
    if length(randomstat2.degeninit)==nrand % if there are still no non-degenerates
      degenflg = true;
    end
  else
    degenflg = false;
  end
  
  % The splithalf logic is now as follows: the splithalf criterion fails, if the splithalf coeffcient is not surpassed for ANY combination of random starts for both partitions
  % The logic used to be: the splithalf criterion fails, if the splithalf coefficient is not surpassed for the random starts which had the maximum expvar from both partitions
  
  % extract all non-degenerate startvalues
  startval1 = randomstat1.startvalall(setdiff(1:nrand,randomstat1.degeninit));
  startval2 = randomstat2.startvalall(setdiff(1:nrand,randomstat2.degeninit));
  % select the minimum non-degenerate startvalues as the maximum FIXME: like, not optimal. (I'm lazy right now, and doesn't matter for SPACE with identity Dmode)
  nondegennrand = min(numel(startval1),numel(startval2));
  startval1 = startval1(1:nondegennrand);
  startval2 = startval2(1:nondegennrand);
  
  % get final decompositions for current incomp
  estcomp = cell(1,2);
  estcomp{1} = cell(1,nondegennrand);
  estcomp{2} = cell(1,nondegennrand);
  % set nitt to a small number in case the solution in random starts has not converged yet (which will cause the below to last very long)
  fnitt = 10; % FIXME: this isn't really necessary, here because of historical reasons
  % set up general options
  optsh1 = {'nitt', fnitt, 'convcrit', convcrit, 'dispprefix',['split-half part 1 ncomp = ' num2str(incomp) ': ']};
  optsh2 = {'nitt', fnitt, 'convcrit', convcrit, 'dispprefix',['split-half part 2 ncomp = ' num2str(incomp) ': ']};
  for inondegenrand = 1:nondegennrand
    switch model
      case 'parafac'
        [estcomp{1}{inondegenrand},dum,dum,dum,dum] = feval(['nwaydecomp_' model], datpart1, incomp, 'startval', startval1{inondegenrand}, 'compmodes', compmodes, optsh1{:});
        [estcomp{2}{inondegenrand},dum,dum,dum,dum] = feval(['nwaydecomp_' model], datpart2, incomp, 'startval', startval2{inondegenrand}, 'compmodes', compmodes, optsh2{:});
      case 'parafac2'
        [estcomp{1}{inondegenrand},dum,dum,dum,dum,dum] = feval(['nwaydecomp_' model], datpart1, incomp, specmodes, 'startval', startval1{inondegenrand}, 'compmodes', compmodes, optsh1{:});
        [estcomp{2}{inondegenrand},dum,dum,dum,dum,dum] = feval(['nwaydecomp_' model], datpart2, incomp, specmodes, 'startval', startval2{inondegenrand}, 'compmodes', compmodes, optsh2{:});
      case 'parafac2cp'
        [estcomp{1}{inondegenrand},dum,dum,dum,dum,dum] = feval(['nwaydecomp_' model], datpart1, incomp, specmodes, 'startval', startval1{inondegenrand}, 'compmodes', compmodes, optsh1{:}, 'ssqdatnoncp', ssqdatnoncppart1);
        [estcomp{2}{inondegenrand},dum,dum,dum,dum,dum] = feval(['nwaydecomp_' model], datpart2, incomp, specmodes, 'startval', startval2{inondegenrand}, 'compmodes', compmodes, optsh2{:}, 'ssqdatnoncp', ssqdatnoncppart2);
      case 'spacetime'
        [estcomp{1}{inondegenrand},dum,dum,dum,dum,dum] = feval(['nwaydecomp_' model], datpart1, incomp, freq, 'Dmode', Dmode, 'startval', startval1{inondegenrand}, optsh1{:});
        [estcomp{2}{inondegenrand},dum,dum,dum,dum,dum] = feval(['nwaydecomp_' model], datpart2, incomp, freq, 'Dmode', Dmode, 'startval', startval2{inondegenrand}, optsh2{:});
      case 'spacefsp'
        [estcomp{1}{inondegenrand},dum,dum,dum,dum,dum] = feval(['nwaydecomp_' model], datpart1, incomp, 'Dmode', Dmode, 'startval', startval1{inondegenrand}, optsh1{:});
        [estcomp{2}{inondegenrand},dum,dum,dum,dum,dum] = feval(['nwaydecomp_' model], datpart2, incomp, 'Dmode', Dmode, 'startval', startval2{inondegenrand}, optsh2{:});
      otherwise
        error('model not yet supported in split-half component number estimation')
    end
  end
  
  % compute splithalf coeffcients for all possible pairs of random starts from the splithalves
  partcombcompsh = cell(nondegennrand,nondegennrand);
  compsh = NaN(incomp,length(randomstat1.startvalall{1})); % NaN in case of degenflg and the below is not executed
  for irandpart1 = 1:nondegennrand
    for irandpart2 = 1:nondegennrand
      % set current estcomp
      currestcomp = cell(1,2);
      currestcomp{1} = estcomp{1}{irandpart1};
      currestcomp{2} = estcomp{2}{irandpart2};
      % compute component congruence for all possible pairs between split halves
      compcongr = zeros(incomp,incomp,length(currestcomp{1}));
      for icompsh1 = 1:incomp
        for icompsh2 = 1:incomp
          for iparam = 1:length(currestcomp{1})
            % perform model specific stuff
            switch model
              case {'parafac','parafac2','parafac2cp'}
                paramc1 = currestcomp{1}{iparam}(:,icompsh1);
                paramc2 = currestcomp{2}{iparam}(:,icompsh2);
                % normalize
                paramc1 = paramc1 ./ sqrt(sum(abs(paramc1).^2));
                paramc2 = paramc2 ./ sqrt(sum(abs(paramc2).^2));
                % put in compsh
                compcongr(icompsh1,icompsh2,iparam) = abs(paramc1' * paramc2);
              case 'spacetime'
                switch iparam
                  case {1,2,3}
                    paramc1 = currestcomp{1}{iparam}(:,icompsh1);
                    paramc2 = currestcomp{2}{iparam}(:,icompsh2);
                    % normalize
                    paramc1 = paramc1 ./ sqrt(sum(abs(paramc1).^2));
                    paramc2 = paramc2 ./ sqrt(sum(abs(paramc2).^2));
                    % put in compsh
                    compcongr(icompsh1,icompsh2,iparam) = abs(paramc1' * paramc2);
                  case 4
                    % create frequency specific phases weighted by spatial maps and frequency profiles
                    A1 = currestcomp{1}{1}(:,icompsh1);
                    A2 = currestcomp{2}{1}(:,icompsh2);
                    B1 = currestcomp{1}{2}(:,icompsh1);
                    B2 = currestcomp{2}{2}(:,icompsh2);
                    S1 = currestcomp{1}{4}(:,icompsh1);
                    S2 = currestcomp{2}{4}(:,icompsh2);
                    % construct complex site by freq matrix
                    Scomp1 = exp(1i*2*pi*repmat(freq(:).',[size(A1,1) 1]).*repmat(S1,[1 size(B1,1)]));
                    Scomp2 = exp(1i*2*pi*repmat(freq(:).',[size(A2,1) 1]).*repmat(S2,[1 size(B2,1)]));
                    % scale with A
                    Scomp1 = Scomp1 .* repmat(A1,[1 size(B1,1)]);
                    Scomp2 = Scomp2 .* repmat(A2,[1 size(B2,1)]);
                    % compute splithalfcoef over freqs, than abs, then average weighted with B
                    shoverfreq = zeros(numel(B1),1);
                    for ifreq = 1:numel(B1)
                      currS1 = Scomp1(:,ifreq);
                      currS2 = Scomp2(:,ifreq);
                      currS1 = currS1 ./ sqrt(sum(abs(currS1).^2)); % not necessary now, but just in case we ever decide to not-normalize A
                      currS2 = currS2 ./ sqrt(sum(abs(currS2).^2));
                      shoverfreq(ifreq) = abs(currS1'*currS2);
                    end
                    shsumfreq = sum(shoverfreq .* (B1.*B2)) ./ sum(B1.*B2);
                    % put in compsh
                    compcongr(icompsh1,icompsh2,iparam) = shsumfreq;
                  case 5
                    switch Dmode
                      case 'identity'
                        % D is fixed with arbitrary order, make its splithalf coefficient irrelevant
                        compcongr(icompsh1,icompsh2,iparam) = 1;
                      case 'kdepcomplex'
                        % scale with B
                        B1 = currestcomp{1}{2}(:,icompsh1);
                        B2 = currestcomp{2}{2}(:,icompsh2);
                        D1 = currestcomp{1}{5}(:,:,icompsh1);
                        D2 = currestcomp{2}{5}(:,:,icompsh2);
                        D1 = D1 .* repmat(B1(:),[1 size(D1,2)]);
                        D2 = D2 .* repmat(B2(:),[1 size(D2,2)]);
                        % vectorize
                        paramc1 = D1(:);
                        paramc2 = D2(:);
                        % normalize
                        paramc1 = paramc1 ./ sqrt(sum(abs(paramc1).^2));
                        paramc2 = paramc2 ./ sqrt(sum(abs(paramc2).^2));
                        % put in compsh
                        compcongr(icompsh1,icompsh2,iparam) = abs(paramc1' * paramc2);
                    end
                end
              case'spacefsp'
                switch iparam
                  case {1,2,3}
                    paramc1 = currestcomp{1}{iparam}(:,icompsh1);
                    paramc2 = currestcomp{2}{iparam}(:,icompsh2);
                    % normalize
                    paramc1 = paramc1 ./ sqrt(sum(abs(paramc1).^2));
                    paramc2 = paramc2 ./ sqrt(sum(abs(paramc2).^2));
                    % put in compsh
                    compcongr(icompsh1,icompsh2,iparam) = abs(paramc1' * paramc2);
                  case 4
                    % create frequency specific phases weighted by spatial maps and frequency profiles
                    A1 = currestcomp{1}{1}(:,icompsh1);
                    A2 = currestcomp{2}{1}(:,icompsh2);
                    B1 = currestcomp{1}{2}(:,icompsh1);
                    B2 = currestcomp{2}{2}(:,icompsh2);
                    L1 = currestcomp{1}{4}(:,:,icompsh1);
                    L2 = currestcomp{2}{4}(:,:,icompsh2);
                    % construct complex site by freq matrix
                    Lcomp1 = exp(1i*2*pi*L1);
                    Lcomp2 = exp(1i*2*pi*L2);
                    % scale with A
                    Lcomp1 = Lcomp1 .* repmat(A1,[1 size(B1,1)]);
                    Lcomp2 = Lcomp2 .* repmat(A2,[1 size(B2,1)]);
                    % compute splithalfcoef over freqs, than abs, then average weighted with B
                    shoverfreq = zeros(numel(B1),1);
                    for ifreq = 1:numel(B1)
                      currL1 = Lcomp1(:,ifreq);
                      currL2 = Lcomp2(:,ifreq);
                      currL1 = currL1 ./ sqrt(sum(abs(currL1).^2)); % not necessary now, but just in case we ever decide to not-normalize A
                      currL2 = currL2 ./ sqrt(sum(abs(currL2).^2));
                      shoverfreq(ifreq) = abs(currL1'*currL2);
                    end
                    shsumfreq = sum(shoverfreq .* (B1.*B2)) ./ sum(B1.*B2);
                    % put in compsh
                    compcongr(icompsh1,icompsh2,iparam) = shsumfreq;
                  case 5
                    switch Dmode
                      case 'identity'
                        % D is fixed with arbitrary order, make its splithalf coefficient irrelevant
                        compcongr(icompsh1,icompsh2,iparam) = 1;
                      case 'kdepcomplex'
                        % scale with B
                        B1 = currestcomp{1}{2}(:,icompsh1);
                        B2 = currestcomp{2}{2}(:,icompsh2);
                        D1 = currestcomp{1}{5}(:,:,icompsh1);
                        D2 = currestcomp{2}{5}(:,:,icompsh2);
                        D1 = D1 .* repmat(B1(:),[1 size(D1,2)]);
                        D2 = D2 .* repmat(B2(:),[1 size(D2,2)]);
                        % vectorize
                        paramc1 = D1(:);
                        paramc2 = D2(:);
                        % normalize
                        paramc1 = paramc1 ./ sqrt(sum(abs(paramc1).^2));
                        paramc2 = paramc2 ./ sqrt(sum(abs(paramc2).^2));
                        % put in compsh
                        compcongr(icompsh1,icompsh2,iparam) = abs(paramc1' * paramc2);
                    end
                end
            end
          end
        end
      end
      % get splithalf coefficients by selecting most-similair unique pairings, but only for those that matter for the splithalf coeff
      % 'clean' compcongr of estshcritval==0
      compcongr(:,:,estshcritval==0) = NaN;
      compsh   = zeros(incomp,length(currestcomp{1}));
      congrsum = nansum(compcongr,3);
      % match from perspective of first splithalf (i.e. find components of splithalf 2 that match those of splithalf 1)
      % do so by starting from the component-pair with the highest similarity, then the next most similar, etc.
      sh1ind   = zeros(1,incomp);
      sh2ind   = zeros(1,incomp);
      for icomp = 1:incomp
        [dum, sh1ind(icomp)] = max(max(congrsum,[],2));
        [dum, sh2ind(icomp)] = max(congrsum(sh1ind(icomp),:));
        congrsum(sh1ind(icomp),:) = 0;
        congrsum(:,sh2ind(icomp)) = 0;
      end
      % sanity check
      if any(diff(sort(sh1ind))==0) || any(diff(sort(sh2ind))==0)
        error('some components were selected multiple times')
      end
      % sort for convenience
      [sh1ind, sortind] = sort(sh1ind);
      sh2ind = sh2ind(sortind);
      for iparam = 1:length(currestcomp{1})
        compsh(:,iparam) = diag(compcongr(sh1ind,sh2ind,iparam));
      end
      % save compsh
      partcombcompsh{irandpart1,irandpart2} = compsh;
    end % irandpart2
  end % irandpart1
  
 
  % check whether any of the combinations of random starts of the partitions pass the splithalf criterion
  % do a splithalf criterion check
  partcombpass = false(size(partcombcompsh));
  for irandpart1 = 1:nondegennrand
    for irandpart2 = 1:nondegennrand
      currcompsh = partcombcompsh{irandpart1,irandpart2};
      pass = true;
      for icomp = 1:incomp
        for iparam = 1:length(estcomp{1}{1})
          if currcompsh(icomp,iparam) < estshcritval(iparam)
            pass = false;
          end
        end
      end
      if pass
        partcombpass(irandpart1,irandpart2) = true;
      end
    end % irandpart2
  end % irandpart1
  
  % determine failure or succes of current incomp
  if ~degenflg
    critfailflg = 0;
    if ~any(partcombpass(:))
      critfailflg = 1;
    end
    
    % pick best possible compsh to pass on, determine 'best' by highest minimal value
    maxminshcoeff = cellfun(@min,cellfun(@min,partcombcompsh,'uniformoutput',0));
    [dum maxind] = max(maxminshcoeff(:));
    [rowind,colind] = ind2sub([nondegennrand nondegennrand],maxind);
    compsh = partcombcompsh{rowind,colind};
    % save which is 'best' by setting it to 2 in partcombpass
    partcombpass(rowind,colind) = 2;
  else
    critfailflg = false; % most accurate given degenflg is false
  end
  
  % save incomp specifics
  allcomp{incomp}{1}        = estcomp{1};
  allcomp{incomp}{2}        = estcomp{2};
  allpartcombcompsh{incomp} = partcombcompsh;
  allcompsh{incomp}         = compsh;
  allrandomstat{incomp}{1}  = randomstat1;
  allrandomstat{incomp}{2}  = randomstat2;
    
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% determine incomp succes 
  % Act on critfailflg and degenflg
  disp(['split-half: lowest relevant absolute split-half coefficient was ' num2str(min(min(compsh(:,estshcritval~=0))))]);
  if critfailflg || degenflg
    % When current incomp fails, decrease incomp. 
    % Then, incomp-1 has either been performed or not 
    % If not performed, succes=false and continue. Then, If incomp==0 succes='true'. 
    % If it has been performed, it can only be successful, succes=true. (error check for unsuccesful)
    
    % update fail fields
    if ~degenflg
      disp(['split-half: several components did not reach split-half criterion of ' num2str(estshcritval)]);
      compshfail = compsh;
      randomstat1fail = randomstat1;
      randomstat2fail = randomstat2;
      failreason = 'split-half criterion';
     elseif degenflg
      disp('split-half: random initializations for both partitions only returned likely degenerate solutions')
      compshfail = compsh;
      randomstat1fail = randomstat1;
      randomstat2fail = randomstat2;
      failreason = 'degeneracy';
    end
    ncompsucc{incomp} = false;
    % decrease incomp by 1
    incomp = incomp-1; % the code below requires this step to be one (succ fields are always at the highest ncomp)
    
    % check for incomp == 0
    if incomp == 0
      warning('split-half: NO COMPONENTS REACHED SPLIT-HALF CRITERION');
      % set 'succes' status and final number of components
      succes = true;
      ncomp  = 1;
      compshsucc = [];
      randomstat1succ = [];
      randomstat2succ = [];
      failreason = ['split-half criterion fail at ncomp = 1'];
      break
    end
    
    % check for new incomp already performed and stop accordingly
    if ~isempty(ncompsucc{incomp})
      if ncompsucc{incomp}
        % set succes status and final number of components
        succes = true;
        ncomp  = incomp;
      elseif ~ncompsucc{incomp}
        % this should not be possible
        error('unexpected error in ncomp estimation')
      end
    end
    
  else
    % When current incomp succeeds, increase incomp. 
    % Then, incomp+i can either be at the maximum or not.
    % If at the maximum, succes = true.
    % If not at the maximum, increment.
    % Then, there have either been attempts at incomp+i or there have not.
    % If not, increment incomp with stepsize limited by the maximum.
    % If there have been incomp+i's, they can only have failed, and none should be present at incomp+stepsize+i
    % Check for this. 
    % Then, find the closest fail. If it is incomp+1, succes = true. If not, increment up until max.
    
    % update succes fields and ncompsucc
    compshsucc = compsh;
    randomstat1succ = randomstat1;
    randomstat2succ = randomstat2;
    ncompsucc{incomp} = true;
    
    % check whether maximum has been reached, and increment incomp intelligently otherwise
    if incomp==estnum(2)
      disp(['split-half: succesfully reached a priori determined maximum of ' num2str(estnum(2)) ' components']);
      compshfail = [];
      randomstat1fail = [];
      randomstat2fail = [];
      failreason = ['split-half stopped: reached a priori maximum of ' num2str(estnum(2)) ' components'];
      % set succes status
      succes = true;      
      ncomp  = incomp;
      
    else % keep on incrementing
      % check for previous fails at incomp+
      if isempty(ncompsucc(incomp+1:end))
        % no previous incomp+ solutions found, increment incomp with step (and check for ncompestend)
        incomp = min(incomp + estnum(3),estnum(2));
        
      else
        % sanity check
        if any([ncompsucc{incomp+1:end}])
          % incomp+i should never be successful
          error('unexpected error in ncomp estimation')
        end
        % incomp+i found, check whether next ncomp was failing
        if ncompsucc{incomp+1}==0
          % next ncomp failed, set succes status for current
          succes = true;
          ncomp  = incomp;
        else
          % next incomp was not the fail, find closest after that (depends on estnum(3))
          found = false;
          while ~found
            if isempty(ncompsucc{incomp+1})
              incomp = incomp+1;
            else
              found = true;
            end
          end
          % correct for ncompestend (because of check for ncompestend above, the correct incomp will never have been tested)
          incomp = min(incomp,estnum(3));
        end
      end
    end
    
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%
  
end % incomp
disp(['split-half: final number of components = ' num2str(ncomp)]);

   
  
%
% FIXME: NO MODEL SPECIFIC COEFFICIENTS HERE YET!!!!
% FIXME: the below piece of code is shit and should be reimplemented to handle flexible non-linearly increasing not-starting-from-1 ncomp's 
% Calcute cross-ncomp congruence
% the organization is as follows:
% crossncomp{i} = all cross-comp calculations started from ncomp = i
% crossncomp{i}{j} = cross-comp calculations of ncomp = i, with ncomp = j
% crossncomp{i}{j} = I*J matrix with congruence coefficents between all components
% nparam      = numel(estcomp{1});
% ncrossncomp = length(allcomp);
% ncompindex  = estnum(1):ncrossncomp; % index of all cross ncomp calculations
% crossncomp  = cell(1,ncrossncomp);
% % loop over number of comps in allcomp
% for incomp = ncompindex
%   remcomp = ncompindex;
%   remcomp(incomp) = []; % remaining ncomps to calculate over
%   % loop over cross-ncomp-calculations
%   for incross = 1:length(remcomp)
%     currncross = remcomp(incross);
%     % set n's and preset crossn
%     nseedcomp = incomp;
%     ntargcomp = currncross;
%     crossn = zeros(2,nparam,nseedcomp,ntargcomp);
%     % loop over parts
%     for ipart = 1:2;
%       currseed = allcomp{incomp}{ipart};
%       currtarg = allcomp{currncross}{ipart};
%       % loop over parameters
%       for iparam = 1:nparam
%         currseedparam = currseed{iparam};
%         currtargparam = currtarg{iparam};
%         % loop over components of seed
%         for icompseed = 1:nseedcomp
%           if ndims(currseedparam)==3 || (ndims(currseedparam)==2 && size(currseedparam,2)~=nseedcomp)
%             currseedcomp = currseedparam(:,:,icompseed);
%             currseedcomp = reshape(permute(currseedcomp,[1 3 2]),[size(currseedcomp,1)*size(currseedcomp,2) size(currseedcomp,3)]);
%           else
%             currseedcomp = currseedparam(:,icompseed);
%           end
%           % loop over components of target
%           for icomptarg = 1:ntargcomp
%             if ndims(currtargparam)==3 || (ndims(currtargparam)==2 && size(currtargparam,2)~=ntargcomp)
%               currtargcomp = currtargparam(:,:,icomptarg);
%               currtargcomp = reshape(permute(currtargcomp,[1 3 2]),[size(currtargcomp,1)*size(currtargcomp,2) size(currtargcomp,3)]);
%             else
%               currtargcomp = currtargparam(:,icomptarg);
%             end
%             if (strncmp(model,'parafac2',8) && specmodes(iparam)==2) || ~all(size(currseedcomp)==size(currtargcomp))
%               crossn(ipart,iparam,icompseed,icomptarg) = 0;
%             else
%               currseedcomp = currseedcomp ./ sqrt(sum(abs(currseedcomp).^2));
%               currtargcomp = currtargcomp ./ sqrt(sum(abs(currtargcomp).^2));
%               crossn(ipart,iparam,icompseed,icomptarg) = abs(currseedcomp' * currtargcomp);
%             end
%           end % end loop over target components
%         end  % end loop over seed components
%       end % end loop over parameters
%     end % loop over parts
%     % mean over parameters and mean over partitions
%     crossn = squeeze(mean(mean(crossn,2),1));
%     crossncomp{incomp}{incross} = crossn;
%   end % end loop over cross-ncomp-calculations
% end % end loop over number of comps in allcomp


% create splithalfstat
splithalfstat.ncomp             = ncomp;
splithalfstat.splithcsucc       = compshsucc;
splithalfstat.splithcfail       = compshfail;
splithalfstat.crosscompcongr    = [];
splithalfstat.randomstat1succ   = randomstat1succ;
splithalfstat.randomstat2succ   = randomstat2succ;
splithalfstat.randomstat1fail   = randomstat1fail;
splithalfstat.randomstat2fail   = randomstat2fail;
splithalfstat.failreason        = failreason;
splithalfstat.allcomp           = allcomp;
splithalfstat.allcompsh         = allcompsh;
splithalfstat.allpartcombcompsh = allpartcombcompsh;
splithalfstat.allrandomstat     = allrandomstat;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Subfunction for calculating randomstart-values         %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [startval, randomstat] = randomstart(model, dat, ncomp, nrand, nitt, convcrit, degencrit, distcomp, dispprefix, varargin)

% get model specific options from keyval
switch model
  case 'parafac'
    compmodes = keyval('compmodes', varargin);
  case 'parafac2'
    compmodes = keyval('compmodes', varargin);
    specmodes = keyval('specmodes', varargin);
  case 'parafac2cp'
    compmodes   = keyval('compmodes',   varargin);
    specmodes   = keyval('specmodes',   varargin);
    ssqdatnoncp = keyval('ssqdatnoncp', varargin);
  case 'spacetime'
    freq  = keyval('freq', varargin);
    Dmode = keyval('Dmode', varargin);
  case 'spacefsp'
    Dmode = keyval('Dmode', varargin);
  otherwise
    error('model not supported')
end


% start random decompositions
nitt = round(nitt); % just to make sure it's an integer
if ~isempty(distcomp.system)
  disp([dispprefix 'random start: determining optimal starting values from random initialization']);
  disp([dispprefix 'random start: decomposition of ' num2str(nrand) ' random initializations started using ' distcomp.system]);
  
  % prepare cell arrays for input
  cellncomp         = repmat({ncomp},[nrand 1]);
  cellnittkey       = repmat({'nitt'},[nrand 1]);
  cellnittval       = repmat({nitt},[nrand 1]);
  cellconvcritkey   = repmat({'convcrit'},[nrand 1]);
  cellconvcritval   = repmat({convcrit},[nrand 1]);
  celldispprefixkey = repmat({'dispprefix'},[nrand 1]);
  celldispprefixval = repmat({[dispprefix 'random start: ']},[nrand 1]);
  % create a cell-array containing copys of the data, or save and pass temporary filename 
  % implemented for all models (file is deleted below)
  if isnumeric(dat) && ~isempty(distcomp.inputpathprefix) && ischar(distcomp.inputpathprefix)
    % make a random file name
    rng(sum(clock.*1e6))
    randname = tempname;
    filename = [distcomp.inputpathprefix randname(end-6:end) '.mat'];
    % put in cell-array input
    celldat  = repmat({filename},[nrand 1]);
    % save data
    save(filename,'dat','-v7.3')
  else
    celldat = repmat({dat},[nrand 1]);
  end
  
  % set up general options
  opt = {cellnittkey, cellnittval, cellconvcritkey, cellconvcritval, celldispprefixkey, celldispprefixval};
  if isempty(distcomp.memreq)
    if ~isnumeric(dat)
      s = whos('-file', dat);
    else
      s = whos('dat');
    end
    memreq = s.bytes * 4; % probably enough for the fourier algorithms, probably not for the parafac ones
  else
    memreq = distcomp.memreq;
  end
  
  % set up distributed computing specific options
  switch distcomp.system
    case 'p2p'
      distcompopt = {'uniformoutput','false','resubmitdelay',distcomp.p2presubdel,'timreq', distcomp.timreq};
      distcompfun = @peercellfun;
    case 'torque'
      % increment timreq (should really be done based on size of data)
      distcomp.timreq = distcomp.timreq * ceil(ncomp/10);
      if distcomp.timreq > (60*60*24*2.9)
        batchqueue = 'batchext';
      else
        batchqueue = 'batch';
      end
      distcompopt = {'backend','torque','queue',batchqueue,'timreq', distcomp.timreq,'matlabcmd','/opt/matlab-R2012b/bin/matlab','options','-V'};
      distcompfun = @qsubcellfun;
    otherwise
      error('distributed computing system not supported')
  end
  
  % state current time as que submission time
  disp(['submitting to queue on: ' datestr(now)])
  
  % distribute function calls to peers
  switch model
    case 'parafac'
      cellcompmodeskey  = repmat({'compmodes'},[nrand 1]);
      cellcompmodesval  = repmat({compmodes},[nrand 1]);
      [cellcomp,cellssqres,cellexpvar,cellscaling,celltuckcongr] = feval(distcompfun,['nwaydecomp_' model ],celldat,cellncomp,cellcompmodeskey,cellcompmodesval,opt{:},distcompopt{:},'memreq', memreq);
    case 'parafac2'
      cellcompmodeskey  = repmat({'compmodes'},[nrand 1]);
      cellcompmodesval  = repmat({compmodes},[nrand 1]);
      cellspecmodes     = repmat({specmodes},[nrand 1]);
      [cellcomp,dum,cellssqres,cellexpvar,cellscaling,celltuckcongr] = feval(distcompfun,['nwaydecomp_' model ],celldat,cellncomp,cellspecmodes,cellcompmodeskey,cellcompmodesval,opt{:},distcompopt{:},'memreq', memreq);
    case 'parafac2cp'
      cellcompmodeskey   = repmat({'compmodes'},[nrand 1]);
      cellcompmodesval   = repmat({compmodes},[nrand 1]);
      cellspecmodes      = repmat({specmodes},[nrand 1]);
      cellssqdatnoncpkey = repmat({'ssqdatnoncp'},[nrand 1]);
      cellssqdatnoncpval = repmat({ssqdatnoncp},[nrand 1]);
      [cellcomp,dum,cellssqres,cellexpvar,cellscaling,celltuckcongr] = feval(distcompfun,['nwaydecomp_' model ],celldat,cellncomp,cellspecmodes,cellcompmodeskey,cellcompmodesval,cellssqdatnoncpkey,cellssqdatnoncpval,opt{:},distcompopt{:},'memreq', memreq);
    case 'spacetime'
      cellfreq = repmat({freq},[nrand 1]);
      cellDmodekey  = repmat({'Dmode'},[nrand 1]);
      cellDmodeval  = repmat({Dmode},[nrand 1]);
      [cellcomp,dum,cellssqres,cellexpvar,cellscaling,celltuckcongr] = feval(distcompfun,['nwaydecomp_' model ],celldat,cellncomp,cellfreq,cellDmodekey,cellDmodeval,opt{:},distcompopt{:},'memreq', memreq);
    case 'spacefsp'
      cellDmodekey  = repmat({'Dmode'},[nrand 1]);
      cellDmodeval  = repmat({Dmode},[nrand 1]);
      [cellcomp,dum,cellssqres,cellexpvar,cellscaling,celltuckcongr] = feval(distcompfun,['nwaydecomp_' model ],celldat,cellncomp,cellDmodekey,cellDmodeval,opt{:},distcompopt{:},'memreq', memreq);
    otherwise
      error('model not yet supported in automatic random starting')
  end
  % delete temporary copy of data if it was created
  if ~isempty(distcomp.inputpathprefix) && ischar(distcomp.inputpathprefix) && strcmp(distcomp.system,'p2p')
    delete(filename)
  end
  
  % format output to ssqres with output below
  randssqres    = [cellssqres{:}];
  randcomp      = cellcomp(:).';
  randexpvar    = [cellexpvar{:}];
  randscaling   = cellscaling(:).';
  randtuckcongr = celltuckcongr(:).';
else 
  % allocate
  randssqres    = zeros(1,nrand);
  randcomp      = cell(1,nrand);
  randexpvar    = zeros(1,nrand);
  randscaling   = cell(1,nrand);
  randtuckcongr = cell(1,nrand);
  disp([dispprefix 'random start: determining optimal starting values from random initialization']);
  for irand = 1:nrand
    disp([dispprefix 'random start ' num2str(irand) ': decomposition of random initialization ' num2str(irand) ' of ' num2str(nrand) ' started']);
    % set up general options
    opt = {'nitt', nitt, 'convcrit', convcrit, 'dispprefix',[dispprefix 'random start ' num2str(irand) ': ']};
    switch model
      case 'parafac'
        [rndcomp,rndssqres,rndexpvar,rndscaling,rndtuckcongr] = feval(['nwaydecomp_' model], dat, ncomp, 'compmodes', compmodes, opt{:});
      case 'parafac2'
        [rndcomp,dum,rndssqres,rndexpvar,rndscaling,rndtuckcongr] = feval(['nwaydecomp_' model], dat, ncomp, specmodes, 'compmodes', compmodes, opt{:});
      case 'parafac2cp'
        [rndcomp,dum,rndssqres,rndexpvar,rndscaling,rndtuckcongr] = feval(['nwaydecomp_' model], dat, ncomp, specmodes, 'compmodes', compmodes, opt{:}, 'ssqdatnoncp', ssqdatnoncp);
      case 'spacetime'
        [rndcomp,dum,rndssqres,rndexpvar,rndscaling,rndtuckcongr] = feval(['nwaydecomp_' model], dat, ncomp, freq, 'Dmode', Dmode, opt{:});
      case 'spacefsp'
        [rndcomp,dum,rndssqres,rndexpvar,rndscaling,rndtuckcongr] = feval(['nwaydecomp_' model], dat, ncomp, 'Dmode', Dmode, opt{:});
      otherwise
        error('model not yet supported in automatic random starting')
    end
    randssqres(irand)    = rndssqres;
    randcomp{irand}      = rndcomp;
    randexpvar(irand)    = rndexpvar;
    randscaling{irand}   = rndscaling;
    randtuckcongr{irand} = rndtuckcongr;
    disp([dispprefix 'random start ' num2str(irand) ': decomposition of random initialization ' num2str(irand) ' of ' num2str(nrand) ' finished: error-term = ' num2str(rndssqres)]);
  end
end

% Create randomstat structure and set best possible start-values
% sort output by ssqres
[randssqres,sortorder] = sort(randssqres);
randexpvar          = randexpvar(sortorder);
randscaling         = randscaling(sortorder);
randcomp            = randcomp(sortorder);
randtuckcongr       = randtuckcongr(sortorder);


% find possible degenerate solutions and create degeneracy index
if ncomp==1 % if ncomp is 1, there can be no degeneracy
  maxtg     = zeros(1,nrand);
  degeninit = [];
else
  maxtg     = cellfun(@max,(randtuckcongr));
  degeninit = find(maxtg >= degencrit);
end
% create index of non-degenerate initializations
if ~(length(degeninit)==nrand)
  initindex = 1:nrand;
  initindex(degeninit) = [];
else
  initindex = [];
end

%
% FIXME: NO MODEL SPECIFIC COEFFICIENTS HERE YET!!!!
%
% create congruence coeficient over ALL initializations (including possible degenerate ones)
% calculate per component per param the abs(inner product)
nparam = numel(randcomp{1});
congrall = NaN(ncomp,nparam);
for icomp = 1:ncomp
  for iparam = 1:nparam
    try % FIX the above, until then, it is more important the code runs through
      randmat = [];
      for irand = 1:nrand
        tmpcomp = randcomp{irand}{iparam};
        if ndims(tmpcomp)==3 || (ndims(tmpcomp)==2 && size(tmpcomp,2)~=ncomp)
          tmpcomp = tmpcomp(:,:,icomp);
          tmpcomp = reshape(permute(tmpcomp,[1 3 2]),[size(tmpcomp,1)*size(tmpcomp,2) size(tmpcomp,3)]);
        else
          tmpcomp = tmpcomp(:,icomp);
        end
        tmpcomp = tmpcomp ./ sqrt(sum(abs(tmpcomp).^2));
        randmat(irand,:) = tmpcomp;
      end
      % take the inner-product, because of cauchy-schartz this goes from 0 -> 1 (norms are already 1)
      congr = randmat * randmat';
      % remove diagonal and put mean abs in congruence
      congr(eye(size(congr))==1) = [];
      congrall(icomp,iparam) = mean(abs(congr));
    end
  end
end

%
% FIXME: NO MODEL SPECIFIC COEFFICIENTS HERE YET!!!!
%
% create congruence coeficient over SELECTION of initializations, disregarding those that might be degenerate and those not thought to be at the global minimum
% select which initilizations to calculate congruence over
if ~isempty(initindex)
  nanexpvar = randexpvar; % create a randexpvar vector with NaNs for degenerate initializations
  nanexpvar(degeninit) = NaN;
  expvarchange = abs(nanexpvar - nanexpvar(initindex(1)));
  randindex    = find((expvarchange < .1) & (maxtg < degencrit)); % criterion means initializations are included if they differ less than 0.1% in expvar to the first (sorted) non-degenerate expvar
  % calculate per component per param the abs(inner product) (if only degenerate solutions are found than matrix will contain only NaNs)
  congrglobmin = NaN(ncomp,nparam);
  for icomp = 1:ncomp
    for iparam = 1:nparam
      try % FIXME fix the above, until then, it is more important the code runs through
        %randmat = zeros(length(randindex),size(randcomp{1}{iparam},1));
        randmat = [];
        for irand = 1:length(randindex)
          tmpcomp = randcomp{randindex(irand)}{iparam};
          if ndims(tmpcomp)==3 || (ndims(tmpcomp)==2 && size(tmpcomp,2)~=ncomp)
            tmpcomp = tmpcomp(:,:,icomp);
            tmpcomp = reshape(permute(tmpcomp,[1 3 2]),[size(tmpcomp,1)*size(tmpcomp,2) size(tmpcomp,3)]);
          else
            tmpcomp = tmpcomp(:,icomp);
          end
          tmpcomp = tmpcomp ./ sqrt(sum(abs(tmpcomp).^2));
          randmat(irand,:) = tmpcomp;
        end
        % take the inner-product, because of cauchy-schwartz this goes from 0 -> 1 (norms are already 1)
        congr = randmat * randmat';
        % remove diagonal and put mean abs in congruence
        congr(eye(size(congr))==1) = [];
        congrglobmin(icomp,iparam) = mean(abs(congr));
      end
    end
  end
else
  randindex = [];
  congrglobmin = [];
end


% put together randomstat
randomstat.expvar         = randexpvar;
randomstat.error          = randssqres;
randomstat.congrall       = congrall;
randomstat.globmininit    = randindex;
randomstat.congrglobmin   = congrglobmin;
randomstat.tuckcongr      = randtuckcongr;
randomstat.degeninit      = degeninit;
randomstat.scaling        = randscaling;
% add settings for randstart
randomstat.nrand          = nrand;
randomstat.convcrit       = convcrit;
randomstat.nitt           = nitt;



% put the scaling coefficients back in to make the startvalues proper startvalues
for irand = 1:nrand
  currrndcomp = randcomp{irand};
  currrndscal = randscaling{irand};
  
  switch model
    
    case {'parafac','parafac2','parafac2cp'}
      % magnitude scaling
      for icomp = 1:ncomp
        % set mode1
        mode1     = currrndcomp{1}(:,icomp);
        mode1norm = currrndscal{1}(icomp);
        mode1     = mode1 .* mode1norm;
        currrndcomp{1}(:,icomp) = mode1;
      end
      % phase scaling
      compmodesindex = find(compmodes==1);
      if sum(compmodes)~=0 % only do this if there is a complex mode
        for icomp = 1:ncomp
          % set mode1
          mode1      = currrndcomp{compmodesindex(1)}(:,icomp);
          phaseshift =  currrndscal{2}(icomp);
          mode1      = mode1 ./ phaseshift;
          currrndcomp{compmodesindex(1)}(:,icomp) = mode1;
        end
      end
      
    case {'spacetime','spacefsp'}
      % magnitude scaling
      for icomp = 1:ncomp
        % set mode1
        param3     = currrndcomp{3}(:,icomp);
        param3norm = currrndscal(icomp);
        param3     = param3 .* param3norm;
        currrndcomp{3}(:,icomp) = param3;
      end
      
    otherwise
      error('unsupported model')
  end
  
  % put back in randcomp
  randcomp{irand} = currrndcomp;
end

% get best possible solution and display number of expected degenerate solutions
disp([dispprefix 'random start: ' num2str(length(degeninit)) ' out of ' num2str(nrand) ' initializations expected to be degenerate'])
if ~isempty(randindex)
  startval = randcomp{randindex(1)};
  disp([dispprefix 'random start: lowest error-term from procedure = ' num2str(randssqres(randindex(1)))])
else % if only degenerate solutions are found, than better pick the "best" one
  startval = randcomp{1};
  disp([dispprefix 'random start: lowest error-term from procedure = ' num2str(randssqres(1))])
  disp([dispprefix 'random start: warning: best random initialization might be degenerate'])
end

% save startvalues
randomstat.startvalglobmin = startval;
randomstat.startvalall     = randcomp;
























