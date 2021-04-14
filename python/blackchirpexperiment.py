# -*- coding: utf-8 -*-
"""
BlackChirp Python Module
Author: Kyle N. Crabtree
"""

import struct
import numpy
import scipy.signal as spsig
import shelve
import os
import bcfitting as bcfit
import concurrent.futures
import sys
import scipy.optimize as spopt
import tabulate

class BlackChirpExperiment:
    """
    Convenience class for reading/processing data from blackchirp
    
    The constructor for the class loads in data for a particular experiment
    number. After initialization, a number of methods are available for
    conveniently performing common operations.
    
    In addition, certain class attributes are initialized from persistent
    storage when the module is loaded. These include default parameters
    for FT routines. All BlackChirpExperiment objects use the same default
    values, but the values for a particular experiment can be overridden
    by directly setting the attributes on that instance.
    
    There are static class methods available for setting the class values
    and for loading the class values. The load_settings static class method
    is automatically called when this module is loaded.
    
    Public attributes:
        
        d_number -- experiment number
        
        d_header_values -- dictionary containing strings of header data
        
        d_header_units -- dictionary containing units for header values
        
        fid_list -- list of BlackChirpFid objects
        
        time_data -- dictionary of time data
        
        quiet -- suppress output messages (default false)
        
    Public Methods:
    
        ft_one -- Perform FT of a single FID
        
        find_peaks -- Perform peak finding operation on arrays
        
    Class attributes:
    
        _fid_start -- Starting point number (if int) or time (in us) for
        FID. All points before this one are set to 0 prior to FT. Default is
        0.
        
        _fid_end -- As _fid_start, but points after are set to 0. If negative
        (or less than _fid_start), all points after _fid_start are kept.
        Default is -1.
        
        _zpf -- Next power of 2 to zero pad FID length to prior to FT.
        Default is 0.
        
        _ft_min -- If positive, frequencies below this (in MHz) are set to 0
        in FT. Default is -1.0.
        
        _ft_max -- If positive, frequencies above this (in MHz) are set to 0
        in FT. Default is -1.0.
        
        _ft_winf -- Window function applied to FID before FT. This must
        be a value understood by scipy.signal.get_window. Default is
        'boxcar', which is equivalent to applying no window at all.
        
    Class Methods:
    
        set_ft_defaults -- Set and store default attributes related to FT
        calculations
        
        load_settings -- Retrieve class attributes from persistent storage
        
        print_settings -- Print out class attributes to standard output
    """
    
    _settings = {
    'fid_start' : 0,
    'fid_end' : -1,
    'ft_zpf' : 0,
    'ft_min' : -1.0,
    'ft_max' : -1.0,
    'ft_winf' : 'boxcar',
    'pf_snr' : 3.0,
    'pf_chunks' : 50,
    'pf_medf' : 3.0,
    'pf_sgorder' : 6,
    'pf_sgwin' : 11,
    'an_win' : 50,
    'an_wf' : 3.0,
    'an_model' : bcfit.bc_multigauss
    }
    
    @classmethod
    def set_ft_defaults(cls,**kwargs):
        """
        Set and store default values for FT
        
        This method stores default arguments for the FT routines in
        persistent, class-scope storage. Class attributes are set for use
        by all instances of BlackChirpExperiment, and the shelve module
        is used to write the values to a file in the user's home directory
        (~/.config/CrabtreeLab/blackchirp-python)
        
        Arguments (all optional; values only set for specified args):
        
            fid_start -- All points before this index (integer) or time
            (float, us) in the FID are set to 0 before FT
            
            fid_end -- All points after this index (integer) or time
            (float, us) in the FID are set to 0 before FT
            
            zpf -- Next power of 2 by which to expand FID length prior to FT
            
            ft_min -- All FT values below this frequency (MHz) are set to 0
            
            ft_max -- All FT values above this frequency (MHz) are set to 0
            
            ft_winf -- Window function applied to FID before FT. This must
            be a value understood by scipy.signal.get_window
            
            pf_snr -- Signal-to-noise threshold used in peak finding algorithm
            
            pf_chunks -- Number of chunks used in baseline estimation
            
            pf_medf -- Factor used in median filter for baseline estimation
            
            pf_sgorder -- Order of polynomial used in savitzky-golay filter
            in peak finding algorithm
            
            pf_sgwin -- Window size used in savitzky-golay filter
            in peak finding algorithm. Must be odd
            
            an_win -- Window size (in points) used to glob together peaks
            in fitting routine
            
            an_wf -- Width factor for fitting. Multiplied by point spacing to
            set bounds on peak position in fit function and as an initial
            guess of linewidth. Should be set to the typical peak width (or
            slightly smaller) in points.
            
            an_model -- Model function used for representing peaks
        """
        
        #store settings
        with shelve.open(os.path.expanduser('~')
                         +'/.config/CrabtreeLab/blackchirp-python') as shf:
            for key,val in kwargs.items():
                if key in cls._settings:
                    cls._settings[key]=val
                    shf[key] = val
                else:
                    print(str(key)+" is not a recognized setting.")            
        
    
    @classmethod
    def load_settings(cls):
        """
        Read default values from persistent storage.
        
        This method is called when the script is processed. It reads class
        settings from persistent storage, if applicable. This function
        should not need to be called by a script user.
        """
        try:
            with shelve.open(os.path.expanduser('~')
                             +'/.config/CrabtreeLab/blackchirp-python') as shf:
                for key,val in shf.items():
                    if key in cls._settings:
                        cls._settings[key]=val
        except (OSError, IOError):
            print("No persistent settings found; using defaults")
            pass
        
        cls.print_settings()
            
        
    @classmethod
    def print_settings(cls):
        """
        Print class settings
        """

        print("")
        print("BlackChirpExperiment Default Settings")
        
        td = []
        ttd = []
        for key, val in cls._settings.items():
            if len(ttd) == 4:
                td.append(ttd)
                ttd = []
            ttd.append(key)
            if key == 'an_model':
                ttd.append(val.__name__)
            else:
                ttd.append(str(val))
            
        td.append(ttd)
        
        print(tabulate.tabulate(td))
        
       
        print(str("To modify these values globally, use BlackChirpExperiment." +
              "set_ft_defaults()."))
        print("")
        
        
    
    def __init__(self,number,path = None,quiet = False):
        """
        Constructor.
        
        Reads data generated by blackchirp. Each experiment performed in
        blackchirp is identified by a number, and is saved by default in
        the directory /home/data/blackchirp/experiments/x/y/z/, where z is the
        experiment number, y is z//1000, and x is z//1000000. That directory
        contains many files, but most important are the header file (ascii,
        z.hdr), the FID file (binary, z.fid), the chirp file (ascii, z.chp),
        the time data file (ascii, z.tdt), and the lif file (binary, z.lif).
        Optionally, the user may specify the path to a directory containing
        these files. Not all files will be present for a given experiment.
        
        The constructor reads the header data into a pair of dictionaries:
        d_header_values and d_header_units. All data in these dicts are stored
        as strings, so may need to be convertes to other types for use.
        
        If the experiment generated FTMW data, the FIDs are loaded from the
        FID file and stored in a list called fid_list as BlackChirpFid
        objects. Time data are stored in a dictionary called time_data; the
        keys are the names of the data and the values are lists of strings.
        
        TODO: Read in LIF data and chirp data
        
        Arguments:
        
            number -- the experiment number to load (z)
        
            path -- optional, the path where the experiment files are located
            
            quiet -- optional, suppress output messages
        """        
        
        if number < 1:
            raise ValueError('Experiment number must be > 0')
            
        self.d_number = number
        
        self.quiet = quiet
        
        millions = number // 1000000
        thousands = number // 1000
        
        if path is None:
            path = "/home/data/blackchirp/experiments/" + str(millions) \
                   + "/" + str(thousands) + "/" + str(number)
            
        self.d_header_values = {}
        self.d_header_units = {}
        self.fid_list = []
        
        #read data from header file
        try:
            with open(path+"/"+str(number)+".hdr") as hdr:
                for line in hdr:
                    if not line:
                        continue
                    l = line.split('\t')
                    if not l or len(l) < 3:
                        continue
                    
                    key = l[0].strip()
                    value = l[1].strip()
                    unit = l[2].strip()
                    
                    self.d_header_values[key] = value
                    self.d_header_units[key] = unit
        except (OSError, IOError):
            #maybe this is BBAcq Data?
            with open(path+"/"+str(number)+".txt") as txt:
                thefid = []
                for line in txt:
                    if len(line) < 2:
                        continue
                    if line.startswith('#'):
                        l = line.split('\t')
                        if not l or len(l) < 3:
                            continue
                        
                        key = l[0].strip('#')
                        value = l[1].strip()
                        unit = l[2].strip()
                    
                        self.d_header_values[key] = value
                        self.d_header_units[key] = unit
                    else:
                        try:
                            thefid.append(float(line.strip()))
                        except (ValueError):
                            pass
                
            self.fid_list.append( BlackChirpFid.fromTextData(thefid,self.d_header_values))
                
        
        #find out if ftmw config is enabled
        try:
            if self.d_header_values['FtmwConfigEnabled'].lower() \
                    in ['true', 't', '1', 'yes', 'y']:
                #load FID file
                with open(path + "/" + str(number) + ".fid", 'rb') as fidfile:
                    buffer = fidfile.read(4)
                    ms_len = struct.unpack(">I", buffer)
                    buffer = fidfile.read(ms_len[0])
                    magic_string = buffer.decode('ascii')
                    if not magic_string.startswith("BCFID"):
                        raise ValueError("Could not read magic string from "
                                         + fidfile.name)
                    
                    l = magic_string.split("v")
                    if len(l) < 2:
                        raise ValueError("Could not determine version \
                                          number from magic string "
                                          + magic_string)
                    
                    version = l[1]
                    
                    buffer = fidfile.read(4)
                    fidlist_size = struct.unpack(">I", buffer)[0]
                    for i in range(0,fidlist_size):
                        self.fid_list.append(BlackChirpFid(version, fidfile))
    #        find out if motorscan config is enabled
            if self.d_header_values['MotorScanEnabled'].lower() \
                    in ['true','t','1','yes','y']:
                self.motorScan = BlackChirpMotorScan(number,path=path,headerData=self.d_header_values)
    #        if self.d_header_values['LifConfigEnabled'].lower() \
    #                in ['true', 't', '1', 'yes', 'y']
    #            with open(path+"/"+str(number)+".lif", 'rb') as liffile:
        except (KeyError):
            pass
        
        #load in time data
        self.time_data = {}
        try:
            with open(path + "/" + str(number) + ".tdt") as tdtfile:
                look_for_header = True                
                header_list = []
                for line in tdtfile:
                    if line.strip() == "":
                        continue
                    if line.startswith('#') and "PlotData" in line:
                        look_for_header = True
                        header_list = []
                        continue
                    if line.startswith('#'):
                        continue
                    
                    l = line.split('\t')
                    if len(l) < 1:
                        continue
                    
                    if look_for_header is True:
                        for i in range(0, len(l)):
                            name = ""
                            l2 = str(l[i]).split('_')
                            for j in range(0, len(l2)-1):
                                name += str(l2[j]).strip()
                            self.time_data[name] = []
                            header_list.append(name)
                        look_for_header = False
                    else:
                        for i in range(0, len(l)):
                            self.time_data[header_list[i]]. \
                            append(str(l[i]).strip())
        except (OSError, IOError):
            pass
        
        if not self.quiet:
            print("Successfully loaded experiment "+str(self.d_number))
            print("Header contains "+str(len(self.d_header_values))
                  +" entries.")
            print("Experiment contains "+str(len(self.fid_list))
                  +" FID(s) in fid_list.")
            print("Experiment contains "+str(len(self.time_data))
                  +" items in time_data.")
            print("Experiment contains Motor Scan data")
            
    def print_exp_settings(self,f=None):
        td = []
        ttd = []
        for key, val in self._settings.items():
            if f is None or key.startswith(f):
                if len(ttd) == 4:
                    td.append(ttd)
                    ttd = []
                ttd.append(key)
                if key == 'an_model':
                    ttd.append(val.__name__)
                else:
                    ttd.append(str(val))
            
        td.append(ttd)
        
        print(tabulate.tabulate(td))
        
            
    
    def ft_one(self, index = 0):
        """
        Return Fourier transform of one FID
        
        Performs the Fourier transform of a single FID. The processing
        settings (start, end, window function, zero padding, etc) are taken
        from the self._settings dictionary.
        
        Unless you moke a change, the current class default value
        is used (printed on startup). For a particular instance of the class,
        the values can also be changed by modifying the _settings dictionary
        directly (e.g., exp._settings['zpf']=2). All subsequent FT operations
        on that instance would use 2 for the zpf instead of the class default
        value.
        
        Note: the y-values of the FT are multiplied by 1e3 to bring them
        to more numerically friendly magnitudes
        
        Returns: frequency array, intensity array
        
        Arguments:
        
            index -- Index of FID in fid_list to FT (default: 0)
        """
        
        print("")
        print("Performing FT using following settings")
        self.print_exp_settings(f="f")
        
        
        #make a copy of the FID, and keep an easy ref
        f = self.fid_list[index]
        f_data = numpy.array(f.data[:])
        
        #figure out starting and ending indices
        #Per python's slice syntax, e_index is 1st point NOT included
        s_index = 0
        e_index = f.size
        
        si = self._settings['fid_start']
        ei = self._settings['fid_end']
            
        if isinstance(si, float):
            #Interpret start value in microseconds
            s_index = int(si // f.spacing)                
        else:
            #Interpret start value as an index
            s_index = si
            
        if(isinstance(ei, float)):
            e_index = min((int(ei // f.spacing), f.size))
            if e_index < s_index:
                e_index = f.size
        else:
            if e_index < s_index:
                e_index = f.size
        
        
        #apply window function to region between start and end
        #todo: catch exception if get_window fails and fall back to boxcar
        win_size = e_index - s_index
        f_data[s_index:e_index] *= spsig.get_window(self._settings['ft_winf'],
                                                        win_size)
        
        #zero out points outside (start,end) range
        if s_index > 0:
            f_data[0:s_index-1] = 0.0
        if e_index < f.size:
            f_data[e_index:] = 0.0
        
        #apply zero padding factor
        z = self._settings['ft_zpf']
            
        if z > 0:
            z = int(z)
            s = 1
            
            #the operation << 1 is a fast implementation of *2
            #starting at 1, double until s is the next bigger power of 2
            #compared to the length of the FID
            while s <= f.size:
                s = s << 1
            
            #if zpf=1, we are done. Otherwise, keep multiplying zpf-1 times
            for i in range(0,z-1):
                s = s << 1
            
            #the resize function automatically sets new elements to 0
            #f_data.resize((s),refcheck=False)
        else:
            s = f.size
        
        #compute FFT
        ft = scipy.fft.rfft(f_data,n=s)
        df = 1.0 / (f.spacing)
        out_x = probe_freq + f.sideband*scipy.fft.rfftfreq(s,df)
        out_y = numpy.absolute(ft)*1e3
        if self._settings['ft_min']>0.0:
            out_y[out_x < self._settings['ft_min']] = 0.0
        if self._settings['ft_min']>0.0:
            out_y[out_x > self._settings['ft_max']] = 0.0

        return out_x, out_y
            
                
    def find_peaks(self, xd, yd):
        """
        Return lists of peaks in yarray
        
        This function locates peaks in y array, returning lists containing
        their x values, y values, and signal-to-noise ratios.
        
        Peak finding is based on a noise estimator and 2nd derivative
        procedure. First, a noise model is constructed by breaking the
        y array into chunks (the number of chunks is determined by
        self._settings['pf_chunks']. Within each chunk, the median noise level
        is iteratively located by calculating the median, removing all points
        self._settings['pf_medf'] times greater than the median, and repeating
        until no more points are removed. After removal of these points, the
        mean and standard deviation are used to model the baseline offset and
        noise for that segment.
        
        A second derivative of the y array is calculated using a Savitzky
        Golay filtered version of yarray. The order and window size of the
        filter are set by self._settings['pf_sgorder'] and
        self._settings['pf_sgwin'] respectively. This smoothed second
        derivative is used later in the peak finding routine.
        
        To locate peaks, the program loops over y array, calculating the SNR
        at each point. If the SNR is above the threshold set by
        self._settings['pf_snr'], then the second derivative is scanned to see
        if the point is a local minimum by comparing its value to two 4-point
        windows ([i-2, i-1, i, i+1] and [i-1,i,i+1,i+2]). If either test
        returns true, then the point is flagged as a peak.
        
        Arguments:
        
            xd -- Array-like x values
            
            yd -- Array-like y values

        Returns outx, outy, outidx, outsnr, noise, baseline:
        
            outx -- Array of peak x values
            
            outy -- Array of peak y values
            
            outidx -- Array of peak indices
            
            outsnr -- Array of estimated signal-to-noise ratios
            
            noise -- Array containing 1 sigma noise estimate
            
            baseline -- Array containing baseline estimate
        
            
        """
        
        
        print("")
        print("Performing peak finding using following settings")
        self.print_exp_settings(f='pf')
        
        #first remove all 0.0 from y array
        ii = 0
        ff = len(yd-1)
        nz = False
        for i in range(len(yd)):
            if yd[i] > 1e-50:
                nz = True
                continue
            if nz:
                ff = i
                break
            else:
                ii = i
                
        yarray = yd[ii:ff]
        xarray = xd[ii:ff]
            
        win = self._settings['pf_sgwin']
        if win < 3:
            print('pf_sgwin nust be >= 3. Setting to 3.')
            win = 3
        if win%2 == 0:
            print('pf_sgwin must be odd. Setting to '+str(win+1))
            win = win+1
        
        order = self._settings['pf_sgorder']
        d2y = spsig.savgol_filter(yarray,win,order,deriv=2)
        
            
        #build noise model
        chunks = self._settings['pf_chunks']
        if chunks < 1:
            print("Cannot have chunks < 1. Setting to 1")
            chunks = 1
        chunk_size = len(yarray)//chunks + 1
        avg = []
        noise = []
        dat = []
        outnoise = numpy.empty(len(yarray))
        outbaseline = numpy.empty(len(yarray))
        for i in range(chunks):
            if i + 1 == chunks:
                dat = yarray[i*chunk_size:]
            else:
                dat = yarray[i*chunk_size:(i+1)*chunk_size]
 
            #Throw out any points that are 3* the median and recalculate
            #Do this until no points are removed. 
            done = False           
            while not done:
                if len(dat) == 0:
                    break
                med = numpy.median(dat)
                fltr = [d for d in dat if d < 3*med]
                if len(fltr) == len(dat):
                    done = True
                dat = fltr
            
            #now, retain the mean and stdev for later use
            if len(dat) > 2:
                avg.append(numpy.mean(dat))
                noise.append(numpy.std(dat))
            else:
                #something went wrong with noise detection... ignore section
                #probably a chunk containing only 0.0
                avg.append(0.0)
                noise.append(1.0)
            
            if i + 1 == chunks:
                outnoise[i*chunk_size:] = noise[i]
                outbaseline[i*chunk_size:] = avg[i]
            else:
                outnoise[i*chunk_size:(i+1)*chunk_size] = noise[i]
                outbaseline[i*chunk_size:(i+1)*chunk_size] = avg[i]
                
        
   
        outx = []
        outy = []
        outidx = []
        outsnr = []
        
        snr = self._settings['pf_snr']
        #if a point has SNR > threshold, look for local min in 2nd deriv.
        for i in range(2,len(yarray)-2):
            snr_i = (yarray[i] - avg[i // chunk_size])/noise[i // chunk_size]
            if snr_i >= snr:
                if (d2y[i-2] > d2y[i-1] > d2y[i] < d2y[i+1] or
                    d2y[i-1] > d2y[i] < d2y[i+1] < d2y[i+2]):
                        outx.append(xarray[i])
                        outy.append(yarray[i])
                        outidx.append(i+ii)
                        outsnr.append(snr_i)
        
        if not self.quiet:
            weak_count = 0
            med_count = 0
            strong_count = 0
            for y in outsnr:
                if y < 10:
                    weak_count += 1
                elif y < 100:
                    med_count += 1
                else:
                    strong_count += 1
            
            print("Toal peaks located : "+str(len(outy)))
            print("Weak   (SNR < 10)  : "+str(weak_count))
            print("Medium (SNR < 100) : "+str(med_count))
            print("Strong (SNR >= 100): "+str(strong_count))
        
        obl = numpy.zeros(len(xd))
        obl[ii:ff] = outbaseline
        on = numpy.zeros(len(xd))
        on[ii:ff] = outnoise
        return outx, outy, outidx, outsnr, on, obl
  
      
    def analyze_fid(self,index=0):
        """
        Perform full FT and peak fitting analysis of a single FID
        
        Analysis is performed in 2 stages: peak finding and peak fitting.
        For the peak finding stage, the ft_one function is used to compute
        the FT of the FID specified by index. The zero padding factor for this
        FT is forced to 1.
        
        For peak fitting, a second FT with zpf=2 is calculated. The extra
        points help improve the numerical stability of the fitting routines
        without affecting the quality of the fit (though it does lead to
        an underestimation of the parameter errors).
        
        Peak fitting begins by breaking the spectrum into chunks. Starting
        at the first peak, the idea is to make a slice of the spectrum
        containing 2*win points (win = self._settings['an_win']) centered on
        the peak. However, additional peaks may lie close enough to affect the
        quality of the baseline. Therefore, the algorithm looks to see if
        there is another peak higher in frequency by 2*win points. If not, the
        original window (2*win centered on the peak) is used to create a
        slice; otherwise, the window is extended to include the next peak,
        and the process repeats until another higher-frequency peak is not
        found within 2*win points. Note that the point indices in this
        discussion refer to the original FT used for peak finding; the
        actual slices used for fitting come from the zero-padded FT, which
        has a different resolution. The function calculates a resolution
        factor to convert point numbers in the original FT to their
        corresponding indices at approximately the same frequency in the
        zero-padded FT.
        
        For each slice, a set of initial guesses for the parameters is
        used in the fit. Depending on the choice of fitting function (see 
        below), a parameters tuple is constructed as follows:
            params = (y0, A1, x01, w1, A2, x02, w2, ...)
                or
            params = (y0, w, A1, x01, A2, x02, ...),
            
        The model equation is determined by self._settings['an_model'], and
        must be one of the functions defined in bc_fitting. Currently, these
        are:
            bc_multigauss:
                f(x) = y0 + sum_i^N Ai * exp( -(x-x0i)^2 / 2*wi^2 )
            bc_multigauss_fixw:
                f(x) = y0 + sum_i^N Ai * exp( -(x-x0i)^2 / 2*w^2 )
            bc_multigauss_area:
                f(x) = y0 + sum_i^N Ai/(sqrt(2pi)wi) * exp( -(x-x0i)^2 / 2*wi^2 )
        The initial guesses for each parameter come from:
            y0 : the baseline estimate from peak finding stage
            A : peak finder's y value
            x0 : peak finder's x value
            w : width_factor*original_FT_spacing (width factor is set by
            self._settings['an_wf']
        In addition, bounds are set on each parameter:
            y0 : [baseline/5, 5*noise+baseline] (from peak finder)
            A : [A/5, A*5]
            x0 : [x-w, x+w] (assume peak finder is good to ~a few points)
            w : [w/3, w*3]
            
        
        The fitting is performed in a parallelized manner using the python
        concurrent.future.map function with a ProcessPoolExecutor, which
        will spin out processes to fill all available processor cores on the
        computer.
        
        As chunk fits finish, the optimized parameters and their 1 sigma
        uncertainies are stored, and a model is generated for plotting.
        The model contains a number of points equal to all the points used
        in the fitting slices. Since the slices are non-contiguous and
        non-overlapping, there will be gaps in the model arrays.
        Nevertheless, the model waves can be plotted on top of the FT
        with good results; there are (by default) only stright line segments
        connecting the chunks.
        
        Arguments:
        
            index : int (optional, default 0)
                Index of FID in fid_list to analyze

                
        Returns: tuple (x, y, xl, yl, il, sl, pl, el, mx, my, cov)
        
            x : array
                X array for original (i.e., non-zero padded) FT
            
            y : array
                Y array for original FT
                
            xl : list
                List of x values from peak finder
            
            yl : list
                List of y values from peak finder
                
            sl : list
                List of estimated peak SNRs from peak finder
                
            nl : array
                Estimated noise level for original FT
                
            bl : array
                Estimates baseline for original FT
                
            pl : list of tuples
                Optimized fit parameters (y0, A, x0, w) for each peak
                
            el : list of tuples
                1-sigma uncertainties on fit parameters
                
            mx : array
                X array for fit model
                
            yx : array
                Y array for fit model
                
            cov : list of matrices
                Covariance matrix from each fit. This list will almost always
                be shorter than pl and el because multiple peaks are fitted
                together. If you wish to use this matrix and associate the
                values with peaks, you will need to use the shape of the
                matrix to figure out how many peaks are included
                (length = 3*num_peaks + 1). The matrices in the list
                are added in the same order as the peaks in pl and el. Note
                that the y0 value and uncertainty are copied into pl and el
                for all peaks that are fit together
        """
        
        oldzpf = self._settings['ft_zpf']
        self._settings['ft_zpf'] = 1
        x, y = self.ft_one(index=index)
        
        
        xl, yl, il, sl, noise, baseline = self.find_peaks(x,y)
        
        self._settings['ft_zpf'] = 2
        xfit, yfit = self.ft_one(index=index)
        
        self._settings['ft_zpf'] = oldzpf
        
        #zero padded FT has higer resolution. One point in x corresponds to
        #more than 1 point in xfit. Calculate resolution factor
        rf = float(len(xfit))/float(len(x))
        
        #it's best to guess a narrow width and let it grow.
        #assume width = 3*(x spacing) by default
        width = abs(self._settings['an_wf']*(x[1]-x[0]))
        
        #ideally, we'd fit one peak at a time. This isn't possible when peaks
        #are too closely spaced. Set a window size in x space, and find out
        #if any other peaks are within 2 window lengths on either side.
        
        #reverse peak index list so that we can pop elements off the end
        il.reverse()
        
        f_list = [] # wrapper for fit function
        x_list = [] # x slices
        y_list = [] # y slices
        p_list = [] # parameters
        b_list = [] # bounds
        the_metadata = []
        most_peaks = 0
        chunk_index = 0
        total_chunk_size = 0
            
        model = self._settings['an_model']
        win = self._settings['an_win']
        
        while len(il) > 0:
            #get next peak
            peaks = []
            this_peak = il.pop()
            peaks.append(this_peak)
            lindex = max(0,peaks[0]-win)
            rindex = 0
            #see if another peak lies within 2*win.
            if len(il) > 0:
                done = False
                while not done:
                    next_peak = il.pop()
                    if(next_peak - this_peak < 2*win) and len(il) > 0:
                        #peak is nearby. Add it to list and repeat
                        peaks.append(next_peak)
                        this_peak = next_peak
                    else:
                        #no peaks nearby. Put peak back into list for next
                        #iteration
                        done = True;
                        il.append(next_peak)
                        rindex = min(len(y),this_peak + win)
            else:
                #no peaks left
                rindex = min(len(y),this_peak + win)
            
            #construct parameter guesses and boundary conditions
            p = [ baseline[this_peak]]
            lowerbound = [baseline[this_peak] / 5.]
            upperbound = [baseline[this_peak] + 3.*noise[this_peak]]
            if model.__name__ == 'bc_multigauss_fixw':
                p.append(width)
                lowerbound.append(width/3.)
                upperbound.append(width*3.)
            for i in peaks:
                a = y[i]
                if model.__name__ == 'bc_multigauss_area':
                    a *= width*numpy.sqrt(2.*numpy.pi)
                p.append(a)
                p.append(x[i])
                lowerbound.append(a/5.)
                upperbound.append(a*5.)
                lowerbound.append(x[i]-width)
                upperbound.append(x[i]+width)
                if model.__name__ != 'bc_multigauss_fixw':
                    p.append(width)
                    lowerbound.append(width/3.)
                    upperbound.append(width*3.)
            
            most_peaks = max(most_peaks,len(peaks))
  
            #construct slices for fitting          
            left = int(round(lindex*rf))
            right = int(round(rindex*rf))            
            x_list.append(xfit[left:right])
            y_list.append(yfit[left:right])
            p_list.append(tuple(p))
            b_list.append([tuple(lowerbound), tuple(upperbound)])
            f_list.append(model)
            
            #keep track of useful data for later
            the_metadata.append( (chunk_index, total_chunk_size,
                                  right-left, len(peaks)))
            
            total_chunk_size += right-left
            chunk_index += 1
        
        num_chunks = len(the_metadata)
        print("Chunks: "+str(num_chunks))
        print("Most peaks in a chunk: "+str(most_peaks))
        print("Starting fit using following settings")
        self.print_exp_settings(f='an')
        
        param_list = [] #optimized parameters
        err_list = [] #uncertainties
        cov_list = [] #covariance matrices
        modelx = numpy.empty(total_chunk_size) #x data for fitted model
        modely = numpy.empty(total_chunk_size) #y data for fitted model
        
        #prepare progress bar
        pos = 0
        line_width = 80
        end = [" ","0".rjust(4," "),"/",
                       str(num_chunks).rjust(4," ")," (",
                       "0".rjust(3," "),"%)"]
        endstr = "".join(end)
        total_progress = line_width - len(endstr) - 3
        num_hash = 0
        hash_str = '#' * num_hash
        tot = ["\r[",hash_str.ljust(total_progress,' '),"]",endstr]
        sys.stdout.write("".join(tot))
        
        #iterate over slices in parallel, performing a fit on each one
        #when a process finishes, the code beneath is called
        
        with concurrent.futures.ProcessPoolExecutor() as exc:
            for xarray, yarray, md, res in zip(x_list,y_list, the_metadata,
                                 exc.map(bcfit.fit_peaks,f_list,
                                         x_list,y_list,p_list,b_list)):
                                             
                #get information about which chunk this is
                chunk, chunk_start, chunk_len, np = md
                
                #get parameters, covariance matrix, and errors
                popt = res[0]
                pcov = res[1]
                perr = res[2]

                #update progress bar
                pos += 1
                pct = (100*pos)//num_chunks               
                end[1] = str(pos).rjust(4," ") 
                end[5] = str(pct).rjust(4," ") 
                endstr = "".join(end)
                                
                num_hash = (total_progress*pct)//100
                hash_str = '#' * num_hash
                tot[1] = hash_str.ljust(total_progress,' ')
                tot[3] = endstr
                sys.stdout.write("".join(tot))
                
                #build model using optimized parameters
                for i in range(len(xarray)):
                    modelx[chunk_start+i] = xarray[i]
                tmp = model(xarray,*popt)
                for i in range(len(tmp)):
                    modely[chunk_start+i] = tmp[i]
                
                #make parameter and sigma lists
                for i in range(np):
                    if model.__name__ == 'bc_multigauss_fixw':
                        param_list.append ( [popt[0],popt[1],
                                             popt[2*i+2],popt[2*i+3]])
                        err_list.append ( [perr[0],perr[1],
                                           perr[2*i+2],perr[2*i+3]])
                    else:
                        param_list.append( [popt[0], popt[3*i+1],
                                             popt[3*i+2], popt[3*i+3]] )
                        err_list.append( [perr[0], perr[3*i+1],
                                             perr[3*i+2], perr[3*i+3]] )
                    
                
                cov_list.append(pcov)

                del xarray
                del yarray
                del tmp
       
        return x, y, xl, yl, sl, noise, baseline, param_list, err_list, \
               numpy.array(modelx), numpy.array(modely), cov_list

    def processMotorMotorScan(self,carrierName = None, pressure = None):
        if pressure is None:
            pressure = self.time_data['chamberPressure']
            avgChamberPressureTorr = 0
            for i in pressure:
                avgChamberPressureTorr += float(i)
            avgChamberPressureTorr /= len(pressure)
            self.chamberPressure = avgChamberPressureTorr
        else:
            self.chamberPressure = pressure
        
        if carrierName is None:
            try:
                carrierName = self.d_header_values['FlowConfigChannel.0.Name']
            except KeyError:
                raise KeyError('No carrier gas input, please provide the carrier gas to the function with the carrierName argument')

        self.carrier = CarrierGas(carrierName)
#        gamma = self.carrier.gamma
        

        for i in range(self.motorScan.Z_size):
            for j in range(self.motorScan.Y_size):
                for k in range(self.motorScan.X_size):
                    for l in range(self.motorScan.t_size):
                        self.motorScan.Pratio = round(self.motorScan.data[i,j,k,l]/self.avgChamberPressure,4)
#                        self.motorscan.data[i,j,k,l] = brentq((MachNumberRootFinder),0.001,200.,args=(Pi,Ps,gamma))
        return
            
                    
        
        
                
class BlackChirpFid:
    """
    Free induction decay data from the blackchirp program
    
    The blackchirp program saves its FIDs in a binary format (big endian).
    Using the data type conventions of the python struct module, the format
    is:
    
        ms_size -- Number of bytes in magic string (I)
        
        magic_string -- ID/version string, e.g., "BCFIDv1.0" (ms_sizec)
        
        n_fids -- Number of FIDs (I)
        
    The above data need to be read from the fid file prior to constructing
    the BlackChirpFid object. Then, for each FID, use a loop to read the
    remainder of the file. Each FID object is represented, as of v1.0, as
    (all of the following are accessible as attributes):
    
        spacing -- Time separation between points in microseconds (d) [Note:
        in the data file the time is in seconds; this script converts to us]
        
        probe_freq -- Frequency (in MHz) to be added to DC (d)
        
        v_mult -- Scaling factor for converting ADC data to voltage (d)
        
        shots -- Number of individual FIDs recorded (q)
        
        sideband -- Downconversion mixer sideband (H). This is converted to a
        float: either 1.0 (upper sideband) or -1.0 (lower sideband). When
        computing the FT with frequency spacing df, the absolute frequency
        scale is constructed by: probe_freq + sideband*df*point_index
        
        point_size -- Number of bytes representing each data point (b)
        
        size -- Number of data points in the FID (I)
        
    The FID data contained in the file are the sums of raw ADC data from
    the oscilloscope. Rather than averaging floating point data, the
    blackchirp program accumulates each raw ADC value in a signed 64-bit
    register, and counts the total number of shots acquired. When writing the
    fid file, the program determines how many bytes are required to encode
    the sums (options are 2, 3, 4, or 8), and then writes the value of each
    point in binary using that number of bytes (point_size). The total number
    of FID data bytes is therfore point_size*size.
    
    To convert these sums to actual voltage values, a each raw data point is
    divided by the number of shots and multiplied by the v_mult scaling
    factor. For convenience, three arrays are accessible as attributes:
    
        raw_data -- Numpy array containing raw ADC sums
        
        data -- Numpy array containing voltages
        
        x_data -- Numpy array containing time values (point_index*spacing)
        
        xy_data -- 2D numpy array containing x and y data
        (xy_data[0] = x_data, xy_data[1] = data)  
    """

    def __init__(self, version, fidfile):
        """
        Constructor
        
        Reads in data from fidfile object to construct the FID object. For
        details on the binary data format, see the class documentation.
        
        Arguments:
        
            version -- The version string in the data file. If the binary
            format changes in future versions of blackchirp, this can be used
            to ensure backward compatibility
            
            fidfile -- An open file object whose read pointer is set to the
            beginning of the binary FID data
        """
        if version == "1.0":
            read_str = ">3dqHbI"
            d = struct.unpack(read_str,
                              fidfile.read(struct.calcsize(read_str)))
            self.spacing = d[0] * 1e6
            self.probe_freq = d[1]
            self.v_mult = d[2]
            self.shots = d[3]
            if d[4] == 1:
                self.sideband = -1.0
            else:
                self.sideband = 1.0
            self.point_size = d[5]
            self.size = d[6]
            
            self.raw_data = numpy.empty(self.size, dtype=numpy.int64)
            self.data = numpy.empty(self.size, dtype=numpy.float64)
            self.x_data = numpy.empty(self.size, dtype=numpy.float64)
            if self.point_size == 2:
                read_string = '>' + str(self.size) + 'h'
                dat = struct.unpack(read_string,
                                    fidfile.read(
                                    struct.calcsize(read_string)))
                for i in range(0, self.size):
                    self.raw_data[i] = dat[i]
                    self.data[i] = float(dat[i]) * self.v_mult / self.shots
                    self.x_data[i] = float(i) * self.spacing
            elif self.point_size == 3:
                for i in range(0, self.size):
                    chunk = fidfile.read(3)
                    dat = struct.unpack('>i', (b'\0' if chunk[0] < 128
                                                     else b'\xff')
                                                     + chunk)[0]
                    self.raw_data[i] = dat
                    self.data[i] = float(dat) * self.v_mult / self.shots
                    self.x_data[i] = float(i) * self.spacing
            elif self.point_size == 4:
                read_string = '>' + str(self.size) + 'i'
                dat = struct.unpack(read_string,
                                    fidfile.read(
                                    struct.calcsize(read_string)))
                for i in range(0, self.size):
                    self.raw_data[i] = dat[i]
                    self.data[i] = float(dat[i]) * self.v_mult / self.shots
                    self.x_data[i] = float(i) * self.spacing
            elif self.point_size == 8:
                read_string = '>' + str(self.size) + 'q'
                dat = struct.unpack(read_string,
                                    fidfile.read(
                                    struct.calcsize(read_string)))
                for i in range(0, self.size):
                    self.raw_data[i] = dat[i]
                    self.data[i] = float(dat[i]) * self.v_mult / self.shots
                    self.x_data[i] = float(i) * self.spacing
            else:
                raise ValueError("Invalid point size: "
                                  + str(self.point_size))
            
        else:
            raise ValueError("Version " + version + " not supported")
        
        self.xy_data = numpy.vstack((self.x_data, self.data))
        
    @classmethod
    def fromTextData(self,fid,header):
        
        shots = int(header["Shots"])
        spacing = 1.0/(float(header["Sample rate"])*1e3)
        probe = float(header["LO Frequency"])
        
        self.data = fid
        self.x_data = numpy.linspace(0.0,(len(fid)-1)*spacing,len(fid))
        self.spacing = spacing
        self.shots = shots
        self.probe_freq = probe
        self.xy_data = numpy.vstack((self.x_data, self.data))
        self.size = len(fid)
        self.sideband = 1.0
        if header["Sideband"] != "Upper":
            self.sideband = -1.0
        return self
  
class BlackChirpMotorScan:
    def __init__(self,number,path=None,headerData=None):
        if number < 1:
            raise ValueError('Experiment number must be > 0')
            
        self.d_number = number
        
        millions = number // 1000000
        thousands = number // 1000
        
        self.x0 = 0.
        self.dx = 1.        
        self.y0 = 0.
        self.dy = 1.
        self.z0 = 0.
        self.dz = 1.
        self.t0 = 0.
        self.dt = 1.
        
        if headerData is not None:
            try:
                self.x0 = float(headerData['MotorScanXStart'])
                self.y0 = float(headerData['MotorScanYStart'])
                self.z0 = float(headerData['MotorScanZStart'])
                self.dx = float(headerData['MotorScanXStep'])
                self.dy = float(headerData['MotorScanYStep'])
                self.dz = float(headerData['MotorScanZStep'])
                self.dt = float(headerData['MotorScanTStep'])
            except:
                pass
        
        if path is None:
            path = "/home/data/blackchirp/experiments/" + str(millions) \
                   + "/" + str(thousands) + "/" + str(number)
        
        mdt = open(path+'/'+'%i.mdt'%number,'rb') 
        buffer = mdt.read(4)
        ms_len = struct.unpack(">I",buffer)
        buffer = mdt.read(ms_len[0])
        magic_string = buffer.decode('ascii')
        buffer = mdt.read(4)
        self.Z_size = struct.unpack(">I", buffer)[0]
        buffer = mdt.read(4)
        self.Y_size = struct.unpack(">I", buffer)[0]
        buffer = mdt.read(4)
        self.X_size = struct.unpack(">I", buffer)[0]
        buffer = mdt.read(4)
        self.T_size = struct.unpack(">I", buffer)[0]
        
        self.rawdata = numpy.zeros((self.Z_size,self.Y_size,self.X_size,self.T_size))
        self.data = numpy.zeros((self.Z_size,self.Y_size,self.X_size,self.T_size))
        
        
        for i in range(self.Z_size):
            for j in range(self.Y_size):
                for k in range(self.X_size):
                    for l in range(self.T_size):
                        unpacked = struct.unpack(">d", mdt.read(struct.calcsize(">d")))[0]
                        self.rawdata[i,j,k,l] = unpacked
                    mdt.read(4)
                mdt.read(4)
            mdt.read(4)
        mdt.close()
        self.data = self.rawdata/0.0193368
        
        self.xMax = self.x0 + self.dx*(self.X_size-1)
        self.yMax = self.y0 + self.dy*(self.Y_size-1)
        self.zMax = self.z0 + self.dz*(self.Z_size-1)
        self.tMax = self.t0 + self.dt*(self.T_size-1)
        
    def xPoints(self):
        return numpy.arange(self.X_size)*self.dx + self.x0
    
    def yPoints(self):
        return numpy.arange(self.y_size)*self.dy + self.y0
    
    def zPoints(self):
        return numpy.arange(self.Z_size)*self.dz + self.z0
    
    def tPoints(self):
        return numpy.arange(self.T_size)*self.dt + self.t0
        
    def tSlice(self,x,y,z):
        if x < 0 or x >= self.X_size:
            return numpy.zeros(0)
        if y < 0 or y >= self.Y_size:
            return numpy.zeros(0)
        if z < 0 or z >= self.Z_size:
            return numpy.zeros(0)
        
        return self.data[z,y,x,:]
    
    def xSlice(self,y,z,t):
        if t < 0 or t >= self.T_size:
            return numpy.zeros(0)
        if y < 0 or y >= self.Y_size:
            return numpy.zeros(0)
        if z < 0 or z >= self.Z_size:
            return numpy.zeros(0)
        
        return self.data[z,y,:,t]
    
    def ySlice(self,x,z,t):
        if t < 0 or t >= self.T_size:
            return numpy.zeros(0)
        if x < 0 or x >= self.X_size:
            return numpy.zeros(0)
        if z < 0 or z >= self.Z_size:
            return numpy.zeros(0)
        
        return self.data[z,:,x,t]
    
    def zSlice(self,x,y,t):
        if t < 0 or t >= self.T_size:
            return numpy.zeros(0)
        if x < 0 or x >= self.X_size:
            return numpy.zeros(0)
        if y < 0 or y >= self.Y_size:
            return numpy.zeros(0)
        
        return self.data[:,y,x,t]
    
    def xySlice(self,z,t):
        if t < 0 or t >= self.T_size:
            return numpy.zeros(0)
        if z < 0 or z >= self.Z_size:
            return numpy.zeros(0)
        
        return self.data[z,:,:,t]
    
    def xzSlice(self,y,t):
        if t < 0 or t >= self.T_size:
            return numpy.zeros(0)
        if y < 0 or y >= self.Y_size:
            return numpy.zeros(0)
        
        return self.data[:,y,:,t]
    
    def yzSlice(self,x,t):
        if t < 0 or t >= self.T_size:
            return numpy.zeros(0)
        if x < 0 or x >= self.X_size:
            return numpy.zeros(0)
        
        return self.data[:,:,x,t]
    
    def xtSlice(self,y,z):
        if y < 0 or y >= self.Y_size:
            return numpy.zeros(0)
        if z < 0 or z >= self.Z_size:
            return numpy.zeros(0)
        
        return self.data[z,y,:,:]
    
    def ytSlice(self,x,z):
        if x < 0 or x >= self.X_size:
            return numpy.zeros(0)
        if z < 0 or z >= self.Z_size:
            return numpy.zeros(0)
        
        return self.data[z,:,x,:]
    
    def ztSlice(self,x,y):
        if y < 0 or y >= self.Y_size:
            return numpy.zeros(0)
        if x < 0 or x >= self.X_size:
            return numpy.zeros(0)
        
        return self.data[:,y,x,:]
    
    def xVal(self,x):
        if x < 0 or x >= self.X_size:
            return None
        
        return self.x0 + x*self.dx
    
    def yVal(self,y):
        if y < 0 or y >= self.Y_size:
            return None
        
        return self.y0 + y*self.dy
    
    def zVal(self,z):
        if z < 0 or z >= self.Z_size:
            return None
        
        return self.z0 + z*self.dz
    
    def tVal(self,t):
        if t < 0 or t >= self.T_size:
            return None
        
        return self.t0 + t*self.dt
    
    def smooth(self,winSize,polyOrder):
        for i in range(self.Z_size):
            for j in range(self.Y_size):
                for k in range(self.X_size):
                    self.data[i,j,k,:] = spsig.savgol_filter(self.rawdata[i,j,k,:]/0.0193368,winSize,polyOrder)
                    
                    
    def machNumber(self,pi,pStag,gamma):
        out = spopt.brentq((machNumberRootFinder),1,200.,args=(pi,pStag,gamma))
        if out < 1 or out > 8:
            return 0;
        
        return out
        
        
        
class CarrierGas:
    def __init__(self,Carrier):
        if Carrier == 'Ar':
            self.name = Carrier
            self.mass = 6.6335209e-26
            self.gamma = 5./3.
        elif Carrier == 'N2':
            self.name = Carrier
            self.mass = 4.6517342e-26
            self.gamma = 7.0/5.0
        elif Carrier == 'He':
            self.name = Carrier
            self.mass = 6.6464764e-27
            self.gamma = 5./3.
        else:
            raise ValueError('Invalid Carrier Gas, currently accepted values are Ar/N2/He')
            sys.exit(41)
    
def machNumberRootFinder(M, Pi, Ps, gamma):
    return (((gamma+1)*M*M)/((gamma-1)*M*M+2))**(gamma/(gamma-1))*(gamma+1)/(2*gamma*M*M-gamma+1)**(1/(gamma-1)) - Pi/Ps


      
############################################################################
#                          INITIALIZATION                                  #
############################################################################
        
BlackChirpExperiment.load_settings()
