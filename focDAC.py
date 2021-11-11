######################################################################
#Malcolm's DAC script
#
#New version for full data reduction in mantid. Allows for set-up and production modes
# hard-wired to process detector columns as separate banks, thus 6 banks on SNAP
#
#
#9th March 2020
#
#28 APRIL 2020 - M. GUTHRIE 
    #-1) added possibilty to calculate diamond attenuation correction (mode 4)    
    #,using output from fitTransPy, and applying same mask and calibration as sample  
    # 2) also added possibility to store useful reduction data: V,VB run number and calibration file name 
    # as an ascii file
    # 3) added trailing / at end of directory specification.
#7 May 2020 - M. Guthrie
    #after getting tied in total knots by binning (histogram vs point) and bin normalisation (distribution vs counts), have
    #completely re-visited JT's code from the beginning
    
#15 June 2020 M. Guthrie
    #more beta-testing indicated it's useful to be able to specify a distinct directory locations for vanadium/background runs
    #versus local datasets. Specified this and set it up to apply during mode 2 vanadium set up.
    
 # 24 July 2020 M. Guthrie
    #edited to handle various things that were different in old data: 
    #   1) data files stored in different directory and with different file names
    #   2) only one Monitor spectrum stored versus the two stored currently
    
# 4 Dec 2020 M. Guthrie
    #added interactivity via QT popup to allow:
    #    1)  entering run mode
    #    2)  possibility to reject existing set-up file
    #
    #modified logic to ensure that in mode 2, existing vanadium and mt data are always re-read and masked.
    #
    #Note: think I will need to save vcorr data, with info on mask used to generate it. then would also
    # write name of specific vcorr data file in set-up log.
    #
# 22 Feb 2021 M. Guthrie
    # added saving of vcorr information to file. script will now check if this file already exists before
    # re-generating vcorr. All settings required to generate vanadium are already written to log file
    # vanadium run number
    # vanadium background
    # vanadium geometry (assumed cylindrical for now)
    # mask file
    #
# 5 Aug 2021 M. Guthrie
    # while trouble shooting calibration have added option to not use a detcal file
    # in this case, the the default IDF (including motor positions recorded in logs) will
    # be used for conversion.
    
# 11 Aug 2021 M. Guthrie
    # added some diagnostics during vcorr creation to trouble-shoot weird edge effects that showed update   
    # when selecting fine d-space binning: 0.5,-0.001,7
    #
    # Also, note to self: add a final step of TOF binning of reduced data prior to 
   
######################################################################
# The following line helps with future compatibility with Python 3
# print must now be used as a function, e.g print('Hello','World')
from __future__ import (absolute_import, division, print_function, unicode_literals)
# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
from mantidqt.utils.qt.qappthreadcall import QAppThreadCall
import matplotlib.pyplot as plt
import sys
import numpy as np
import shutil
import os
import mgutilities as mgutilities

############################################################################################
# User editable stuff starts here
############################################################################################
#All files are expected to use this tags and extensions and to exist in these directories
tag = 'SNAP'
extn = 'nxs.h5' #changed from 'nxs' to 'nxs.h5' in later runs
IPTS = 24179 #ipts for sample data
VIPTS = 24179 #ipts for vanadium and its background (not always same as sample)
datafolder = 'nexus' #changed from 'data' to 'nexus' at some point
sharedDir = '/SNS/snfs1/instruments/SNAP/IPTS-'+str(IPTS)+'/shared/' #equivalent to "shared" directory in SNS file structure  
dataDir = '/SNS/snfs1/instruments/SNAP/IPTS-'+str(IPTS)+'/'+datafolder+'/' #location of sample nexus files, typically '../data' at SNS
VDir = '/SNS/snfs1/instruments/SNAP/IPTS-'+str(VIPTS)+'/'+datafolder+'/' #location of vanadium and vanadium background nexus files
instFileLoc = '/SNS/snfs1/instruments/SNAP/shared/Malcolm/dataFiles/SNAP_Definition.xml'
dbinning ='0.65,-0.001,7' #dmin, dbin, dmax applied to all pixels
MonTBinning = '3520,30,15800' #tof_min, tof_bin, tof_max for monitor spectra !THIS ALSO NEEDS TO BE STANDARDISED!
runs = [51947] #enter as comma separated list:
#measurement specific info:
msknm = '' #(DONT use .xml extension) will be ignored in mode 1
                        #leave empty if no mask is required
Vrun = 51940
VBrun = 51941
vHeight = 0.06 #illuminated height of vanadium pin (cm)
vRad = 0.035 #radius of vanadium pin (cm)
detCalName = 'none'  # detcal filename, expected to be in shared folder                                                # if you don't want to use a detcal, replace with 'none' (all lower case)
uStreamDiam = 1 #should be equal to either 1 or 2 depending on which diamond is upstream  
# Limits below will change with chopper setting and detector positions.
# To do: define standard values for these.
nCol = 6 #don't change
dLim = np.zeros([nCol,2]) #Contains useful d-spacing limits for individual columns
dLim[0,0]=0.5 # Col 1
dLim[0,1]=2.15
dLim[1,0]=0.55 # Col 2
dLim[1,1]=2.4
dLim[2,0]=0.60 # Col 3
dLim[2,1]=2.75
dLim[3,0]=0.63 # Col 4
dLim[3,1]=2.95
dLim[4,0]=0.75 # Col 5
dLim[4,1]=3.75
dLim[5,0]=0.93 # Col 6
dLim[5,1]=5.0
############################################################################################
# User editable stuff ends here
############################################################################################


##################################################################
# set running mode
#
#If this is the first time running, check that lines 75-97 have the correct information in them,
#You can also copy an existing SNAP_____SetUp.log file, changing name to match first run in runs
##################################################################
input = QAppThreadCall(mgutilities.workbench_input_fn)
mode = input('Enter mode: ','1: setup UB and mask, 2: setup vanadium, 3: process data, 4: process Att corrn ', 'int')
#modes:
# 1 - Keep diamond reflections: calculate UB and Define Mask
# 2 - Set up Vanadium: 
     #File containing mask with name below should exist! 
     #Appropriate run numbers for V and its background should have been defined below
# 3 - Reduce Sample Data and calculate diamond transmission
     #VCorr workspace (vanadium correction should exist i.e. script has been run in mode 2)
#4 -  Focus diamond attenuation correction



##################################################################
# Initial Set up here
##################################################################
# Check if set-up file exists and, if so, read set up information from there.
# n.b. assumption here that existing set up file for first run in runs applies to all subsequent runs
setupFname = sharedDir + tag + str(runs[0]) + 'SetUp.log'
if os.path.isfile(setupFname):
    print(setupFname)
    print('Previous set-up file found. Options: 0 - use this, 1 - ignore and overwrite')
    setupAction = input('Previous setup file found: ','0: use file, 1: overwrite with defaults ', 'int')
    print(setupAction)
    if setupAction==0:
        with open(setupFname,'r') as f:
            msknm = f.readline().strip()
            Vrun = int(f.readline().strip())
            VBrun = int(f.readline().strip())
            detCalName = f.readline().strip()
            dbinning = f.readline().strip()
            MonTBinning = f.readline().strip()
            vHeight = float(f.readline().strip())
            vRad = float(f.readline().strip())
            nCol = int(f.readline().strip())
            dLim = np.zeros([nCol,2])
            for i in range(nCol):
                dLim[i,0] = float(f.readline().strip())
                dLim[i,1] = float(f.readline().strip())        
    else:
        print('Over-writing previous set-up file found using defaults')
        
else:
    print('No set up file written, defaults will be written to file')
    

##################################################################
# process data
##################################################################
tlims = MonTBinning.split(',')
tof_min = tlims[0]
tof_max = tlims[2]
#Set up various toggles for operations that depend on running mode
#modeSet[0] = Apply vanadium correction
#modeSet[1] = Mask: 0 = don't use mask; 1 use mask in workspace; 2 use mask read from file
#modeSet[2] = if true save ascii file of monitor data vs TOF 
#modeSet[3] = if true save ascii file of sample data vs d-spacing

if mode ==2:
    runs = [Vrun, VBrun] #overwrite runs, ignoring what's written above
    dataDir = VDir #
    VCorrFname = sharedDir + tag + str(runs[0]) + '_' + str(runs[1]) + 'VCorr.nxs'
    VBMonFname = '/SNS/SNAP/IPTS-%s/shared/%s%s_VBmon.nxs'%(IPTS,tag,VBrun)
    if os.path.isfile(VCorrFname):
        print('Previous vanadium correction file found. Options: 0 - use this, 1 - ignore')
        useVCorrFile = input('Previous vanadium correction file found: ','0: use file, 1: ignore ', 'int')
        print(useVCorrFile)
        if useVCorrFile == 0:
            LoadNexus(Filename=VCorrFname, OutputWorkspace='VCorr')
            LoadNexus(Filename=VBMonFname, OutputWorkspace='%s%s_monitors'%(tag,VBrun))
    else:
        useVCorrFile = 1


if mode==1:
    modeSet = [0,0,0,0] 
elif mode==2:
    modeSet = [0,2,0,0]
elif mode==3:
    modeSet = [1,2,0,0]
elif mode==4:
    modeSet = [0,0,0,0]

if not msknm:
    modeSet[1] = 0  #no mask has been specified so don't use one.

if modeSet[1] == 2: 
    mskfilenm = r'%s%s.xml'%(sharedDir,msknm)
    try:
        a = mtd[msknm] #if workspace already exists, don't process again!
    except:
        LoadMask(Instrument='snap', InputFile='%s'%(mskfilenm), OutputWorkspace='%s'%(msknm))

# Set up column grouping workspace
if mode!=1:
    try:
       a = mtd['SNAPColGp'] #if workspace already exists, don't process again!
    except:
       CreateGroupingWorkspace(InstrumentFilename=instFileLoc, GroupDetectorsBy='Column', OutputWorkspace='SNAPColGp')

for run in runs:
    print('Processing run :',run)
# load nexus file including monitor data		
    if mode==1:
        LoadEventNexus(Filename=r'%s/%s_%s.%s'%(dataDir,tag,run,extn),OutputWorkspace='%s%s'%(tag,run),FilterByTofMin=tof_min, FilterByTofMax=tof_max, Precount='1', LoadMonitors=True)
        #LoadEventNexus(Filename=r'%s/%s_%s.%s'%(dataDir,tag,run,extn),OutputWorkspace='%s%s'%(tag,run),Precount='1', LoadMonitors=True)
        NormaliseByCurrent(InputWorkspace='%s%s_monitors'%(tag,run),OutputWorkspace='%s%s_monitors'%(tag,run))
        Rebin(InputWorkspace='%s%s_monitors'%(tag,run),OutputWorkspace='%s%s_monitors'%(tag,run),Params=MonTBinning)
        NormaliseByCurrent(InputWorkspace='%s%s'%(tag,run),OutputWorkspace='%s%s'%(tag,run))  
        CompressEvents(InputWorkspace='%s%s'%(tag,run),OutputWorkspace='%s%s'%(tag,run))
    elif mode==2 and useVCorrFile!=0: #always re-read and mask vanadium data unless VCorr taken from file.
        LoadEventNexus(Filename=r'%s/%s_%s.%s'%(dataDir,tag,run,extn),OutputWorkspace='%s%s'%(tag,run),FilterByTofMin=tof_min, FilterByTofMax=tof_max, Precount='1', LoadMonitors=True)
        NormaliseByCurrent(InputWorkspace='%s%s_monitors'%(tag,run),OutputWorkspace='%s%s_monitors'%(tag,run))
        Rebin(InputWorkspace='%s%s_monitors'%(tag,run),OutputWorkspace='%s%s_monitors'%(tag,run),Params=MonTBinning)
        NormaliseByCurrent(InputWorkspace='%s%s'%(tag,run),OutputWorkspace='%s%s'%(tag,run))  
        CompressEvents(InputWorkspace='%s%s'%(tag,run),OutputWorkspace='%s%s'%(tag,run))
    elif mode==2 and useVCorrFile == 0:
        continue #no need to read in any data.
    else:
        try:
            a = mtd['%s%s'%(tag,run)] #if workspace already exists, don't process again!
        except:
            LoadEventNexus(Filename=r'%s/%s_%s.%s'%(dataDir,tag,run,extn),OutputWorkspace='%s%s'%(tag,run),FilterByTofMin=tof_min, FilterByTofMax=tof_max, Precount='1', LoadMonitors=True)
            NormaliseByCurrent(InputWorkspace='%s%s_monitors'%(tag,run),OutputWorkspace='%s%s_monitors'%(tag,run))
            Rebin(InputWorkspace='%s%s_monitors'%(tag,run),OutputWorkspace='%s%s_monitors'%(tag,run),Params=MonTBinning)
            NormaliseByCurrent(InputWorkspace='%s%s'%(tag,run),OutputWorkspace='%s%s'%(tag,run))  
            CompressEvents(InputWorkspace='%s%s'%(tag,run),OutputWorkspace='%s%s'%(tag,run))
        if modeSet[3]==1:
            SaveAscii(InputWorkspace='%s%s_monitors'%(tag,run),Filename='%s%s%smon.csv'%(sharedDir,tag,run))
# Process data as required:
    if modeSet[1] == 2 and mode ==3: # mask detectors here if mask read in from file (currently always the case for mode 2 and 3
        MaskDetectors(Workspace='%s%s'%(tag,run),MaskedWorkspace='%s'%(msknm))
    elif modeSet[1] ==2 and mode==2 and useVCorrFile != 0:
        MaskDetectors(Workspace='%s%s'%(tag,run),MaskedWorkspace='%s'%(msknm))

    if mode==1 or mode ==3:    
        if detCalName.lower() != 'none':
            LoadIsawDetCal(InputWorkspace='%s%s'%(tag,run), Filename=sharedDir+detCalName)
        ConvertUnits(InputWorkspace='%s%s'%(tag,run), OutputWorkspace='%s%s_d'%(tag,run), Target='dSpacing')
    elif mode==2 and useVCorrFile !=0:
        if detCalName.lower() != 'none':
            LoadIsawDetCal(InputWorkspace='%s%s'%(tag,run), Filename=sharedDir+detCalName)
        ConvertUnits(InputWorkspace='%s%s'%(tag,run), OutputWorkspace='%s%s_d'%(tag,run), Target='dSpacing')        
 #   DeleteWorkspace(Workspace='%s%s'%(tag,run))
        #Rebin(InputWorkspace='%s%s_d'%(tag,run),OutputWorkspace='%s%s_d'%(tag,run),Params=dbinning)
    if mode != 2:
        try:
            a = mtd['%s%s_d_8x8'%(tag,run)] #if workspace already exists, don't process again! NOTE bug here: if mask has been changed it won't be applied!
        except:
            SumNeighbours(InputWorkspace='%s%s_d'%(tag,run),SumX=8,SumY=8,OutputWorkspace='%s%s_d_8x8'%(tag,run))
    elif mode ==2 and useVCorrFile !=0:
        try:
            a = mtd['%s%s_d_8x8'%(tag,run)] #if workspace already exists, don't process again! NOTE bug here: if mask has been changed it won't be applied!
        except:
            SumNeighbours(InputWorkspace='%s%s_d'%(tag,run),SumX=8,SumY=8,OutputWorkspace='%s%s_d_8x8'%(tag,run))
    elif mode ==2 and useVCorrFile ==0:
        continue
        
    if modeSet[1] ==1: # mask detectors here if mask saved direct to workspace
        MaskDetectors(Workspace='%s_%s_d_8x8'%(tag,run),MaskedWorkspace='%s'%(msknm))	

    if mode!=1:
        
        try:
            a = mtd['%s%s_d6'%(tag,run)] #if workspace already exists, don't process again!
        except:
            DiffractionFocussing(InputWorkspace='%s%s_d'%(tag,run), OutputWorkspace='%s%s_d6'%(tag,run), GroupingWorkspace='SNAPColGp')
            Rebin(InputWorkspace='%s%s_d6'%(tag,run),OutputWorkspace='%s%s_d6'%(tag,run),Params=dbinning,PreserveEvents=False)
#    if mode != 4:    
#        DeleteWorkspace(Workspace='%s%s_d'%(tag,run))
#    SumSpectra(InputWorkspace='%s_%s_d_8x8'%(tag,run),OutputWorkspace='%s_%s_d_8x8_sum'%(tag,run),IncludeMonitors='0')	
    if modeSet[0] ==1:
        Divide(LHSWorkspace='%s%s_d6'%(tag,run), RHSWorkspace='VCorr', OutputWorkspace='%s%s_d6_VC'%(tag,run))
    if modeSet[3] ==1:
        SaveAscii(InputWorkspace='%s%s_d6_VC'%(tag,run),Filename='/SNS/SNAP/IPTS-%s/shared/%s%sdspa_obs_sam.csv'%(IPTS,tag,run))
        
    if mode==1:# kept needing this again! can probably be deleted if I ever get out of beta mode...
        CloneWorkspace(InputWorkspace='%s%s_d_8x8'%(tag,run), OutputWorkspace='%s%s_d_8x8_noMsk'%(tag,run))        

if mode==2 and useVCorrFile!= 0:# set up vanadium correction
    Minus(LHSWorkspace='SNAP'+str(Vrun)+'_d6', RHSWorkspace='SNAP'+str(VBrun)+'_d6', OutputWorkspace='VCorr')
    StripVanadiumPeaks(InputWorkspace='VCorr', OutputWorkspace='VCorr')
    SmoothData(InputWorkspace='VCorr', OutputWorkspace='VCorr_sm', NPoints='100')
    SetSampleMaterial(InputWorkspace='VCorr_sm', ChemicalFormula='V')
    ConvertUnits(InputWorkspace='VCorr_sm', OutputWorkspace='VCorr_sm', Target='Wavelength')
    CylinderAbsorption(InputWorkspace='VCorr_sm', OutputWorkspace='VCorr_a', AttenuationXSection=5.08, ScatteringXSection=5.10, CylinderSampleHeight=vHeight, CylinderSampleRadius=vRad, CylinderAxis='0,0,1')
    Divide(LHSWorkspace='VCorr_sm', RHSWorkspace='VCorr_a', OutputWorkspace='VCorr_sma')
    ConvertUnits(InputWorkspace='VCorr_sma', OutputWorkspace='VCorr_sma', Target='dSpacing')
    DeleteWorkspace(Workspace='VCorr_a')
    DeleteWorkspace(Workspace='SNAP'+str(Vrun)+'')
    #DeleteWorkspace(Workspace='SNAP'+str(Vrun)+'_d6')
    DeleteWorkspace(Workspace='SNAP'+str(Vrun)+'_d_8x8')
    DeleteWorkspace(Workspace='SNAP'+str(Vrun)+'_monitors')
    DeleteWorkspace(Workspace='SNAP'+str(VBrun)+'')
    #DeleteWorkspace(Workspace='SNAP'+str(VBrun)+'_d6')
    DeleteWorkspace(Workspace='SNAP'+str(VBrun)+'_d_8x8')
    SaveNexus(InputWorkspace='VCorr', Filename='/SNS/SNAP/IPTS-%s/shared/%s%s_%sVCorr.nxs'%(IPTS,tag,Vrun,VBrun), \
    Title='Van: %s, Van Backgnd: %s'%(Vrun,VBrun))
    SaveNexus(InputWorkspace='%s%s_monitors'%(tag,VBrun), \
    Filename='/SNS/SNAP/IPTS-%s/shared/%s%s_VBmon.nxs'%(IPTS,tag,VBrun), \
    Title='Downstream monitor counts run: %s'%(VBrun))
#    DeleteWorkspace(Workspace='SNAP'+str(VBrun)+'_monitors')

if mode==3: 
    #first write set-up information to file for future use
    setupFname = sharedDir + tag + str(run) + 'SetUp.log'
    with open(setupFname,'w') as f:
        print('Saving set-up info to file: ',setupFname)
        f.write(msknm+'\n')
        f.write(str(Vrun)+'\n')
        f.write(str(VBrun)+'\n')
        f.write(detCalName+'\n')
        f.write(dbinning+'\n')
        f.write(MonTBinning+'\n')
        f.write(str(vHeight)+'\n')
        f.write(str(vRad)+'\n')
        f.write(str(nCol)+'\n')
        for i in range(nCol):
            f.write(str(dLim[i,0])+'\n')
            f.write(str(dLim[i,1])+'\n')
    #now continue processing
    MTmonWs = mtd['%s%s_monitors'%(tag,VBrun)]
    MTdsMonX = MTmonWs.readX(1) #second histogram is d/stream monitor on SNAPReduce
    MTdsMonY = MTmonWs.readY(1)
    MTdsMonE = MTmonWs.readE(1) 
    MTdsMon_relE = np.square(np.divide(MTdsMonE,MTdsMonY))    #square of relative errors
    for run in runs:
        # remove NAN and clean up edges of range after division by VCorr
        ws = mtd['%s%s_d6_VC'%(tag,run)]
        xIn = ws.readX(0) #x-values of 0th histogram 
        yTmp = ws.readY(0)
        nPts = yTmp.size
        nHst = ws.getNumberHistograms()
        yOut = np.zeros([nHst,nPts]) # empty array to store modified data
        eOut = np.zeros([nHst,nPts]) # empty array to store modified data
        clr = np.zeros([nHst,nPts]) # empty array to store modified data
        for i in range(nHst):
            yIn = ws.readY(i)
            ynan = np.isnan(yIn)
            eIn = ws.readE(i)
            enan = np.isnan(eIn)
            usefulIndx = np.where(np.logical_and(xIn>dLim[i,0], xIn<dLim[i,1])) #indices inside useful range
            clr[i,usefulIndx[0][:]]=1.0
            eOut[i,] = eIn
            eOut[i,enan] = 0.0
            yOut[i,] = yIn
            eOut[i,ynan] = 0.0
        yOut = np.multiply(clr,yOut) #set all non-usable y-values to zero
        eOut = np.multiply(clr,eOut) #set all non-usable e-values to zero
#        outTrm= CreateWorkspace(DataX = xIn, DataY = yOut, NSpec = nHst, Distribution = False, UnitX = "d-Spacing ")
        outTrm = CloneWorkspace('%s%s_d6_VC'%(tag,run))
        for i in range(nHst):
            outTrm.setY(i,yOut[i,])
            outTrm.setE(i,eOut[i,])
        RenameWorkspace(InputWorkspace='outTrm', OutputWorkspace='%s%s_d6_VCT'%(tag,run))
        #SumSpectra(InputWorkspace='%s%s_d6_VCT'%(tag,run),OutputWorkspace='SNAP_%s_All_nor'%(run))
        
        # generate I/I_0 for attenuation correction, using ratio of sample d/stream monitor spectrum and empty instrument monitor spectrumInfo
        monWs = mtd['%s%s_monitors'%(tag,run)]
        dsMonX = monWs.readX(1) #second histogram is d/stream monitor on SNAPReduce
        dsMonY = monWs.readY(1)
        dsMonE = monWs.readE(1)
        IoIo = np.divide(dsMonY,MTdsMonY)
        #add errors in quadrature
        dsMon_relE = np.square(np.divide(dsMonE,dsMonY)) #square of relative errors
        IoIoE = np.sqrt(dsMon_relE + MTdsMon_relE)
        
        # normalise to spectrum to mean value between 13000 and 16500 us
        nrmIndx = np.where(np.logical_and(dsMonX>=13000.0, dsMonX<=float(tof_max)-100)) 
        nrm = np.mean(IoIo[nrmIndx])
        IoIo = IoIo/nrm
        IoIows =  CreateWorkspace(DataX = MTdsMonX, DataY = IoIo, DataE = IoIoE, NSpec = 1, Distribution = True, UnitX = "TOF")
        RenameWorkspace(InputWorkspace='IoIows', OutputWorkspace='%s%s_monitors_ioio'%(tag,run))

if mode == 4:
    # if upstream diamond transmission exists process, otherwise can't do anything
#    try:
      TransIn_lam=CloneWorkspace('%s%s_trns_diam%s'%(tag,run,uStreamDiam))
      TransIn_lam=ConvertToHistogram('TransIn_lam') #JT: This is important to match the data type before rebinning
      Xin = TransIn_lam.readX(0)
      NPts = Xin.size
      minXIn = Xin[1]# take value one in from end to avoid bin edges
      maxXIn = Xin[NPts-2]
      #following code courtesy of J. Taylor (with comments labelled "JT")...
      RefData_d = CloneWorkspace('%s%s_d_8x8'%(tag,run))
      ConvertUnits(Inputworkspace='RefData_d',outputWorkspace='RefData_lam',Target='Wavelength')
      Rebin(InputWorkspace='RefData_lam', OutputWorkspace='RefData_lam', Params='0.5,0.0025,10', PreserveEvents=False)
      TransIn_lam = RebinToWorkspace('TransIn_lam','RefData_lam') #single histogram containing correction with same binning as reference dataset
      tmpY = TransIn_lam.readY(0)
      tmpX = TransIn_lam.readX(0)
      #JT: here we find the max value of the Transmission data, it should be one, the rebin changes the y values so here we reset them to be 1
      maxVal=tmpY.max()
      print('Max value of trans and its reciprocal: ',maxVal, 1.0/maxVal)
      tmpY=tmpY * (1.0/maxVal)
      factor=CreateSingleValuedWorkspace((1.0/maxVal))
      TransIn_lam=TransIn_lam * factor
      tmpY =TransIn_lam.extractY()
      tmpY=tmpY[0,:]
      tmpX =TransIn_lam.extractX()
      tmpX=tmpX[0,:]
      #Outside of calculated range, set transmission to 100%...
      for i in range(tmpY.size):
          if tmpX[i]>=maxXIn:
              tmpY[i]=1.0
          if tmpX[i]<=minXIn:
              tmpY[i]=1.0
      TransIn_lam.setY(0,tmpY)
      RefData_lam = CloneWorkspace('RefData_lam') #don't know why, but script can't see RefData_lam...infuriating!!!!
      NumberOfSpectra= RefData_lam.getNumberHistograms()
      for i in range(NumberOfSpectra):
        RefData_lam.setY(i,tmpY)
        RefData_lam.setX(i,tmpX)
      RefData_lam.setYUnitLabel('Attenuation')   
      axis = RefData_lam.getAxis(0)
      inunit = axis.getUnit()
      print("input caption:{0}".format(inunit.caption()))
      print("input symbol:{0}\n".format(inunit.symbol())) 
#      RefData_lam = 

#      ConvertToMatrixWorkspace(InputWorkspace='tst', OutputWorkspace='')
      MaskDetectors(Workspace='RefData_lam',MaskedWorkspace='%s'%(msknm))
      ConvertUnits(InputWorkspace='RefData_lam', OutputWorkspace='RefData_dcorr', Target='dSpacing')
      DiffractionFocussing(InputWorkspace='RefData_dcorr', OutputWorkspace='RefData_d6', GroupingWorkspace='SNAPColGp')
      #For reasons I don't yet understand, DiffractionFocusing is dividing by the bin widths, as though it believes the input ws is a distribution and it wants it to be a 
      #histogram. I don't know why DiffractionFocusing is making this decision.
      #Anyway, to reverse the effect, I have to use a convertToDistribution, which divides by binwidth. The results then *not* a distribution, but is the histogram I want
      #I can't quite find words to express my annoyance with this...lol.
      ConvertToDistribution('RefData_d6')
      #need to renormalise correction after focusing, can use high d-spacing to do this. By inspection, all columns in SNAPReduce  
      #have 100% transmission above 4 Angstrom d-spacing so use this...
      #
      #While I'm at it, apply same d-spacing limits as for data
     
      nrmTransMinD = 4.5
      nrmTransMaxD = 5.0
      wsIn = mtd['RefData_d6']
      RebinToWorkspace(WorkspaceToRebin=wsIn, WorkspaceToMatch='%s%s_d6_VCT'%(tag,run), OutputWorkspace=wsIn)
#      wsOut = CloneWorkspace('CloneOfInputData_d6')
      keep_ws = CloneWorkspace(wsIn)
      xIn = wsIn.readX(0) #x-values of 0th histogram 
      yTmp = wsIn.readY(0)
      nPts = yTmp.size
      nHst = wsIn.getNumberHistograms()
      yOut = np.zeros([nHst,nPts]) # empty array to store modified data
      eOut = np.zeros([nHst,nPts]) # empty array to store modified data
      clr = np.zeros([nHst,nPts]) # empty array to store modified data
#      yIn = wsIn.extractY()
#      eIn = wsIn.extractE()
      for i in range(nHst):
            xIn = wsIn.readX(i)
            yIn = wsIn.readY(i)
            eIn = wsIn.readE(i)
            nrmIndx = np.where(np.logical_and(xIn>=nrmTransMinD, xIn<=nrmTransMaxD)) #indices for normalisation
            normFactor = np.average(yIn[nrmIndx])
            #for j in range(len(nrmIndx)):
            #    print(j,nrmIndx[j],xIn[nrmIndx[j]],yIn[nrmIndx[j]])
            #print('histo:',i,' nrm factor: ',normFactor) 
            yOut[i,] = yIn/normFactor
            #outside usable range, just set correction equal to 1.0
            print('for hist: ',i,' dmin is: ',dLim[i,0],' dmax is: ',dLim[i,1])
            print('setting these d-values to one: ')
            NotUsefulIndx = np.where(np.logical_or(xIn<=dLim[i,0],xIn>=dLim[i,1])) #indices outside useful range
            print(len(NotUsefulIndx[0]))
            print('In Hist: ',i,' found ',NotUsefulIndx[0].size,' points that are out of useful range')
            yOut[i,NotUsefulIndx[0][0:-1]]=1.0
      wsOut = CloneWorkspace('%s%s_d6_VCT'%(tag,run))
      for i in range(nHst):
            wsOut.setY(i,yOut[i,])
            wsOut.setE(i,eOut[i,]) #errors are just zeros at the moment
      RenameWorkspace(InputWorkspace='wsOut', OutputWorkspace='%s%s_d6_attcorr'%(tag,run))
#    except:
#      print('Diamond transmission data do not exist. Generate this using FitTransPy')
        
