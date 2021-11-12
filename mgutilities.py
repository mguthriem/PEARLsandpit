# import mantid algorithms, numpy and matplotlib
#from mantid.simpleapi import *
#import matplotlib.pyplot as plt
#import numpy as np
from mantid.simpleapi import *
import json
import numpy as np
import os

def d2Q(listOfd):
    #converts a list of d-spacings to Q and returns these. Should be able to handle multi-dimensional
    inQ = [2*np.pi/x for x in listOfd]
    return inQ

def workbench_input_fn(dTitle,dInstruction,inpType):
  from qtpy.QtWidgets import QInputDialog
  
  if inpType=='int':
    item, ok = QInputDialog.getInt(None, dTitle, dInstruction)
  elif inpType=='str': 
    item, ok = QInputDialog.getText(None, dTitle, dInstruction)
    
  if ok:
      return item
  else:
      raise ValueError("Error retrieving input")

def gridPlot(WSNames,xlims,SpectrumMap,inLegend,TickMarks,ROILims,plotName):

  # a script to automate plotting of multiple histograms in a single window
  # driver is to separately inspect the 6 columns on SNAP without the
  # unweildy manipulation of 6 different windows on the display   

  # import mantid algorithms, numpy and matplotlib

  #WSName is a list of one or more mantid workspace names to be plotted
  #WSs must have same number of spectra and these will be plotted according to the same
  #specified x-axis limits

  #xlims is a 2xnhst list of xlimits (min and max for each histogram in WS)

  #spectrumMap is a multi-dimensional list representing how spectra should be arranged.
  #e.g. if spectrumMap = [[1,2,3],[4,5,6]] display will contain 2 rows and 3 columns. The first
  # row will show spectra 1,2 and 3 from left to right and the bottom row will show spectra 4,5&6
  # I allowed for the possibility that number of columns in each row are different 

  #inLegend is a list of labels to use for the WS and should have equal length to WSNames

  #TickMarks is a list of d-spacings to plot as vertical lines on all plots

  #ROILims is a simple marker showing minimum and maximum limits on ROI on the x-axis it requires a
  #pair of limits for each histogram in the plot

  from mantid.simpleapi import mtd
  import matplotlib.pyplot as plt
  import numpy as np
  from mantid.api import AnalysisDataService as ADS

  #print('version is 1.0')
  nWS = len(WSNames) #number of ws to plot
  ws = mtd[WSNames[0]]# first ws in list is considered the reference and all other ws 
  #must have matching number of histograms and matching x-axis units
  
  #print('inLegend:',inLegend,'len is:',len(inLegend))
  #print('WSNames: ',WSNames,'len is:',len(WSNames))
  #print('These are equal',len(inLegend)==len(WSNames))
  
  #print('In gridPlot')
  refnHst = ws.getNumberHistograms()
  refXAxisCaption = ws.getAxis(0).getUnit().caption()
  for i in range(nWS):
    ws = mtd[WSNames[i]]
    nHst = ws.getNumberHistograms()
    XAxisCaption = ws.getAxis(0).getUnit().caption()
    #print(i,'histograms:',nHst,'caption:',XAxisCaption)

  if plt.fignum_exists(plotName): 
    plt.close(plotName) #get rid of window if it already exists
  
  #########################################################################
  # plotting data 
  #########################################################################

  nrows = len(SpectrumMap)
  ncols = [None]*nrows
  for i in range(nrows):
    ncols[i] = len(SpectrumMap[i])
  maxCols = max(ncols)
  
  ColorPalette = ['#2ca02c','#ef2929','#3465a4','#9467bd','#f57900','#8f5902','#000000','#a40000','#888a85','#4e9a06']
  
  #create figure and axes to plot data in
  fig, axes = plt.subplots(figsize=[10, 6.5258],\
  nrows=nrows, ncols=maxCols, num=plotName, \
  #sharex = True,\
  subplot_kw={'projection': 'mantid'})
  #loop through data to plot
  wslab = 0
  for ws in WSNames:
    inWSName = ws
    #print('plotting:',ws)
    wsIn = mtd[ws]
    nhst = wsIn.getNumberHistograms()
    try:
      sampleColor = ColorPalette[wslab]
    except:
      sampleColor = ['#000000']
      print('only 10 colours defined and more than 10 overlays requested. Additional will be black')
    
    
    axisXLabel = wsIn.getAxis(0).getUnit().caption()
    axisYLabel = wsIn.getAxis(1).getUnit().caption()

    sample = ADS.retrieve(inWSName) # vanadium corrected sample diffraction data for all columns
    for i in range(nrows):
      for j in range(ncols[i]):
        SpecIndex = SpectrumMap[i][j]
        if len(inLegend)==len(WSNames):
          axes[i][j].plot(sample, color=sampleColor, wkspIndex=SpecIndex-1, label=inLegend[wslab])
        else:
          axes[i][j].plot(sample, color=sampleColor, wkspIndex=SpecIndex-1)
        subPlotYLims = axes[i][j].get_ylim()
        subPlotXLims = axes[i][j].get_xlim()
        if len(TickMarks)!=0:
          for k in range(len(TickMarks)):
            axes[i][j].plot([TickMarks[k],TickMarks[k]],[subPlotYLims[0],subPlotYLims[1]],color='#888a85')
        if len(ROILims)!=0:
          ROImin = ROILims[SpecIndex-1][0]
          ROImax = ROILims[SpecIndex-1][1]
          #print('Spec:',SpecIndex,'ROI lims:',ROImin,ROImax)
          axes[i][j].plot([ROImin,ROImin],[subPlotYLims[0],subPlotYLims[1],],color='#000000',linestyle='--', linewidth=0.5)
          axes[i][j].plot([ROImax,ROImax],[subPlotYLims[0],subPlotYLims[1],],color='#000000',linestyle='--', linewidth=0.5)
        axes[i][j].set_title('Spec '+str(SpectrumMap[i][j]))
        axes[i][j].set_xlim(subPlotXLims[0],subPlotXLims[1])
    wslab = wslab + 1
  
  
  #set axis ranges and labels
  
  if len(xlims) != 0:
   for i in range(nrows):
     for j in range(maxCols):
        axes[i][j].set_xlim(xlims[0], xlims[1])
        #axes[i][j].set_ylim(0.0,250)

  for i in range(nrows):
    for j in range(maxCols):
      axes[i][0].set_ylabel(axisYLabel)
      axes[i][1].set_ylabel('')
      axes[i][2].set_ylabel('')

  for j in range(maxCols):
    axes[0][j].set_xlabel('')
    axes[1][j].set_xlabel(axisXLabel)

  legend = axes[0][0].legend(fontsize=8.0).draggable().legend
  plt.show()    
  return

def genHstNameLst(rootName,nHst):
  str = '%s%s,'%(rootName,0)
  for i in range(nHst-2):
    str = str+'%s%s,'%(rootName,i+1)
  str = str + '%s%s'%(rootName,nHst-1)
  return str

def getConfigDict(FName):
    #print('attempting to open file:',FName)
    if os.path.exists(FName):
        with open(FName, "r") as json_file:
            dictIn = json.load(json_file)
        #print('got config dictionary')
        return dictIn
    else:
        print('file not found')

def loadAndPrep(run,msknm,configDict,mode,modeSet): #initial steps of reduction
  #preliminary normalisation (to proton charge)
  #calibration (currently via detCal or modifying sample logs)
  
  #run = INT a single run number
  #msknm = STRING the name of a mask. If empty, no mask is applied.
  #configDict = DICT dictionary containing the current configuration
  #mode = INT a historical setting controlling how data are processed (TO BE REMOVED AT SOME POINT SOON)
  #modeSet = LIST of INT is also enables passing of options controlling how data are proceed:
      #modeSet[0] = Apply vanadium correction [NOT USED HERE]
      #modeSet[1] = Mask: 0 = don't use mask; 1 use mask in workspace; 2 use mask read from file
      #modeSet[2] = 1 load monitor data, = 0 don't load monitor data 
      #modeSet[3] = label for location of neutron data either 0 = dataDir,1  = VDir or 2 = VBDir 

  tag = configDict['instrumentTag']
  #sharedDir = configDict['sharedDir']
  extn = configDict['extn']
  MonTBinning = configDict['MonTBinning']
  TBinning = configDict['TBinning']
  tlims = TBinning.split(',')
  tof_min = tlims[0]
  tof_max = tlims[2]
  detCalName = configDict['detCalName']
  detLogVal = configDict['detLogVal']
  MonTBinning = configDict['MonTBinning']

  # Set up column grouping workspace if it doesn't yet exist
  try:
      a = mtd['SNAPColGp'] #if workspace already exists, don't reload
  except:
      CreateGroupingWorkspace(InstrumentFilename=configDict['instFileLoc'], \
      GroupDetectorsBy='Column', OutputWorkspace='SNAPColGp')
  
  if modeSet[3]==0:
      dataDir = configDict['dataDir']
  elif modeSet[3]==1:
      dataDir = configDict['VDir']
  elif modeSet[3]==2:
      dataDir = configDict['VBDir']

  #BASIC LOADING AND NORMALISATION
  if modeSet[2]==1:
      try:
        mtd['%s%s'%(tag,run)] #try/except prevents re-loading of event data if ws already exists
      except:
        LoadEventNexus(Filename=r'%s%s_%s.%s'%(dataDir,tag,run,extn),OutputWorkspace='%s%s'%(tag,run),
        FilterByTofMin=tof_min, FilterByTofMax=tof_max, Precount='1', LoadMonitors=True)
        NormaliseByCurrent(InputWorkspace='%s%s'%(tag,run),OutputWorkspace='%s%s'%(tag,run))  
        CompressEvents(InputWorkspace='%s%s'%(tag,run),OutputWorkspace='%s%s'%(tag,run))
        Rebin(InputWorkspace='%s%s'%(tag,run),OutputWorkspace='%s%s'%(tag,run),Params=TBinning,FullBinsOnly=True)
        NormaliseByCurrent(InputWorkspace='%s%s_monitors'%(tag,run),OutputWorkspace='%s%s_monitors'%(tag,run))
        Rebin(InputWorkspace='%s%s_monitors'%(tag,run),OutputWorkspace='%s%s_monitors'%(tag,run),Params=MonTBinning,FullBinsOnly=True)
  elif modeSet[2] == 0:
      try:
        mtd['%s%s'%(tag,run)]
      except:
        LoadEventNexus(Filename=r'%s%s_%s.%s'%(dataDir,tag,run,extn),OutputWorkspace='%s%s'%(tag,run),
        FilterByTofMin=tof_min, FilterByTofMax=tof_max, Precount='1', LoadMonitors=False)
  #POSTIONAL CALIBRATION =(currently via either an ISAW detcal or changelogs)
  if detCalName.lower() == 'changelogs':
    AddSampleLog(Workspace='SNAP%s'%run,LogName='det_arc1',LogText=detLogVal.split(",")[0],LogType='Number Series')
    AddSampleLog(Workspace='SNAP%s'%run,LogName='det_lin1',LogText=detLogVal.split(",")[1],LogType='Number Series')
    AddSampleLog(Workspace='SNAP%s'%run,LogName='det_arc2',LogText=detLogVal.split(",")[2],LogType='Number Series')
    AddSampleLog(Workspace='SNAP%s'%run,LogName='det_lin2',LogText=detLogVal.split(",")[3],LogType='Number Series')
    LoadInstrument(Workspace='SNAP%s'%run,MonitorList='-1,1179648', RewriteSpectraMap='False',InstrumentName='SNAP')
  elif (detCalName.lower() != 'none' and detCalName.lower() !='changelogs'):
    LoadIsawDetCal(InputWorkspace='%s%s'%(tag,run), Filename=detCalName)
  
  #MASK DETECTORS (Need to do prior to any groupoing if mask read from file
  CloneWorkspace(InputWorkspace='%s%s'%(tag,run),OutputWorkspace='%s%s_msk'%(tag,run))
  if modeSet[1] == 2 and mode !=1: # mask detectors here if mask read in from file (currently always the case for mode 2 and 3
      MaskDetectors(Workspace='%s%s_msk'%(tag,run),MaskedWorkspace='%s'%(msknm))
  return

def generateVCorr(msknm,mskLoc,configDict,inQ,showFit): #generates vanadium correction from two runs and metadata in configDict


  #msknm = STRING can either be a file name (without the assumed XML extension, which MUST be stored in the shared folder)
  #   or it can be the name of a pre-stored mantid workspace containing a mask
  #   or it can be an empty string if no mask is needed
  #mskLoc = INT = 0 don't use mask, = 1 mask is in workspace, = 2 mask is stored in file in shared directory
  #configDict = DICT contains configuration.
  #inQ = INT = 1 output in Q, otherwise will output in d-spacing
  #showfit = INT = 1 shows progress using gridPlot

  import sys

  vanPeakFWHM = configDict['vanPeakFWHM'].split(',')
  vanPeakTol = configDict['vanPeakTol'].split(',')
  vanHeight = configDict['vanHeight']
  vanRadius = configDict['vanRadius']
  vanSmoothing = configDict['vanSmoothing'].split(',')
  VPeaks = configDict['VPeaks']
  TBinning = configDict['TBinning']
  tlims = TBinning.split(',')
  tof_min = tlims[0]
  tof_max = tlims[2]
  tag = configDict['instrumentTag']
  dminsStr = configDict['dmins'].split(',')
  dmaxsStr = configDict['dmaxs'].split(',')
  dmins = [float(x) for x in dminsStr]
  dmaxs = [float(x) for x in dmaxsStr]
  c = VPeaks.split(',')
  allPeaksToStrip = [float(x) for x in c]
  dBinSize = configDict['dBinSize']
  QBinSize = configDict['QBinSize']

  #print(allPeaksToStrip)
  #There are some costly steps in here, which can be ignored if there is no change in masking
  #this will definitely be the case when vanadium parameters are being optimised during set up
  #
  #this is likely the case when showFit == 1 so use this as a flag to speed up where possible.
  #Also using showFit to control interactions: 
  #showFit =2 - strip peaks
  #        =3 - set smoothing

  Vrun = configDict['Vrun']
  VDataDir = configDict['VDir']
  VBrun = configDict['VBrun']
  VBDataDir = configDict['VBDir']
  #mskloc = configDict['mskloc']
  #msknm = configDict['msknm']
  dStart = min(dmins)
  dEnd = max(dmaxs)
  QStart = 2*np.pi/dEnd
  QEnd = 2*np.pi/dStart
  print('ragged parameters:')
  RebinParams = str(dStart)+','+ dBinSize + ',' + str(dEnd)
  print('d-space binning parameters are:',RebinParams)

  
  if showFit >= 1 and mtd.doesExist('VCorr_VmB'):
    pass
  else:
    loadAndPrep(Vrun,msknm,configDict,2,[0,mskLoc,1,1])
    ConvertUnits(InputWorkspace='%s%s_msk'%(tag,Vrun),OutputWorkspace='%s%s_d'%(tag,Vrun),Target='dSpacing')
    DiffractionFocussing(InputWorkspace='%s%s_d'%(tag,Vrun), OutputWorkspace='%s%s_d6'%(tag,Vrun), GroupingWorkspace='SNAPColGp')
    loadAndPrep(VBrun,msknm,configDict,2,[0,mskLoc,1,2])
    ConvertUnits(InputWorkspace='%s%s_msk'%(tag,VBrun),OutputWorkspace='%s%s_d'%(tag,VBrun),Target='dSpacing')
    DiffractionFocussing(InputWorkspace='%s%s_d'%(tag,VBrun), OutputWorkspace='%s%s_d6'%(tag,VBrun), GroupingWorkspace='SNAPColGp')
    Minus(LHSWorkspace='SNAP'+str(Vrun)+'_d6', RHSWorkspace='SNAP'+str(VBrun)+'_d6', OutputWorkspace='VCorr_VmB')
  ws = mtd['SNAP'+str(Vrun)+'_d6']
  #Conduct angle dependent corrections on a per-spectrum basis
  nHst = ws.getNumberHistograms()
  for i in range(nHst):
    #print('working on spectrum:',i+1)
    ExtractSingleSpectrum(InputWorkspace='VCorr_VmB',OutputWorkspace='VCorr_VmB%s'%(i),WorkspaceIndex=i)
    if len(VPeaks) != 0:       
        StripPeaks(InputWorkspace='VCorr_VmB%s'%(i), OutputWorkspace='VCorr_VmB_strp%s'%(i), \
        FWHM=vanPeakFWHM[i], PeakPositions=VPeaks, PeakPositionTolerance=vanPeakTol[i])
    FFTSmooth(InputWorkspace='VCorr_VmB_strp%s'%(i),\
                         OutputWorkspace='VCorr_VmB_strp_sm%s'%(i),\
                         Filter="Butterworth",\
                         Params=str(vanSmoothing[i]+',2'),\
                         IgnoreXBins=True,
                         AllSpectra=True)
    SetSampleMaterial(InputWorkspace='VCorr_VmB_strp_sm%s'%(i), ChemicalFormula='V')
    ConvertUnits(InputWorkspace='VCorr_VmB_strp_sm%s'%(i), OutputWorkspace='VCorr_VmB_strp_sm%s'%(i), Target='Wavelength')
    CylinderAbsorption(InputWorkspace='VCorr_VmB_strp_sm%s'%(i), OutputWorkspace='VCorr_a', AttenuationXSection=5.08, \
    ScatteringXSection=5.10, CylinderSampleHeight=vanHeight, CylinderSampleRadius=vanRadius, CylinderAxis='0,0,1')
    Divide(LHSWorkspace='VCorr_VmB_strp_sm%s'%(i), RHSWorkspace='VCorr_a', OutputWorkspace='VCorr_VmB_strp_sm_a%s'%(i))
    ConvertUnits(InputWorkspace='VCorr_VmB_strp_sm_a%s'%(i), OutputWorkspace='VCorr_VmB_strp_sm_a%s'%(i), Target='dSpacing')
    Rebin(InputWorkspace='VCorr_VmB_strp_sm_a%s'%(i), OutputWorkspace='VCorr_VmB_strp_sm_a%s'%(i), Params=RebinParams,FullBinsOnly=True)

  #conjoin original spectra into a single workspace again
  root = 'VCorr_VmB_strp'
  ConjoinSpectra(InputWorkspaces=genHstNameLst(root,6), OutputWorkspace=root, LabelUsing=genHstNameLst('spec ',6))
  DeleteWorkspaces(genHstNameLst(root,6))
  root = 'VCorr_VmB_strp_sm'
  ConjoinSpectra(InputWorkspaces=genHstNameLst(root,6), OutputWorkspace=root)
  DeleteWorkspaces(genHstNameLst(root,6))
  root = 'VCorr_VmB_strp_sm_a'
  ConjoinSpectra(InputWorkspaces=genHstNameLst(root,6), OutputWorkspace='VCorr_VmB_strp_sm_a_afterConjoin')
  #DeleteWorkspaces(genHstNameLst(root,6))
  root = 'VCorr_VmB'
  DeleteWorkspaces(genHstNameLst(root,6))
  DeleteWorkspace(Workspace='VCorr_a')

  #ConvertUnits(InputWorkspace='VCorr_VmB',OutputWorkspace='VCorr_VmB', Target='dSpacing')
  #ConvertUnits(InputWorkspace='VCorr_VmB_strp',OutputWorkspace='VCorr_VmB_strp', Target='dSpacing')
  ConvertUnits(InputWorkspace='VCorr_VmB_strp_sm',OutputWorkspace='VCorr_VmB_strp_sm',Target='dSpacing')
  #ConvertUnits(InputWorkspace='VCorr_VmB_strp_sm_a',OutputWorkspace='VCorr_VmB_strp_sm_a', Target='dSpacing')
  
  ROILims=[ [dmins[0],dmaxs[0]],[dmins[1],dmaxs[1]],[dmins[2],dmaxs[2]],[dmins[3],dmaxs[3]],[dmins[4],dmaxs[4]],[dmins[5],dmaxs[5]] ]
  if inQ ==1:
      
      allPeaksToStripQ = d2Q(allPeaksToStrip)
      for i in range(len(ROILims)):
            ROILims[i] = [2*np.pi/x for x in ROILims[i]] #convert ROI limits to Q
      ConvertUnits(InputWorkspace='VCorr_VmB', OutputWorkspace='VCorr_VmB_Q', Target='MomentumTransfer')
      ConvertUnits(InputWorkspace='VCorr_VmB_strp', OutputWorkspace='VCorr_VmB_strp_Q', Target='MomentumTransfer')
      ConvertUnits(InputWorkspace='VCorr_VmB_strp_sm', OutputWorkspace='VCorr_VmB_strp_sm_Q', Target='MomentumTransfer')
      ConvertUnits(InputWorkspace='VCorr_VmB_strp_sm_a', OutputWorkspace='VCorr_VmB_strp_sm_a_Q', Target='MomentumTransfer')
      if showFit == 2:
        gridPlot(['VCorr_VmB_Q','VCorr_VmB_strp_Q','VCorr_VmB_strp_sm_Q'],[],[[1,2,3],[4,5,6]],['Raw','Peaks stripped','Smoothed'],allPeaksToStripQ,ROILims,'Vanadium Setup')
      elif showFit == 3:
        gridPlot(['VCorr_VmB_strp_Q','VCorr_VmB_strp_sm_Q'],[],[[1,2,3],[4,5,6]],['Peaks stripped','smoothed'],[],ROILims,'Vanadium Setup')
      elif showFit == 4:
        gridPlot(['VCorr_VmB_strp_sm_Q','VCorr_VmB_strp_sm_a_Q'],[],[[1,2,3],[4,5,6]],['Smoothed','Att corrected'],[],[],'Vanadium Setup')
      else:
        DeleteWorkspace(Workspace='VCorr_VmB')
        DeleteWorkspace(Workspace='VCorr_VmB_strp')
        DeleteWorkspace(Workspace='VCorr_VmB_strp_sm')
        DeleteWorkspace(Workspace='VCorr_VmB_strp_sm_a')
  else:
    if showFit == 2:
        gridPlot(['VCorr_VmB','VCorr_VmB_strp','VCorr_VmB_strp_sm'],[],[[1,2,3],[4,5,6]],['Raw','Peaks stripped','Smoothed'],allPeaksToStrip,ROILims,'Vanadium Setup')
    elif showFit == 3:
        ROILims=[ [dmins[0],dmaxs[0]],[dmins[1],dmaxs[1]],[dmins[2],dmaxs[2]],[dmins[3],dmaxs[3]],[dmins[4],dmaxs[4]],[dmins[5],dmaxs[5]] ]
        gridPlot(['VCorr_VmB_strp','VCorr_VmB_strp_sm'],[],[[1,2,3],[4,5,6]],['Peaks stripped','smoothed'],[],ROILims,'Vanadium Setup')
    elif showFit == 4:
        gridPlot(['VCorr_VmB_strp_sm','VCorr_VmB_strp_sm_a'],[],[[1,2,3],[4,5,6]],['Smoothed','Att corrected'],[],'Vanadium Setup')
    else:
        DeleteWorkspace(Workspace='VCorr_VmB')
        DeleteWorkspace(Workspace='VCorr_VmB_strp')
        DeleteWorkspace(Workspace='VCorr_VmB_strp_sm')
        DeleteWorkspace(Workspace='VCorr_VmB_strp_sm_a')

  
  for i in range(nHst):
    print(i,dmins[i],dmaxs[i])
  CropWorkspaceRagged(InputWorkspace='VCorr_VmB_strp_sm_a_afterConjoin',OutputWorkspace='VCorr_d06',Xmin=dmins,Xmax=dmaxs)

  ws=mtd['VCorr_d06']
  ax = ws.getAxis(1)
  for i in range(nHst):
    ax.setLabel(i,'Spec %s'%(i+1))
  #Rebin operation is creating artifacts at the edge of spectra where data drops to zero. 
  #SumSpectra(InputWorkspace='VCorr_d06', OutputWorkspace='VCorr_dEast', ListOfWorkspaceIndices='0-2')
  #SumSpectra(InputWorkspace='VCorr_d06', OutputWorkspace='VCorr_dWest', ListOfWorkspaceIndices='3-5')
  #AppendSpectra(InputWorkspace1='VCorr_dEast', InputWorkspace2='VCorr_dWest', OutputWorkspace='VCorr_dEastWest')
  #DeleteWorkspace(Workspace='VCorr_dEast')
  #DeleteWorkspace(Workspace='VCorr_dWest')
  #SumSpectra(InputWorkspace='VCorr_d06', OutputWorkspace='VCorr_dAll', ListOfWorkspaceIndices='0-5')
  
  if inQ ==1:
      RebinParams = str(QStart)+','+ QBinSize + ',' + str(QEnd)
      print('d-space binning parameters are:',RebinParams)
      Rebin(InputWorkspace='VCorr_VmB_strp_sm_a_Q', OutputWorkspace='VCorr_Q6', Params=RebinParams,FullBinsOnly=True)
      SumSpectra(InputWorkspace='VCorr_Q6', OutputWorkspace='VCorr_QEast', ListOfWorkspaceIndices='0-2')
      SumSpectra(InputWorkspace='VCorr_Q6', OutputWorkspace='VCorr_QWest', ListOfWorkspaceIndices='3-5')
      AppendSpectra(InputWorkspace1='VCorr_QEast', InputWorkspace2='VCorr_QWest', OutputWorkspace='VCorr_QEastWest')
      DeleteWorkspace(Workspace='VCorr_QEast')
      DeleteWorkspace(Workspace='VCorr_QWest')
      SumSpectra(InputWorkspace='VCorr_Q6', OutputWorkspace='VCorr_QAll', ListOfWorkspaceIndices='0-5')
  return

#def GetConfig(ConfigFileName):
#    with open(ConfigFileName, "r") as json_file:
#        dictIn = json.load(json_file)
#    return dictIn

def iceVol(a):

#UNTITLED calculates ice VII molar volume based on lattice parameter 
# if lattice negative is given as negative number, assumes it's the 110
# then gives estimate of pressure based on fit to Hemley EOS 1987
  if a>=0:
    vm = a**3*0.5*0.6022
  elif a<0:
    a = -np.sqrt(2)*a
    vm = a**3*0.5*0.6022

  ad = 6/a**3; # 6 atoms in unit cell

  #print('molar volume is : ' +str(vm)+ ' cm^3')
 # disp(['atom. density is: ' +str(ad) + ' atom/A^3'])


  p_hem = 1490.05895-558.82463*vm+81.14429*vm**2-5.3341*vm**3+0.13258*vm**4
  p_zul = 2202.87622-898.92604*vm+141.2403*vm**2-10.02239*vm**3+0.26903*vm**4
  #print('*correct* Pressure (Hemley EOS): ' + str(p_hem) + ' GPa')
  #print('*correct* Pressure (Zulu radial EOS): ' + str(p_zul) + ' GPa')

  return p_hem

def getSNAPStateID(arc1,arc2,wav,guide,spare):

    import glob
    import os
    import os.path, time
    from datetime import datetime
    #Need a function that will return instrument state ID a short string of integers
    #the main experimental parameters of the SNAP instrument, which must be provided
    #the numbers needed are;
    #  arc1 = EAST DETECTOR BANK ANGLE (deg)
    #  arc2 = WEST DETECTOR BANK ANGLE (deg)
    #  wav = wavelength setting (Angstrom)
    # guide = guide status 1 = in 0 = out
    # spare == 0 is an extra parameter in case we need it in the future (maybe SE?)

    # The SNAP state definitions live in a simple csv file stored in SNS/SNAP/shared/Calibration
    # The name is SNAPStateListYYYYMMDD.csv
    # This function shall always choose the most recent. This allow for the possibility that a 
    # new state can be defined at any time and added to the list
    # 

    # First task is to find the most recent StateList
    pattern = '/SNS/SNAP/shared/Calibration/SNAPStateList*.csv'
    FindMostRecent = False

    refDate = datetime.now().timestamp()

    for fname in glob.glob(pattern, recursive=True):
        ShortestTimeDifference = 10000000000 # a large number of seconds
        if os.path.isfile(fname):
            #rint(fname)
            #print("Created: %s" % time.ctime(os.path.getctime(fname)))
            #print('epoch:',os.path.getctime(fname))
            #print('refdate epoch:',refDate)
            delta = refDate - os.path.getctime(fname)
            #print('difference:',delta)
            if delta <= ShortestTimeDifference:
                MostRecentFile = fname
                ShortestTimeDifference = delta
    if ShortestTimeDifference == 10000000000:
        print('no matching file found')
    else:
        print('Most recent matching file:',fname)
        print('Created: %s'% time.ctime(os.path.getctime(fname)))
        print('Will use this one')
            #print(refDate-vvos.path.getctime(fname))
            #timestr = SetUpDate.strftime("_%d-%b-%Y-%H%M%S")
   # states = np.loadtxt(fname,delimiter=",", skiprows = 6,usecols=[1:])# array of floats
    #refState = np.array([arc1,arc2,wav,guide,spare])
    #delta = states-refState
    #print(delta)
    #print(np.sum(delta,axis=0))
    #print(np.sum(delta,axis=1))

