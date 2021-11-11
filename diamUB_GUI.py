#Version: 20201214 Original script:diamUBQt20201214
###############################################################################################
#diamUB is used to calculate the UB matrices corresponding to the two diamond anvils that are always present in a DAC measurement
#A prerequisite is a that a SingleCrystalPeaksWorkspace exists with the (unindexed) diamond reflections in it. 
#For each diamond, a UB is calculated using two observed reflections. This UB is then used to index all remaining reflections and, iff
# there are enough (>3) reflections, this will be refined
#The resulting UB's are written to ISAW format files
#
#M. Guthrie 15 Jan 2020
#
#modifiations
#20200309  added automatic peak search
#20200422 .added Antonio's peak search suggestion (FindPeaksMD)
#20200615 encountered issue with convertToMD, which is creating an empty MDworkspace and subsequently, no peaks found
#have added back the ability to use a manual peak table
#20200825 allowed indexing of some shorter d-spacing peaks...but still excluding the 511/333 peaks
#20201214 added (clumsy) correction for writing UB file: user specifies IPTS and it assumes SNS folder structure
#20201214 also realised that setUB ignores lattice parameters and added a "findUBUsingLatticeParameters to force that these
#stay cubic as they should be. There might be some unintended consequences of this, but it was necessary to fit the data.

#############################################################################################
# The following line helps with future compatibility with Python 3
# print must now be used as a function, e.g print('Hello','World')
from __future__ import (absolute_import, division, print_function, unicode_literals)
# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
from mantidqt.utils.qt.qappthreadcall import QAppThreadCall
import matplotlib.pyplot as plt
import sys
import numpy as np
if sys.version_info > (3,):
    if sys.version_info < (3,4):
        from imp import reload
    else:
        from importlib import reload
        
        
import os

print('getcwd:      ', os.getcwd())
print('__file__:    ', __file__)        
#Import the various functions needed to perform calculations
import diamUBFunctionsQt20210114 as diamUBFunctions
reload(diamUBFunctions)
input = QAppThreadCall(diamUBFunctions.workbench_input_fn)
#start malcolm's code
insName = 'SNAP'
IPTS = input('File output: ','Enter IPTS number: ', 'str')
runName = input('File output: ','Enter run number: ', 'str')
manMode = input('Choose mode: ','Attempt automatic peak search? (y/[n]) (if not manually created SingleCrystalPeakTable must exist)','str')
if manMode.lower()=='y':
    dens_thres = 400
    wks_d = insName+runName+'_d_8x8_noMsk' #note this has had a sumNeighbours run
    wks_test = insName+runName+'_d_8x8_test_MD'
    wks_pks = insName+runName+'_d_8x8_test_pks'
    wks_int = insName+runName+'_d_8x8_test_pks_int'
    wks_filt = insName+runName+'_d_8x8_test_pks_int_filt'
    try:
        a = mtd[wks_test] # don't recreate MD if it already exists
    except:
        ConvertToMD(InputWorkspace=wks_d, QDimensions='Q3D', dEAnalysisMode='Elastic', Q3DFrames='Q_lab', OutputWorkspace=wks_test)
    FindPeaksMD(InputWorkspace=wks_test, PeakDistanceThreshold=0.25, MaxPeaks=50, DensityThresholdFactor=dens_thres , OutputWorkspace=wks_pks)
    IntegratePeaksMD(InputWorkspace=wks_test, PeakRadius=0.12, BackgroundInnerRadius=0.14, BackgroundOuterRadius=0.17, PeaksWorkspace=wks_pks, OutputWorkspace=wks_int)
    FilterPeaks(InputWorkspace=wks_int, OutputWorkspace=wks_filt, FilterVariable='Signal/Noise', FilterValue=40, Operator='>')
    sxlLst = mtd[wks_filt]
else:
    wks_filt = 'SingleCrystalPeakTable'
    sxlLst = mtd[wks_filt]
#SetGoniometer(Workspace=wks_filt, Axis='omega, 0,1,0,1')# magical step
npk = sxlLst.getNumberPeaks()
print('Read: ', npk, ' peaks')
purgePks = 1
#purgePks = input('Enter 1 to set all existing hkl to 0: ')
#UB calculation uses d-spacing and Q-vectors of peaks, so read all of these...
d = np.zeros((npk,));
Q = np.zeros((npk,3));
pkInt = np.zeros((npk,));
pkLam = np.zeros((npk,));
for i in range(npk):
    p = sxlLst.getPeak(i)
    d[i] = p.getDSpacing()
    Q[i,:] = p.getQLabFrame()
    pkInt[i] = p.getIntensity()
    pkLam[i] = p.getWavelength()
    if purgePks==1:
        p.setHKL(0,0,0)

# make copies of peaks workspace to separately store the two UB matrices in 
#CloneWorkspace(InputWorkspace='SingleCrystalPeakTable', OutputWorkspace='diamUB1')
#CloneWorkspace(InputWorkspace='SingleCrystalPeakTable', OutputWorkspace='diamUB2')    

#############
#Call indexing program

UB1,UB2,ubLab,Hr1,Hr2,dOut,QOut,LamOut,intenOut = diamUBFunctions.findDiamUB(d,Q,pkLam,pkInt)
nOut = dOut.size

UB1 = UB1/(2*np.pi)
UB2 = UB2/(2*np.pi)
print('UB has been divided by 2pi to match ISAW standard')

#IPTS = 24179
#locations on analysis cluster
sharedDir = '/SNS/snfs1/instruments/SNAP/IPTS-'+str(IPTS)+'/shared/' #equivalent to "shared" directory in SNS file structure  
FullOName1 = sharedDir + insName + runName + 'UB1.mat'
FullOName2 = sharedDir + insName + runName + 'UB2.mat'
print('Will write UB1 to: ',FullOName1)
print('Will write UB2 to: ',FullOName2)
#run through peaks list, creating a workspace to store each set of indexed peaks
#if sufficient indexed peaks exist, then UB will be refined and peaks reindexed 

#actions depend on number of peaks that have been indexed on each diamond, so determine these
nInd = [0,0]
for i in range(nOut):
    if ubLab[i]==1:
        nInd[0] = nInd[0] + 1
    elif ubLab[i]==2:
        nInd[1] = nInd[1] + 1
#sequentially process diamonds:
thisDiam = 0 #corresponds to diamond 1
print('working on diamond',thisDiam+1,' with: ',nInd[thisDiam],' peaks')
if nInd[thisDiam]==0:
    print('no indexed peaks found for diamond 1, skipping')
else: #
    UBlst = [UB1[0,0],UB1[0,1],UB1[0,2],UB1[1,0],UB1[1,1],UB1[1,2],UB1[2,0],UB1[2,1],UB1[2,2]]
    #UBlst = np.zeros([3,3])
    wsName1 = insName+runName+'diamUB'+str(thisDiam+1)
    wsHandle = CreatePeaksWorkspace(InstrumentWorkspace=wks_filt,NumberOfPeaks=0,OutputWorkspace=wsName1) #NumberOfPeaks=nInd[thisDiam],create empty peaks workspace
    ClearUB(Workspace=wsName1)
    SetUB(Workspace=wsName1, a=3.567, b=3.567, c=3.567, UB=UBlst)
    pws = mtd[wsName1]
    UBinit1 = np.array(pws.sample().getOrientedLattice().getUB())
    for i in range(nOut):
        if ubLab[i]==1:  
            print('Peak: %2i hkl: %6.3f %6.3f %6.3f lambda: %7.4f d-spacing: %7.4f Q: %6.3f %6.3f %6.3f Int: %7.3f'%(i,
            Hr1[i,0],Hr1[i,1],Hr1[i,2],LamOut[i],dOut[i],QOut[i,0],QOut[i,1],QOut[i,2],intenOut[i])) 
            p = wsHandle.createPeak(QOut[i,])
            AddPeakHKL(wsHandle,[Hr1[i,0],Hr1[i,1],Hr1[i,2]])
            p.setWavelength(LamOut[i])


thisDiam = 1 #corresponds to diamond 2
print('working on diamond',thisDiam+1,' with: ',nInd[thisDiam],' peaks')
if nInd[thisDiam]==0:
    print('no indexed peaks found for diamond 2, skipping')
else: #
    UBlst = [UB2[0,0],UB2[0,1],UB2[0,2],UB2[1,0],UB2[1,1],UB2[1,2],UB2[2,0],UB2[2,1],UB2[2,2]]
    wsName2 = insName+runName+'diamUB'+str(thisDiam+1)
    wsHandle = CreatePeaksWorkspace(InstrumentWorkspace=wks_filt,NumberOfPeaks=0,OutputWorkspace=wsName2) #NumberOfPeaks=nInd[thisDiam],create empty peaks workspace
    ClearUB(Workspace=wsName2)
    SetUB(Workspace=wsName2, a=3.567, b=3.567, c=3.567, UB=UBlst)
    
    pws = mtd[wsName2]
    for i in range(nOut):
        if ubLab[i]==2:  
            print('Peak: %4i hkl: %6.3f %6.3f %6.3f lambda: %7.4f d-spacing: %7.4f Q: %6.3f %6.3f %6.3f Int: %7.3f'%(i,
            Hr2[i,0],Hr2[i,1],Hr2[i,2],LamOut[i],dOut[i],QOut[i,0],QOut[i,1],QOut[i,2],intenOut[i])) 
            p = wsHandle.createPeak(QOut[i,])
            AddPeakHKL(wsHandle,[Hr2[i,0],Hr2[i,1],Hr2[i,2]])
            p.setWavelength(LamOut[i])

#lastly, offer possibility to refine UB if there are enough  indexed reflections
if nInd[0]>=3:
    print('You have more than three peaks for diamond 1')
    rTog = input('Input: ','You have more than three peaks for diam 1, refine UB?: (y/[n] ', 'str')
    if rTog == 'y':
        print('UB1 refined')
        #resultant lattice is not exactly cubic. Can force this with the following
        FindUBUsingLatticeParameters(PeaksWorkspace=wsName1, a=3.567, b=3.567, c=3.567, alpha=90, beta=90, gamma=90, FixParameters=True, Iterations=5)
        #this scrambles the UB orientation, so need to read UB, reorient, write UB back
        pws = mtd[wsName1]
        mat = np.array(pws.sample().getOrientedLattice().getUB())# copy refined UB
        mat = diamUBFunctions.UBStandardOrient(mat) # apply standard orientation
        SetUB(Workspace=wsName1, a=3.567, b = 3.567, c = 3.567, UB=mat) #write std orientation UB to ws
if nInd[1]>=3:
    print('You have more than three peaks for diamond 2')
    rTog = input('Input: ','You have more than three peaks for diam 2, refine UB?: (y/[n] ', 'str')
    if rTog == 'y':
        print('UB2 refined')
        FindUBUsingLatticeParameters(PeaksWorkspace=wsName2, a=3.567, b=3.567, c=3.567, alpha=90, beta=90, gamma=90, FixParameters=True, Iterations=5)
        #this scrambles the UB orientation, so need to read UB, reorient, write UB back
        pws = mtd[wsName2]
        mat = np.array(pws.sample().getOrientedLattice().getUB())
        mat = diamUBFunctions.UBStandardOrient(mat)
        SetUB(Workspace=wsName2, a=3.567, b = 3.567, c = 3.567, UB=mat)

#SAVE BOTH UB's
IndexPeaks(PeaksWorkspace=wsName1, Tolerance=0.10000000000000001, RoundHKLs=False)
SaveIsawUB(InputWorkspace=wsName1, Filename=FullOName1)
IndexPeaks(PeaksWorkspace=wsName2, Tolerance=0.10000000000000001, RoundHKLs=False)
SaveIsawUB(InputWorkspace=wsName2, Filename=FullOName2)

print("UB matrix diamond 1 (before refining):")
print(np.array_str(UBinit1, precision=4, suppress_small=True))

pws1 = mtd[wsName1]
mat = np.array(pws1.sample().getOrientedLattice().getUB())
mat = diamUBFunctions.UBStandardOrient(mat)
print("UB matrix diamond 1 (after refining):")
print(np.array_str(mat, precision=4, suppress_small=True))
TM2I = np.array([[0, 0, 1],[1,0,0],[0,1,0]]) #transforms mantid convention coords into isaw convention coords
mat = np.dot(TM2I,mat)
print("UB matrix diamond 1 (after refining) ISAW conv:")
print(np.array_str(mat, precision=4, suppress_small=True))
print("Transpose of ISAW UB (as given in isaw.mat file)")
mat = np.transpose(mat)
print(np.array_str(mat, precision=4, suppress_small=True))

#readUnitCell = ws1.getUnitCell()
#print(readUnitCell.a())
#print(ws1.getUnitCell())
#    diamLst1 = mtd['diamUB1']
#    vol1 = diamLst1.sample().getOrientedLattice().volume()
#    print('refined volume of unit cell is: ',vol1)

##now process UB2 if it exists
#if np.any(UB2):
#    print('working on UB2')
#    UBlst = [UB2[0,0],UB2[0,1],UB2[0,2],UB2[1,0],UB2[1,1],UB2[1,2],UB2[2,0],UB2[2,1],UB2[2,2]]
#    print('UB2 associated with diamUB2 workspace')
#    SetUB(Workspace='diamUB2', a=3.567, b=3.567, c=3.567, UB=UBlst)
#    IndexPeaks(PeaksWorkspace='diamUB2', Tolerance=0.10, ToleranceForSatellite=0.10)
#    FindUBUsingIndexedPeaks(PeaksWorkspace='diamUB2')
#    SaveIsawUB(InputWorkspace='diamUB2', Filename=FullOName2)

#    diamLst2 = mtd['diamUB2']
#lattice2 = np.array(diamLst2.sample().getOrientedLattice().getUB())
#    vol2 = diamLst2.sample().getOrientedLattice().volume()
#    print('refined volume of unit cell is: ',vol2)
#print('Diamond unit vectors along diamond cell axes:')


#Last step is to use volumes to determine which diamond is upstream
#Assuming the instrument is well calibrated, this should be clearly measurable, with the 
#upstream diamond having a volume that is ~ 0.7% smaller than the downstream diamond

#
# ONLY DO THIS IF TWO UB's EXIST!
#
#if np.any(UB1)&np.any(UB2):
#    print('for cell with 5 mm diamonds, and sample centred in instrument upstream diamond')
#    print('will have a smaller apparent volume by amount d_vol = -0.0076')
#    print(' ') 
#    if vol1<vol2:
#        print('UB1 is upstream diamond d_vol is: ',(vol1-vol2)/vol2)
#    elif vol2<vol1:
#        print('UB2 is upstream diamond d_vol is: ',(vol2-vol1)/vol1)
#    elif vol1==vol2:
#        print('Unable to determine which diamond is upstream. d_vol is 0')
    
