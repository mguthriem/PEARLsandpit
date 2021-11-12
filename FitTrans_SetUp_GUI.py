#version: 20210331 original script: MXD_GUI_20210331_tmp
# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
#     NScD Oak Ridge National Laboratory, European Spallation Source
#     & Institut Laue - Langevin
# SPDX - License - Identifier: GPL - 3.0 +
'''
1. all the functions are defined and built consistently.




Data types:
- Use only numpy arrays to ensure consistency across formatting and type
*** x, x0 = parameters vector - 1D numpy array
*** setang1, setang2 = angles for refinement - 1D numpy arrays, 3 elements
*** hkl1, hkl2 = numpy arrays (3 columns) having all the hkl indices from the 2 diamonds
*** UB1, UB2 = numpy arrays (3x3) holds UB matrices from input files
***
'''
# Import all needed libraries
from __future__ import (absolute_import, division, print_function)
from matplotlib import pyplot as plt
import numpy as np
import itertools as itt
import time
from lmfit import minimize, Parameters, fit_report
import lmfit
print('lmfit version: ',lmfit.__version__)
from mantid.api import *
from mantid.kernel import *
from mantid.simpleapi import *
from mantid.geometry import *
from mantidqt.utils.qt.qappthreadcall import QAppThreadCall
from mantid.kernel import (StringListValidator, FloatBoundedValidator, StringMandatoryValidator)

from mantidqt.utils.qt.qappthreadcall import QAppThreadCall
from qtpy.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QGridLayout, QLabel, QLineEdit, QPushButton) 
#from mantid.simpleapi import *


__author__ = 'cip+malcolm'

# Define global variables
global hkl1, hkl2
global UB1, pkcalcint1
global UB2, pkcalcint2
global pktype
global lam, y, e, TOF
global L1
global ttot
global fxsamediam
global neqv1, eqvlab1, neqv2, eqvlab2
global difa, function_verbose
global figure_name_attenuation, runNumber
global nit,allnit,allchi2

PLOTTING_WINDOW_STAT = None
PLOTTING_WINDOW_CHI2 = None


def iteration_output(pars, iter, resid, *args, **kws):
    
    import sys
    import os
    global nit,allnit,allchi2,currentnit,currentchi2        
    global maxiter
    global fxsamediam
    global neqv1, eqvlab1, neqv2, eqvlab2
    global runNumber,directory,insTag
    global pktype,nbgd,bgdtype
    '''
    @ prints the process during fitting
    Function prints iteration number and calculated chi2 in %
    '''
    
    #continually write parameters to file
    logFileLoc = directory + insTag + str(runNumber) + 'FitTransPars.log'
          
    updatestp = 1
    chi2 = np.sum(np.square(resid))/resid.size
    nit = nit+1 #
    currentnit = np.append(currentnit,iter+2)
    allnit = np.append(allnit,nit) #
    
    f = open(logFileLoc,'w')
    f.write('iteration: %u\n'%(nit))
    f.write('chi2: %10.2f\n'%(chi2))
    f.write('sf: %5.4E\n'%(pars['sf'].value))
    f.write('fxsamediam: %u\n'%(fxsamediam))
    f.write('relsf: %5.4f\n'%(pars['relsf'].value))
    f.write('setang1_alp: %5.4f\n'%(pars['setang1_alp'].value))
    f.write('setang1_bet: %5.4f\n'%(pars['setang1_bet'].value))
    f.write('setang1_gam: %5.4f\n'%(pars['setang1_gam'].value))
    f.write('setang2_alp: %5.4f\n'%(pars['setang2_alp'].value))
    f.write('setang2_bet: %5.4f\n'%(pars['setang2_bet'].value))
    f.write('setang2_gam: %5.4f\n'%(pars['setang2_gam'].value))
    f.write('PeakType: %u\n'%(pktype))
    for i in range(31):
        coefPar = 'pkCoef'+str(i).zfill(2)
        f.write('%s: %.4E\n'%(coefPar,pars[coefPar].value))
    f.write('L2: %5.4f\n'%(pars['L2'].value))
    f.write('difa: %.4E\n'%(pars['difa'].value))
    f.write('Estr11: %8.5f\n'%(pars['Estr11'].value))
    f.write('Estr22: %8.5f\n'%(pars['Estr22'].value))
    f.write('Estr33: %8.5f\n'%(pars['Estr33'].value))
    f.write('Estr12: %8.5f\n'%(pars['Estr12'].value))
    f.write('Estr13: %8.5f\n'%(pars['Estr13'].value))
    f.write('Estr23: %8.5f\n'%(pars['Estr23'].value))
    f.write('BackgroundType: %u\n'%(bgdtype))
    f.write('NoBackgroundTerms: %u\n'%(nbgd))
    for i in range(24):
        bgdPar = 'bgd'+str(i+1).zfill(3)
        f.write('%s: %.10E\n'%(bgdPar,pars[bgdPar].value))
    f.write('neqv1: %u\n'%(neqv1))    
    for i in range(neqv1):
        pkmult1Par = 'pkmult1_'+str(i+1).zfill(3)
        f.write('%s: %10.7f\n'%(pkmult1Par,pars[pkmult1Par].value))
    if fxsamediam!=1:
        f.write('neqv2: %u\n'%(neqv2))
        for i in range(neqv2):
            pkmult2Par = 'pkmult2_'+str(i+1).zfill(3)
            f.write('%s: %10.7f\n'%(pkmult2Par,pars[pkmult2Par].value))
    f.close() #this is the fit status, write to file
    

    if chi2>=500:
      allchi2 = np.append(allchi2,500) # limit maximum value of chi2 to remove glitches that screw up display.
      currentchi2 = np.append(currentchi2,500)
    else:
      allchi2 = np.append(allchi2,chi2)
      currentchi2 = np.append(currentchi2,chi2)
    md = np.modf(nit/updatestp)
    if md[0]==0: #only update plot every updatestp steps
#       chi2vals = CreateWorkspace(DataX=allnit,DataY=allchi2,NSpec=1,UnitX='Iteration',YUnitLabel='Chi2')
       chi2valsCurrentIter = CreateWorkspace(DataX=currentnit,DataY=currentchi2,NSpec=1,UnitX='Iteration',YUnitLabel='Chi2')
    
#    print('Iterations in current cycle: ',iter)
#    print('Total iterations: ',nit)
    limit = 250
    if iter>int(maxiter) and iter<=limit-1:
        print('max number of iterations: ',maxiter, ' reached')
        print('Last iteration was: ',iter-1)
        return True #this should cancel refinement 
    elif iter>limit:
        print('calling exit()')
        exit()
        print('think I am stuck in a loop. calling sys.exit()')
        sys.exit()
        print('this did not work, so now calling os._exit(0)')
        os._exit(0)

def dlmread(filename):
    '''
    Function to read parameters from file after previous fit
    '''
    content = []
    with open(filename, "r") as f:
        for line in f.readlines():
            content.append(float(line))
    return np.array(content)

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

def calcDspacing(a, b, c, alp, bet, gam, h, k, l):
    '''
    %CALCDSPACING for general unit cell: a,b,c,alp,bet,gam returns d-spacing for
    %reflection h,k,l
    %
    '''
    ca = np.cos(np.radians(alp))
    cb = np.cos(np.radians(bet))
    cg = np.cos(np.radians(gam))
    sa = np.sin(np.radians(alp))
    sb = np.sin(np.radians(bet))
    sg = np.sin(np.radians(gam))

    oneoverdsq = (1.0 - ca ** 2 - cb ** 2 - cg ** 2 + 2 * ca * cb * cg) ** (-1) * \
                 ((h * sa / a) ** 2 + (k * sb / b) ** 2 + (l * sg / c) ** 2
                  + (2 * k * l / (b * c)) * (cb * cg - ca) + (2 * l * h / (c * a)) * (cg * ca - cb)
                  + (2 * h * k / (a * b)) * (ca * cb - cg))

    d = np.sqrt(1.0 / oneoverdsq)
    return d


def genhkl(hmin, hmax, kmin, kmax, lmin, lmax):
    '''
    genhkl generates array of hkl values
    total number of points will be (hmax-hmin)
    '''
    hvals = np.arange(hmin, hmax + 1, 1)
    kvals = np.arange(kmin, kmax + 1, 1)
    lvals = np.arange(lmin, lmax + 1, 1)

    nh = len(hvals)
    nk = len(kvals)
    nl = len(lvals)

    l = 0
    hkl = np.zeros(shape=(nh * nl * nk, 3))
    for i in range(nh):
        for j in range(nk):
            for k in range(nl):
                hkl[l][0] = hvals[i]
                hkl[l][1] = kvals[j]
                hkl[l][2] = lvals[k]
                l += 1
    return hkl


def mod(a, b):
    return a % b


def forbidden(h, k, l):
    '''
    %returns logical positive if this hkl is fobidden according to
    %   diamond reflections conditions....
    '''
    ah = abs(h)
    ak = abs(k)
    al = abs(l)

    if ((h == 0) and (k == 0) and (l == 0)):
        result = 1
        boolresult = bool(result)
        return boolresult
    else:
        result = 0

    if ((ah == 2) and (ak == 2) and (al == 2)):  # allowed, but vanishingly weak
        result = 1
        boolresult = bool(result)
        return boolresult
    else:
        result = 0

    # condition 1
    if ((h != 0) and (k != 0) and (l != 0)):  # general hkl
        term1 = h + k
        term2 = h + l  # all have to be even
        term3 = k + l
        if not ((term1 % 2) == 0 and (term2 % 2) == 0 and (term3 % 2) == 0):
            result = 1
            boolresult = bool(result)
            return boolresult
        else:
            result = 0

    # % condition 2
    if ((h == 0) and (k != 0) and (l != 0)):  # 0kl reflections
        term1 = k + l
        mod4 = mod(term1, 4)
        if not (mod4 == 0 and mod(k, 2) == 0 and mod(l, 2) == 0):
            result = 1
            boolresult = bool(result)
            return boolresult
        else:
            result = 0

    # condition 3
    if (h == k):  # hhl reflections
        if not (mod(h + l, 2) == 0):
            result = 1
            boolresult = bool(result)
            return boolresult
        else:
            result = 0

    # condition 4
    if ((h == 0) and (k == 0) and (l != 0)):  # 00l reflections not including 000
        mod4 = mod(l, 4)
        if not (mod4 == 0):
            result = 1
            boolresult = bool(result)
            return boolresult
        else:
            result = 0

    boolresult = bool(result)
    return boolresult


def allowedDiamRefs(hmin, hmax, kmin, kmax, lmin, lmax):
    '''
    %UNTITLED6 generates a list of allowed reflections for diamond between
    %   limits provided sorted descending according to d-spacing
    '''
    # obtain all hkl within limits...
    allhkl = genhkl(hmin, hmax, kmin, kmax, lmin, lmax)
    # now purge those violating extinction conditions...

    n = len(allhkl)

    # set all forbidden hkl's to zero
    # hkl or lhk or klh
    for i in range(n):
        h = allhkl[i][0]
        k = allhkl[i][1]
        l = allhkl[i][2]
        if forbidden(h, k, l) or forbidden(l, h, k) or forbidden(k, l, h):
            allhkl[i] = 0  # set equal to zero

    k = 0
    d = []  # np.zeros(0)
    # create new array with all h!=0 k!=0 l!=0
    hkl = np.zeros(shape=(0, 3))
    for i in range(n):
        if not (allhkl[i][0] == 0 and allhkl[i][1] == 0 and allhkl[i][2] == 0):
            hkl = np.vstack((hkl, [allhkl[i][0], allhkl[i][1], allhkl[i][2]]))
            d.append(calcDspacing(3.56683, 3.56683, 3.56683, 90,
                                  90, 90, hkl[k][0], hkl[k][1], hkl[k][2]))
            k += 1
    d = np.array(d)

    # ORDER hkl according to d-spacing
    B = sorted(d)[::-1]  # returns d sorted in descending order
    IX = np.argsort(d)[::-1]  # and corresponding elements

    sorthkl = np.zeros(shape=(k, 3))
    for i in range(k):
        sorthkl[i] = hkl[IX[i]]
        d[i] = B[i]
        # print('hkl: {0:0.3f} {1:0.3f} {2:0.3f} d-spacing: {3:0.3f} A'.format(sorthkl[i][0], sorthkl[i][1],
        #    sorthkl[i][2], d[i]))

    return sorthkl


def getISAWub(fullfilename):
    '''
    %getISAWub reads UB determined by ISAW and stored in file "fname"
    %   Detailed explanation goes here


    % [filename pathname ~] = ...
    %     uigetfile('*.dat','Choose UB file (generated by ISAW)');
    % fullfilename = [pathname filename];
    '''
    fileID = fullfilename
    if fileID == 1:
        print(('Error opening file: ' + fullfilename))
    f = open(fileID, "r")
    lines = f.readlines()
    f.close()

    # Build UB matrix and lattice
    UB = np.zeros(shape=(3, 3))
    lattice = np.zeros(shape=(2, 6))
    for i in range(3):
        UB[i][0], UB[i][1], UB[i][2] = lines[i].split()
    UB = UB.transpose()
    for i in range(3, 5):
        lattice[i - 3][0], lattice[i - 3][1], \
            lattice[i - 3][2], lattice[i - 3][3], \
            lattice[i - 3][4], lattice[i - 3][5], \
            non = lines[i].split()

    print('Successfully got UB and lattice')

    return UB, lattice


def pkintread(hkl, loc):
    '''
    %reads calculated Fcalc and converts to
    %Fobs using Buras-Gerard Eqn.
    %inputs are hkl(nref,3) and
    % loc(nref,3), which contains, lambda, d-spacing and ttheta for
    % each of the nref reflections.

    % get Fcalcs for diamond, generated by GSAS (using lattice parameter 3.5668
    % and Uiso(C) = 0.0038

    % disp('in pkintread');


    returns pkint = np. array - 1D vector
    '''
    # A = np.genfromtxt('diamond_reflist.csv', delimiter=',', skip_header=True)
    # print A
    A = np.array([[1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 8.00000000e+00,
                   2.06110000e+00, 5.54000000e+04],
                  [2.00000000e+00, 2.00000000e+00, 0.00000000e+00, 1.20000000e+01,
                   1.26220000e+00, 7.52000000e+04],
                  [3.00000000e+00, 1.00000000e+00, 1.00000000e+00, 2.40000000e+01,
                   1.07640000e+00, 2.98000000e+04],
                  [2.00000000e+00, 2.00000000e+00, 2.00000000e+00, 8.00000000e+00,
                   1.03060000e+00, 2.50000000e-25],
                  [4.00000000e+00, 0.00000000e+00, 0.00000000e+00, 6.00000000e+00,
                   8.92500000e-01, 4.05000000e+04],
                  [3.00000000e+00, 3.00000000e+00, 1.00000000e+00, 2.40000000e+01,
                   8.19000000e-01, 1.61000000e+04],
                  [4.00000000e+00, 2.00000000e+00, 2.00000000e+00, 2.40000000e+01,
                   7.28700000e-01, 2.18000000e+04],
                  [5.00000000e+00, 1.00000000e+00, 1.00000000e+00, 2.40000000e+01,
                   6.87000000e-01, 8.64000000e+03],
                  [3.00000000e+00, 3.00000000e+00, 3.00000000e+00, 8.00000000e+00,
                   6.87000000e-01, 8.64000000e+03],
                  [4.00000000e+00, 4.00000000e+00, 0.00000000e+00, 1.20000000e+01,
                   6.31100000e-01, 1.17000000e+04],
                  [5.00000000e+00, 3.00000000e+00, 1.00000000e+00, 4.80000000e+01,
                   6.03400000e-01, 4.65000000e+03],
                  [4.00000000e+00, 4.00000000e+00, 2.00000000e+00, 2.40000000e+01,
                   5.95000000e-01, 1.83000000e-12],
                  [6.00000000e+00, 2.00000000e+00, 0.00000000e+00, 2.40000000e+01,
                   5.64500000e-01, 6.31000000e+03],
                  [5.00000000e+00, 3.00000000e+00, 3.00000000e+00, 2.40000000e+01,
                   5.44400000e-01, 2.50000000e+03],
                  [6.00000000e+00, 2.00000000e+00, 2.00000000e+00, 2.40000000e+01,
                   5.38200000e-01, 8.80000000e-26],
                  [4.00000000e+00, 4.00000000e+00, 4.00000000e+00, 8.00000000e+00,
                   5.15300000e-01, 3.40000000e+03],
                  [5.00000000e+00, 5.00000000e+00, 1.00000000e+00, 2.40000000e+01,
                   4.99900000e-01, 1.35000000e+03],
                  [7.00000000e+00, 1.00000000e+00, 1.00000000e+00, 2.40000000e+01,
                   4.99900000e-01, 1.35000000e+03],
                  [6.00000000e+00, 4.00000000e+00, 2.00000000e+00, 4.80000000e+01,
                   4.77100000e-01, 1.83000000e+03],
                  [7.00000000e+00, 3.00000000e+00, 1.00000000e+00, 4.80000000e+01,
                   4.64800000e-01, 7.25000000e+02],
                  [5.00000000e+00, 5.00000000e+00, 3.00000000e+00, 2.40000000e+01,
                   4.64800000e-01, 7.25000000e+02],
                  [8.00000000e+00, 0.00000000e+00, 0.00000000e+00, 6.00000000e+00,
                   4.46200000e-01, 9.84000000e+02],
                  [7.00000000e+00, 3.00000000e+00, 3.00000000e+00, 2.40000000e+01,
                   4.36100000e-01, 3.90000000e+02],
                  [6.00000000e+00, 4.00000000e+00, 4.00000000e+00, 2.40000000e+01,
                   4.32900000e-01, 1.53000000e-13],
                  [6.00000000e+00, 6.00000000e+00, 0.00000000e+00, 1.20000000e+01,
                   4.20700000e-01, 5.30000000e+02],
                  [8.00000000e+00, 2.00000000e+00, 2.00000000e+00, 2.40000000e+01,
                   4.20700000e-01, 5.30000000e+02],
                  [5.00000000e+00, 5.00000000e+00, 5.00000000e+00, 8.00000000e+00,
                   4.12200000e-01, 2.10000000e+02],
                  [7.00000000e+00, 5.00000000e+00, 1.00000000e+00, 4.80000000e+01,
                   4.12200000e-01, 2.10000000e+02],
                  [6.00000000e+00, 6.00000000e+00, 2.00000000e+00, 2.40000000e+01,
                   4.09500000e-01, 1.98000000e-26],
                  [8.00000000e+00, 4.00000000e+00, 0.00000000e+00, 2.40000000e+01,
                   3.99100000e-01, 2.85000000e+02],
                  [7.00000000e+00, 5.00000000e+00, 3.00000000e+00, 4.80000000e+01,
                   3.91900000e-01, 1.13000000e+02],
                  [9.00000000e+00, 1.00000000e+00, 1.00000000e+00, 2.40000000e+01,
                   3.91900000e-01, 1.13000000e+02],
                  [8.00000000e+00, 4.00000000e+00, 2.00000000e+00, 4.80000000e+01,
                   3.89500000e-01, 4.44000000e-14],
                  [6.00000000e+00, 6.00000000e+00, 4.00000000e+00, 2.40000000e+01,
                   3.80600000e-01, 1.53000000e+02],
                  [9.00000000e+00, 3.00000000e+00, 1.00000000e+00, 4.80000000e+01,
                   3.74200000e-01, 6.08000000e+01],
                  [8.00000000e+00, 4.00000000e+00, 4.00000000e+00, 2.40000000e+01,
                   3.64400000e-01, 8.26000000e+01],
                  [9.00000000e+00, 3.00000000e+00, 3.00000000e+00, 2.40000000e+01,
                   3.58800000e-01, 3.27000000e+01],
                  [7.00000000e+00, 5.00000000e+00, 5.00000000e+00, 2.40000000e+01,
                   3.58800000e-01, 3.27000000e+01],
                  [7.00000000e+00, 7.00000000e+00, 1.00000000e+00, 2.40000000e+01,
                   3.58800000e-01, 3.27000000e+01]])

    diamd = A[:, 4]
    # diamMult = A[:, 3] # unused variable
    diamFCalcSq = A[:, 5]
    nref = hkl.shape[0]
    # % disp(['there are: ' num2str(nref) ' reflections']);
    # % whos loc

    '''
    % [i,j] = size(x);
    % dipspec = zeros(i,j); %array containing dip spectrum
    % difspec = zeros(i,j); %array containing diffraction spectrum
    % d = x/sqrt(2);    %dspacings for this lamda range at 90 degrees

    % In order to calculate the scattered intensity I from the Fcalc^2, need to
    % apply the Buras-Gerward formula:
    %
    % Fcalc^2 = I*2*sin(theta)^2/(lamda^2*A*E*incFlux*detEffic)
    '''
    pkint = np.zeros(nref)

    for i in range(nref):
        if loc[i][0] > 0:
            # % satisfies Bragg condition (otherwise ignore)
            Fsq = Fsqcalc(loc[i][1], diamd, diamFCalcSq)
            # % Fsq = 1;
            L = (np.sin(np.radians(loc[i][2] / 2.0))) ** 2  # Lorentz correction
            R = 1.0  # %dipLam(i)^4; %reflectivity correction
            A = 1.0  # %Absorption correction
            Ecor = 1
            pkint[i] = Fsq * R * A / (L * Ecor)  # %scattered intensity

    '''
    % whos difspec
    % whos van
    % whos dipspec

    % difspec = difspec.*van;
    % dipspec = dipspec.*van;

    % figure(1)
    % plot(d,difspec)
    '''
    return pkint


def Fsqcalc(d, diamd, diamFCalcSq):
    '''
    % diamond reflections are identified according to their d-spacing
    % and corresponding calculated Fsq are returned

    % global sf111 sf220 sf311 sf400 sf331
    '''
    # n = len(diamd) # unused variable
    ref = d
    dif = abs(diamd - ref)
    i = dif.argmin(0)  # i is index of diamd closest to d
    Fsq = diamFCalcSq[i]
    return Fsq

def pkposcalc_strain(hkl, UB, setang, epsilon):
    '''
    calculates some useful numbers from (ISAW calculated) UB
    hkl is a 2D array containing all hkl's
    n.b. Adapted from the original pkposcalc to include infinitesimal strain
    present in the diamond
    
    The 3x3 matrix epsilon is the infinitesimal strain tensor.
    Thus, the diagonal elements are normal strains and off diagonal elements
    are shear strains.
    
    Here, the strain tensor operates on the reciprocal lattice vector. Accordingly
    any strain component that is > 1 implies lengthening of the corresponding
    Q component and, therefore, a real-space *compression*
    
    As epsilon is symmetric it includes only three independent terms

    '''

    ome = setang[0]
    phi = setang[1]
    chi = setang[2]
    thkl = hkl.transpose()

    Q = UB.dot(thkl)
    Q = epsilon.dot(Q)  # pre-multiply Q by strain matrix

    Rx = np.array([[1, 0, 0], [0, np.cos(np.radians(ome)), -np.sin(np.radians(ome))],
                   [0, np.sin(np.radians(ome)), np.cos(np.radians(ome))]])
    Ry = np.array([[np.cos(np.radians(phi)), 0, np.sin(np.radians(phi))], [0, 1, 0],
                   [-np.sin(np.radians(phi)), 0, np.cos(np.radians(phi))]])
    Rz = np.array([[np.cos(np.radians(chi)), -np.sin(np.radians(chi)), 0],
                   [np.sin(np.radians(chi)), np.cos(np.radians(chi)), 0], [0, 0, 1]])
    Rall = Rz.dot(Ry).dot(Rx)  # all three rotations

    Q = 2*np.pi*Rall.dot(Q)
    magQ = np.sqrt((Q * Q).sum(axis=0))

    d = (2*np.pi / magQ)  # by definition (note ISAW doesn't use 2pi factor)
    d = d.transpose()
    alp = np.degrees(np.arccos(-Q[2,:]/magQ)) #dot product of unit vector along beam and Q related to 2theta
    ttheta = -(180 - 2*alp) #don't quite understand where negative sign comes from, but it's correct
    ttheta = ttheta.transpose()
    # and Bragg's law gives:
    lambda_1 = 2 * d * np.sin(np.radians(ttheta / 2))
    lambda_1 = lambda_1.transpose()
    Q = Q.transpose()
    #print(lambda_1.shape,d.shape,Q.shape)

    return lambda_1, d, ttheta, Q


def getMANTIDdat_keepbinning(csvfile):
    '''
    getMANTIDdat reads data from mantid "SaveAscii" output
    %   input file name should be 'csvfilename'.csv
    %   data are returned with binning (xmin:xbin:xmax)

    returns TOF, y , e
    '''
    fid = open(csvfile, "r")
    lines = fid.readlines()
    x = []
    y = []
    e = []
#    if fid < 0:
#        print(('Error opening file: ' + csvfile))
    for i in range(1, len(lines)):
        a, b, c = lines[i].split(",")
        x.append(float(a))
        y.append(float(b))
        e.append(float(c))
    fid.close()
    x = np.array(x)
    y = np.array(y)
    e = np.array(e)

    return x, y, e

def findeqvs(hkl):
    '''
    FINDEQVS runs through array of hkls and labels those that are equivalent
    %in the m-3m point group.
    %
    % there are n reflections.
    % hkl has dimensions nx3
    % eqvlab has dimensions nx1
    '''
    n, m = hkl.shape
    eqvlab = np.zeros(n)
    lab = 1
    #print('there are: ',n,' reflections to check')
    for i in range(n):
        if eqvlab[i] == 0:  # then it's not been checked yet, so check it
            eqvlab[i] = lab
            refhkl = np.array([abs(hkl[i][0]), abs(hkl[i][1]), abs(hkl[i][2])])
            for j in range(i + 1, n):  # check remaining indices
                comphkl = np.array(
                    [abs(hkl[j][0]), abs(hkl[j][1]), abs(hkl[j][2])])
                #print('ref hkl is: ',refhkl,' comparing with ref: ',j,': ',comphkl)    
                # all possible permutations
                permcomphkl = list(itt.permutations(comphkl))
                nperm = len(permcomphkl)
                for k in range(nperm):
                    if refhkl[0] == permcomphkl[k][0] and refhkl[1] == permcomphkl[k][1] and \
                                    refhkl[2] == permcomphkl[k][2]:
                        eqvlab[j] = lab
                        #print('equivalent!')
            lab += 1
        
    #for i in range(n):
    #    print(hkl[i][0],hkl[i][2],hkl[i][2],'label: ',eqvlab[i])
    return eqvlab, lab

def lam2tof(lam,L,difa):
    #where lam is an np array, L total flight path and difa quadratic parameter
    #returns corresponding values of tof
    
#    print('in lam2tof, processing: ',lam.size,' reflections')
#    print('difa = ',difa)
#    print('Ltot = ',L)
    
    a = difa;
    b = (0.0039558/L)
    
    if a < 1e-10 : #expression becomes ill conditioned
        tof = (L/0.0039558)*lam
    else:
        tof = (b+np.sqrt(b^2+4*a*(lam)))/(2*a)
    
    return tof

def tof2lam(tof,L,difa):
    
    lam = (0.0039558/L)*tof + difa*np.square(tof)
    
    return lam

def SimTrans3_tof(fit_params, tof, y, e):
    '''
    %SimTrans calculates transmission spectrum from two crystals
    %   lam - array containing wavelengths to calc over
    %   hkl - contains all Nref hkl's that calculation is performed for
    %   bgd - array containing coefficients of polynomial for background
    %   sf - overall scale factor
    %   pktype - 1 = gauss; 2 = lorentz; ...
    %   UB1 - UB matrix for first crystal
    %   setang1 - setting angles for first crystal (deviations from ideal UB
    %               location).
    %   pkpars1 - position(lambda), position(d-spacing), position(ttheta), width, intensity for each Nref reflections
    %   UB2,setang2,pkpars2 - as above, for second crystal
    %
    % M. Guthrie 21st Jan 2014
    %
    % calculate background profile
    % determine number of coeffs bgd
    %
    % version 2 constrains intensities for equivalent dips to be the same
    % M. Guthrie 3 April 2014
    %
    % M. Guthrie 7 April 2014, realised I was making an (obvious) mistake by
    % adding the transmissions from the two diamonds. Clearly, they should be
    % multiplied. I've implemented the change...will see what difference it
    % makes.
    %
    % M. Guthrie 9 April 2014, introduced possibility to refine L2 and also a
    % scale factor for calculated dip wavelengths (to account for diamond
    % compressibility).
    %
    %M. Guthrie 12 March 2020: (version a) want to upgrade simulated spectrum with developments since
    % 2014. This will include: 
    % 1) including the stress tensor to allow peak positions to shift asymmetrically according to (the massive) diamond strain 
    % 2) allowing more background coefficiets for more complex baxkground shape
    % 3) attempt to allow refinement of "difa", currently passed as via a global
    % In doing this, it makes sense to re-number all of the parameters, as it's extremely messy at the moment
    % The parameters that depend on the number of observed reflections should be moved to the end
    % of the array so that the numbering of other static parameters doesn't change with every successive
    % edit of the software. I will just leave sufficient space to allow for future tweaking...
    %
    % M. Guthrie 16 March 2020 on suggestion of Thomas Holm Rod, have switched to using lmfit
    % this looks much better for setting up as parameters are specified by a library
    % and not some random location in an array. Also, setting up of constraints and 
    % boundary conditions looks much more intuitive.
    %
    %
    % M. Guthrie 19 March 2020modified fitting so that it occurs in TOF space, this is more correct as
    % that is the measurement parameter. To allow easy flipping of lambda and TOF
    % I defined the functions lam2tof and tof2lam. 
    % note in previous versions, I used the term difa in analogy with the gsas term
    % however, I realise the resulting expression lam = A*T + difa*T^2 is confusing as
    % gsas operates with the inverse of this function (and with d instead of lambda)
    % long story short: I'm keeping difa, but it is now a different quantity...
    % 
    % M. Guthrie 22 Dec 2020 (version a): added a psuedovoigt peak shape adjusted gaussian
    % peak definitions for consistency.
    %
    % M. Guthrie 23 Dec 2020: added tick marks for calculated peak positions
    %
    % M. Guthrie 07 Jan 2021: fixed bug in pkmult setting during fitting
    % adjusted use of pktype to allow pure gauss, pure lorentz or psuedovoigt. Found another bug in peak definitions where
    % pkmult1 was used as multiplier for diamond 2 dips.
    % Added file output containing dip: hkl, lambda
    %
    % M. Guthrie 26 Jan 2021: added Bk2BkExpConvPV peak type. this was quite a bit of hacking as I had to create additional peak par terms
    % (I ended up increasing to 10 although I only need 6, for future proofing). Also, am using a bit of a clunky way to evaluate the 
    % peaks (using a mantid ws), currently the only way I can figure out to do it. 
    %
    % M Guthrie 5 Feb 2021: major change to peak profile definitions, now using GSAS TOF profile 3 as reference
    % increased number of possible parameter to 31 and renamed everything for more consistency.
    '''
    
    
    
    global hkl1, hkl2
    global UB1, pkcalcint1
    global UB2, pkcalcint2
    global pktype,nbgd,bgdtype  #pktype = 1 is Guass, = 2 is Lorenzt, 3 = psuedovoigt and mix par refined.
    global lam, shftlam
    global L1
    global ttot,t1,t2,bgdprof
    global fxsamediam
    global neqv1, eqvlab1, neqv2, eqvlab2
    global function_verbose
    global runNumber,directory,insTag
    
    ioioName = insTag + str(runNumber) + '_monitors_ioio' #this workspace is used to provide instrument info for Bk2BkExpConvPV peak shape

    nref1 = hkl1.shape[0]  # % number of reflections to integrate over
    nref2 = hkl2.shape[0]  # % number of reflections to integrate over
    # % returns array with same dim as input labelling equivs
    eqvlab1, neqv1 = findeqvs(hkl1)
    eqvlab2, neqv2 = findeqvs(hkl2)

    bgd = np.zeros(24) # up to 24 term chebychev background used 
    pkmult1 = np.zeros(neqv1)
    pkmult2 = np.zeros(neqv2)  
    sf = fit_params['sf'].value                 
    relsf =fit_params['relsf'].value            
    setang1 = np.zeros(3)
    setang1[0] = fit_params['setang1_alp'].value        #setting angles for diamond 1
    setang1[1] = fit_params['setang1_bet'].value
    setang1[2] = fit_params['setang1_gam'].value
    setang2 = np.zeros(3)
    setang2[0] = fit_params['setang2_alp'].value        #setting angles for diamond 2
    setang2[1] = fit_params['setang2_bet'].value
    setang2[2] = fit_params['setang2_gam'].value
    epsilon = np.zeros(shape=(3,3),dtype=float)
    epsilon[0,0] = fit_params['Estr11'].value
    epsilon[1,1] = fit_params['Estr22'].value
    epsilon[2,2] = fit_params['Estr33'].value
    
    epsilon[0,1] = fit_params['Estr12'].value
    epsilon[0,2] = fit_params['Estr13'].value
    epsilon[1,2] = fit_params['Estr23'].value

    alp = fit_params['pkCoef00'].value   
    bet0 = fit_params['pkCoef01'].value 
    bet1 = fit_params['pkCoef02'].value  
    sig0 = fit_params['pkCoef03'].value
    sig1 = fit_params['pkCoef04'].value
    sig2 = fit_params['pkCoef05'].value 
    gam0 = fit_params['pkCoef06'].value
    gam1 = fit_params['pkCoef07'].value
    gam2 = fit_params['pkCoef08'].value
    eta = fit_params['pkCoef21'].value
    regwid1 = fit_params['pkCoef26'].value
    regwid2 = fit_params['pkCoef27'].value
    
    L2 = fit_params['L2'].value              #secondary flight path (to monitor)
    difa = fit_params['difa'].value            #quadratic TOF to lambda term
    for i in range(24):
        bgd[i] = fit_params['bgd'+str(i+1).zfill(3)]
    for i in range(neqv1):
        pkmult1[i] = fit_params['pkmult1_'+str(i+1).zfill(3)]
    for i in range(neqv2):
        pkmult2[i] = fit_params['pkmult2_'+str(i+1).zfill(3)]
    if fxsamediam == 1:
        if neqv1==neqv2:
            pkmult2 = pkmult1*relsf
        elif neqv2>neqv1:
            pkmult2[0:neqv1] = pkmult1[0:neqv1]*relsf
        elif neqv1>neqv2:
            pkmult2[0:neqv2]=pkmult1[0:neqv2]*relsf

    # number of data points to calculate over fit over
    npt = tof.shape[0]
    
    # calculate information for peaks for crystal 1 using hkl,UB1, setang,
    # pkpos    
    a, b, c, q = pkposcalc_strain(hkl1, UB1, setang1, epsilon)
    diamTOF1 = lam2tof(a,(L1+L2),difa) #get time of flight for each hkl
    lam1 = a
    srtlam1 = np.sort(lam1)[::-1] #array of lambdas sorted with lambda descending
    isrtlam1 = np.argsort(lam1)[::-1] #corresponding sorted indices
    srthkl1 = hkl1[isrtlam1] #and sorted hkl
    srtd1 = b[isrtlam1] #sorted d-spacing
    srtQ1 = q[isrtlam1,:] #sorted Q coords
    srtTT1 = c[isrtlam1] #sorted TTheta
    srtdiamTOF1 = diamTOF1[isrtlam1] #sorted TOFs for diamond 1
    srteqvlab1 = eqvlab1[isrtlam1] #sorted equivalent lables
    # calculate information for peaks for crystal 2 using hkl,UB1, setang,
    # pkpos
    a, b, c, q = pkposcalc_strain(hkl2, UB2, setang2, epsilon)
    diamTOF2 = lam2tof(a,L1+L2,difa)
    lam2 = a
    srtlam2 = np.sort(lam2)[::-1] #useful to have file output sorted with lambda descending
    isrtlam2 = np.argsort(lam2)[::-1] #corresponding sorted indices
    srthkl2 = hkl2[isrtlam2] #and sorted hkl       
    srtd2 = b[isrtlam2] #sorted d-spacing
    srtQ2 = q[isrtlam2,:] #sorted Q coords
    srtTT2 = c[isrtlam2]
    srtdiamTOF2 = diamTOF2[isrtlam2] #sorted TOFs for diamond 2
    srteqvlab2 = eqvlab2[isrtlam2] #sorted equivalent lables
            
    yvals = np.divide(lam1,lam1)*1.03;        
    ticks1 = CreateWorkspace(OutputWorkspace='ticks1',DataX=lam1, DataY=yvals, NSpec=1,UnitX='Wavelength')        

    yvals = np.divide(lam2,lam2)*1.03;        
    ticks2 = CreateWorkspace(OutputWorkspace='ticks2',DataX=lam2, DataY=yvals, NSpec=1,UnitX='Wavelength')        
    
    # used to add custom ticks if necessary
    #mystery = np.array([2.245,2.274,2.318,2.500])
    #yvals = np.divide(mystery,mystery)*1.04;
    #ticks3 = CreateWorkspace(OutputWorkspace='ticks3',DataX=mystery, DataY=yvals, NSpec=1,UnitX='Wavelength')        

    if bgdtype ==1: #Chebychev background
#        print('Chebychev background being fitted')
        print('nBgd= ',nbgd)
        print('BGD: ',bgd[0:nbgd])
        bgdprof = np.polynomial.chebyshev.chebval(tof,bgd[0:nbgd])
        
    tofsclfact = 2.00e-3 

#####################################################################################
# calculate dip spectrum here and write peak info to file
#####################################################################################

    peakInfoFname1 = directory + insTag + str(runNumber) + '_TransDipInfo1.log'    
    peakInfoFname2 = directory + insTag + str(runNumber) + '_TransDipInfo2.log'  
    
    f1 = open(peakInfoFname1,'w')
    f2 = open(peakInfoFname2,'w')
    
    regionTOFmin = 9620.0
    regionTOFmax = 10970.0
    regionwidscale = np.array([regwid1,regwid2])# first affect diam 1, second diam 2
    regionintscale = np.array([1.0,1.0])#

    # calculate peaks for crystal 1

    peakInfoFname1 = directory + insTag + str(runNumber) + '_TransDipInfo1.log'    
    peakInfoFname2 = directory + insTag + str(runNumber) + '_TransDipInfo2.log'  
    
    f1 = open(peakInfoFname1,'w')
    f1.write('   h   k   l     lam      Qx      Qy      Qz  ttheta     clk    mult eqv\n')
    f2 = open(peakInfoFname2,'w')
    f2.write('   h   k   l     lam      Qx      Qy      Qz  ttheta     clk    mult eqv\n')

    # calculate peaks for diamond 1
    dips1 = np.zeros(npt)  # initialise array containing dip profile
    for i in range(nref1):

        TOF = srtdiamTOF1[i]
        scl = TOF*tofsclfact
        Sigma2 = sig0**2+sig1**2*TOF**2+sig2**2*TOF**4  #gaussian variance
        if TOF>regionTOFmin and TOF<regionTOFmax:
            Sigma2 = Sigma2*regionwidscale[0]
            scl = scl*regionintscale[0]

        Alpha = alp/TOF
        Beta = bet0 + bet1/TOF**4
        Gamma = gam0+gam1*TOF+gam2*TOF**2
        FWHM = 2*np.sqrt(2*np.log(2))*np.sqrt(Sigma2)
       

        Amp = 1e4*scl*pkmult1[int(srteqvlab1[i])]
        
        if pktype == 1: # Gauss
            func = Gaussian()
            Height = Amp/(np.sqrt(2*np.pi*Sigma2))
            pin = [Height,TOF,np.sqrt(Sigma2)]
        elif pktype == 2: # Lorentz
            func = Lorentzian()
            pin = [Amp,TOF,FWHM]
        elif pktype == 3: # psuedovoigt
            func = PseudoVoigt()
            pin = [eta,Amp,TOF,FWHM]
        elif pktype == 4: # Bk2BkExpConvPV            
            func = Bk2BkExpConvPV()#name = Bk2BkExpConvPV,X0=-0,Intensity=0,Alpha=1,Beta=1,Sigma2=1,Gamma=0
            pin = [TOF,Amp,Alpha,Beta,Sigma2,Gamma]

        ws = mtd['tmp_peak_ws']
        [func.function.setParameter(ii,p) for ii,p in enumerate(pin)]
        ws_eval = func(ws)
        testyin = ws_eval.readY(1)     
        dips1= dips1 - testyin
        

        if srtQ1[i,0]<=0: #in mantid this mean rhs of vertical centre, looking along beam
            clock = 360.0- np.degrees( np.arccos(srtQ1[i,1]/np.linalg.norm(srtQ1[i,:])))
        else: #in mantid this mean lhs of vertical centre, looking along beam
            clock = np.degrees( np.arccos(srtQ1[i,1]/np.linalg.norm(srtQ1[i,:]))) 
        f1.write('%4u%4u%4u%8.4f%8.4f%8.4f%8.4f%8.1f%8.1f%8.4f%4u\n'%(srthkl1[i,0],srthkl1[i,1],\
        srthkl1[i,2],srtlam1[i],srtQ1[i,0],srtQ1[i,1],srtQ1[i,2],srtTT1[i],clock,\
        pkmult1[int(srteqvlab1[i])],int(srteqvlab1[i])+1))
            
    # calculate peaks for diamond 2
    dips2 = np.zeros(npt)  # initialise array containing profile
    for i in range(nref2):
        TOF = srtdiamTOF2[i]
        scl = TOF*tofsclfact
        Sigma2 = sig0**2+sig1**2*TOF**2+sig2**2*TOF**4  #gaussian variance
        if TOF>regionTOFmin and TOF<regionTOFmax:
            Sigma2 = Sigma2*regionwidscale[1]
            scl = scl*regionintscale[1]
        Alpha = alp/TOF
        Beta = bet0 + bet1/TOF**4
        Gamma = gam0+gam1*TOF+gam2*TOF**2
        FWHM = 2*np.sqrt(2*np.log(2))*np.sqrt(Sigma2)
       

        Amp = 1e4*scl*pkmult2[int(srteqvlab2[i])]
        
        if pktype == 1: # Gauss
            func = Gaussian()
            Height = Amp/(np.sqrt(2*np.pi*Sigma2))
            pin = [Height,TOF,np.sqrt(Sigma2)]     
        elif pktype == 2: # Lorentz
            func = Lorentzian()
            pin = [Amp,TOF,FWHM]
        elif pktype == 3: # psuedovoigt
            func = PseudoVoigt()
            pin = [eta,Amp,TOF,FWHM]
        elif pktype == 4: # Bk2BkExpConvPV            
            func = Bk2BkExpConvPV()#name = Bk2BkExpConvPV,X0=-0,Intensity=0,Alpha=1,Beta=1,Sigma2=1,Gamma=0
            pin = [TOF,Amp,Alpha,Beta,Sigma2,Gamma]

        ws = mtd['tmp_peak_ws']
        [func.function.setParameter(ii,p) for ii,p in enumerate(pin)]
        ws_eval = func(ws)
        testyin = ws_eval.readY(1)     
        dips2= dips2 - testyin
                
        if srtQ2[i,0]<=0: #in mantid this mean rhs of vertical centre, looking along beam
            clock = 360.0 - np.degrees( np.arccos(srtQ2[i,1]/np.linalg.norm(srtQ2[i,:])))
        else: #in mantid this mean lhs of vertical centre, looking along beam
            clock = np.degrees( np.arccos(srtQ2[i,1]/np.linalg.norm(srtQ2[i,:]))) 
        f2.write('%4u%4u%4u%8.4f%8.4f%8.4f%8.4f%8.1f%8.1f%8.4f%4u\n'%(srthkl2[i,0],srthkl2[i,1],\
        srthkl2[i,2],srtlam2[i],srtQ2[i,0],srtQ2[i,1],srtQ2[i,2],srtTT2[i],clock,\
        pkmult2[int(srteqvlab2[i])],int(srteqvlab2[i])+1))
  
        
    f1.close()
    f2.close()

    # calculate final profile
    t1 = 1+sf*dips1 #individual transmission 
    t2 = 1+sf*dips2 #individual transmission
    ttot = t2*t1*bgdprof #note order of multiplication doesn't have an affect (i.e. either diamond could be upstream of the other)
    # explanation:
    #Beam from left to right
    # I_0 ------> [diam1]----->I_1----->[diam2]-----I_2 [d/s monitor]
    #
    # so I_1 = t_1*I_0, and
    # I_2 = t_2*I_1 = t_2*t_1*I_0
    #
    # following Guthrie et al J Appl Cryst.( 2017) 50 https://doi.org/10.1107/S1600576716018185
    # I_0 taken to equal bgdprof and t_1 and t_2 = 1.0 where there is no dip.
    
    if y.size == 0:
        return ttot
    if e.size == 0:
        return ttot - y
    resid = np.divide((ttot-y),e)
    return resid #lmfit requies array to be returned


def set_refinement_flags(values, fit_params):

        fit_params['sf'].vary = values["Refine Scale Factor"]
        fit_params['relsf'].vary = values["Refine Dip Intensities"]
        fit_params['setang1_alp'].vary = values["Refine Setting Angles (diam1)"]
        fit_params['setang1_bet'].vary = values["Refine Setting Angles (diam1)"]
        fit_params['setang1_gam'].vary = values["Refine Setting Angles (diam1)"]
        fit_params['setang2_alp'].vary = values["Refine Setting Angles (diam2)"]
        fit_params['setang2_bet'].vary = values["Refine Setting Angles (diam2)"]
        fit_params['setang2_gam'].vary = values["Refine Setting Angles (diam2)"]
        
        fit_params['pkCoef00'].vary = values["Refine Peak Widths (Instr.)"]
        fit_params['pkCoef01'].vary = values["Refine Peak Widths (Instr.)"]
        fit_params['pkCoef02'].vary = values["Refine Peak Widths (Instr.)"]
        fit_params['pkCoef03'].vary = values["Refine Peak Widths (Sample)"]
        fit_params['pkCoef04'].vary = values["Refine Peak Widths (Sample)"]
        fit_params['pkCoef05'].vary = values["Refine Peak Widths (Sample)"]
        fit_params['pkCoef06'].vary = False
        fit_params['pkCoef07'].vary = False
        fit_params['pkCoef08'].vary = False
        fit_params['pkCoef09'].vary = False
        fit_params['pkCoef10'].vary = False
        fit_params['pkCoef11'].vary = False
        fit_params['pkCoef12'].vary = False
        fit_params['pkCoef13'].vary = False
        fit_params['pkCoef14'].vary = False
        fit_params['pkCoef15'].vary = False
        fit_params['pkCoef16'].vary = False
        fit_params['pkCoef17'].vary = False
        fit_params['pkCoef18'].vary = False
        fit_params['pkCoef19'].vary = False
        fit_params['pkCoef20'].vary = False
        fit_params['pkCoef21'].vary = values["Refine Peak Widths (Sample)"]
        fit_params['pkCoef22'].vary = False
        fit_params['pkCoef23'].vary = False
        fit_params['pkCoef24'].vary = False
        fit_params['pkCoef25'].vary = False
        fit_params['pkCoef26'].vary = values["Refine Peak Widths (Instr.)"]
        fit_params['pkCoef27'].vary = values["Refine Peak Widths (Instr.)"]
        fit_params['pkCoef28'].vary = False
        fit_params['pkCoef29'].vary = False
        fit_params['pkCoef30'].vary = False
 
        fit_params['L2'].vary = values["Refine L2"]
        fit_params['difa'].vary = values["Refine Positional Parameters"]
        fit_params['Estr11'].vary = values["Refine Positional Parameters"]
        fit_params['Estr22'].vary = values["Refine Positional Parameters"]
        fit_params['Estr33'].vary = values["Refine Positional Parameters"]
        fit_params['Estr12'].vary = values["Refine Positional Parameters"]
        fit_params['Estr13'].vary = values["Refine Positional Parameters"]
        fit_params['Estr23'].vary = values["Refine Positional Parameters"]
        for i in range(nbgd):
            fit_params['bgd'+str(i+1).zfill(3)].vary = values["Refine Background"]
        for i in range(24-nbgd):
            fit_params['bgd'+str(nbgd+1+i).zfill(3)].vary = False
        for i in range(neqv1):
            fit_params['pkmult1_'+str(i+1).zfill(3)].vary = values["Refine Dip Intensities"]
        for i in range(neqv2):
            fit_params['pkmult2_'+str(i+1).zfill(3)].vary = False

        return fit_params    

def showResultGraph(x,y_obs,y_calc):
    
    global lam, shftlam
    global ttot
    
    plt.figure('fitting')
    plt.plot(x, y_obs, label='Observed')
    plt.plot(x, y_calc, label='Calculated')
    plt.plot(x, (y_obs - y_calc), label='Residual')
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Transmission')
    plt.grid()
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=3, mode="expand", borderaxespad=0.)
    plt.gcf().show()
    plt.pause(0.001)
    return

def input_qinputdialog(prompt=None):
    from qtpy.QtWidgets import QInputDialog

    dlg = QInputDialog()
    dlg.setInputMode(QInputDialog.TextInput)
    dlg.setLabelText(str(prompt) if prompt is not None else "")
    accepted = dlg.exec_()
    if accepted:
        return dlg.textValue()
    else:
        raise RuntimeError("User input request cancelled")
        

def do_stat_plot(w):
    global PLOTTING_WINDOW_STAT
    if PLOTTING_WINDOW_STAT is None: # or PLOTTING_WINDOW_STAT._getHeldObject() is None:
        PLOTTING_WINDOW_STAT = plotSpectrum(w,indices=[0],type = 0)
    else:
        plotSpectrum(w,indices=[0],type = 0,window=PLOTTING_WINDOW_STAT,clearWindow=True)
        
def do_CHI2_plot(w):
    global PLOTTING_WINDOW_CHI2
    if PLOTTING_WINDOW_CHI2 is None: # or PLOTTING_WINDOW_CHI2._getHeldObject() is None:
        PLOTTING_WINDOW_CHI2 = plotSpectrum(w,indices=[0],type = 0)
    else:
        plotSpectrum(w,indices=[0],type = 0,window=PLOTTING_WINDOW_CHI2,clearWindow=True)


class GatherUserInput(QDialog):
    
    # map types to (input widget, getter function, setter function)
    TYPE_TO_WIDGET = {
      bool: (QCheckBox, "isChecked", "setChecked"),
      list: (QComboBox, "currentText", "addItems")
    }
    
    def __init__(self, options):
        """
        :param options: A list 2-tuple (option_name, default_value)
        """
        QDialog.__init__(self)
        self._options = options
        self._widgets = None
        self._setup_ui()
        self.reset_requested = False
        self.exit_requested = False
    
    def values(self):
        """Return a dictionary of input values"""
        values = {}
        for option_name, _ in self._options:
            values[option_name] = self._get_value(option_name)

        return values
    
    def reset(self):
        """Set Reset Parameters to True for the refinement"""
        self.reset_requested = True
        self.accept()
        
    def exit(self):
        """Finish Refinement and Move on to report"""
        self.exit_requested = True
        self.accept()
        
    def _setup_ui(self):
        layout = QGridLayout(self)
        self.setLayout(layout)
        
        widgets = {}
        for row_index, (option_name, default_value) in enumerate(self._options):
            layout.addWidget(QLabel(option_name), row_index, 0)
            widget = self._create_widget(default_value)
            widgets[option_name] = widget
            layout.addWidget(widget, row_index, 1)
        self._widgets = widgets
        
        refineButton = QPushButton(self.tr("&Refine"))
        refineButton.clicked.connect(self.accept)
        
        exitButton = QPushButton(self.tr("&Exit"))
        exitButton.clicked.connect(self.exit)
        
        resetButton = QPushButton(self.tr("&Reset"))
        resetButton.clicked.connect(self.reset)
        
        button_box = QDialogButtonBox(); # | QDialogButtonBox.Reset);
        button_box.addButton(refineButton, QDialogButtonBox.AcceptRole)
        button_box.addButton(resetButton, QDialogButtonBox.AcceptRole)
        button_box.addButton(exitButton, QDialogButtonBox.AcceptRole)

        layout.addWidget(button_box, len(self._options), 1)

    def _create_widget(self, default_value):
        """
        Create the correct widget type for the value
        :param default_value: Default value for the entry
        """
        try:
            widget_cls, _, setter = self.TYPE_TO_WIDGET[type(default_value)]
        except KeyError:
            widget_cls, setter = QLineEdit, "setText"
            default_value = str(default_value)

        widget = widget_cls()
        # set default value
        getattr(widget, setter)(default_value)  
        return widget

    def _get_value(self, option_name):
        """
        Return the current value of the given option
        :param option_name: The name of the option
        """
        widget = self._widgets[option_name]
        value_getter = None
        for cls, getter, _ in self.TYPE_TO_WIDGET.values():
            if widget.__class__ == cls:
                value_getter = getter
                break
        if value_getter is None:
            # assume QLineEdit
            value_getter = "text"
        
        return getattr(widget, value_getter)()


def ask_user_impl(a):
    
    options = [
        ("Refine Scale Factor", a[0]),
        ("Refine Background", a[1]),
        ("Refine L2",a[2]),
        ("Refine Positional Parameters", a[3]),
        ("Refine Setting Angles (diam1)", a[4]),
        ("Refine Setting Angles (diam2)", a[5]),
        ("Refine Peak Widths (Sample)", a[6]),
        ("Refine Peak Widths (Instr.)", a[7]),
        ("Refine Dip Intensities",a[8]),
        ("Max. Iterations", a[9]) 
    ]
#    options = [
#        ("Refine Mode", ['0','1','2','3','4','5','6','7','99']),
#        ("L1", 15.0),
#        ("Refine L1", True),
#        ("Peak Type", ["Gaussian", "Lorentzian"]),
#        ("StringOption", "")
#    ]

    dialog = GatherUserInput(options)
    dialog_result = dialog.exec_()
    if dialog_result == GatherUserInput.Accepted:
        values = dialog.values()
        values["Reset parameters"] = dialog.reset_requested
        values["Exit refinement"] = dialog.exit_requested
        return values
            
        # get results
        return dialog.values()
   
    else:
        raise RuntimeError("Input cancelled")

class FitTrans(PythonAlgorithm):
    def category(self):
        return 'Diffraction\Malcolm'
        
    def PyInit(self):
        #self.declareProperty('ioioName','snap46756_tof_obs_mon_nrm.csv',Direction.Input)
#        self.declareProperty('directory','/SNS/snfs1/instruments/SNAP/IPTS-20627/shared/malcolm/data/')
        #self.declareProperty('directory','/SNS/snfs1/instruments/SNAP/IPTS-12814/shared/Malcolm2020/data/')
        self.declareProperty('directory','/SNS/SNAP/IPTS-27111/shared/')
        self.declareProperty('instrument','SNAP')
        self.declareProperty('L1 (m)','15.0') #was 1.94ms        
        self.declareProperty('initial L2 (m)','1.94') #was 1.94m #has been 0.72m also
        self.declareProperty('runNumber','51968')
        self.declareProperty('Background terms','8')
        self.declareProperty('Dip peakshape (1=Gauss,2=Lorentz,3=Psuedovoigt,4=Bk2BkExpConvPS)','3')
        
        
    def PyExec(self):
        
        '''
        Main part of the program

        M. Guthrie 16/03/2020: modified fitting procedure to use lmfit as recommended by
        Thomas Holm Rod.
        M. Guthrie 27/07/2020: 1)removed Angstrom in comments that was causing script to crash workbench
            2)removed keyword "source" from plotSpectrum calls. This was crashing script.
            3)added a max custom iteration limit to avoid infinite refinements 
            4)added a GUI to input refinement variables and
            5)added a few more options to the initial set-up GUI.
        M. Guthrie 25/08/2020: allowed for different instrument names to be specified e.g. PEARL
        M. Guthrie 22/12/2020: added matlab feature to be able to save fits parameters and
            to allow these to be re-read at the start of a refinement. This is quite critical to allow
            parameters determined with low pressure, sharper and less strained dips to be retained
            in higher pressure refinements. Also, ultra annoying, when mantid crashes, to have to repeat
            entire refinement from scratch. I am writing the parameters in the LMFIT callback, so the most
            recent parameters will always be stored if there's a crash.
            
            Also added psuedovoight peakshape option for fitting. Was getting very poor fit with dataset I'm using
            This required adding an extra peak parameter, 
            following: https://docs.mantidproject.org/nightly/fitting/fitfunctions/PseudoVoigt.html
            
        M. Guthrie 23/12/2020 - added possibility to plot calculated peak positions for each diamond
        
        M. Guthrie 19/01/2021  - added step to convert UB read from ISAW file to mantid coord system convention (z || beam)
        - Also had to correct bug in pkPosCalc + pkPosCalc_strain (transforming theta to -theta). Subsequently, calculated
        - hkl indices in FitTrans correctly match those in mantid.
            
        '''
        
#        from mantid.simpleapi import *
        import mgutilities as mgutilities
        import matplotlib.pyplot as plt
        from mantid.api import AnalysisDataService as ADS
        
        global hkl1, hkl2
        global UB1, pkcalcint1
        global UB2, pkcalcint2
        global pktype,nbgd,bgdtype
        global lam,shftlam#, y, e, TOF
        global L1
        global ttot,t1,t2,bgdprof
        global fxsamediam
        global neqv1, eqvlab1, neqv2, eqvlab2
        global function_verbose
        global runNumber
        global nit,allnit,allchi2,currentnit,currentchi2
        global directory
        global PLOTTING_WINDOW_STAT
        global PLOTTING_WINDOW_CHI2
        global maxiter
        global runNumber,directory,insTag
        # set-up simulation
        L1 = float( self.getProperty('L1 (m)').value )# m dist to centre of instrument in m
        initL2 = float( self.getProperty('initial L2 (m)').value )
        sf = 5.7e-5
        pktype = int( self.getProperty('Dip peakshape (1=Gauss,2=Lorentz,3=Psuedovoigt,4=Bk2BkExpConvPS)').value )    # pktype = 1 (Gauss), 2(Lorentz), 3(psuoedovoigt)  # 1 = Gaussian, only current working peaktype
        bgdtype = 1 # 1 = Chebychev, currently only option
        nbgd = int( self.getProperty('Background terms').value ) #number of bgd terms to use
        insTag = self.getProperty('instrument').value #name of instrument

# peak profile default values and constraints
        pkCoef = np.zeros([31,3]) #array to store default values and min + max limits of peak coefficients
        pkCoef[0,] = [8000.0,6000.0,10000.0] #alp
        pkCoef[1,] = [0.02,0.0,np.inf] #bet-0
        pkCoef[2,] = [0.00,-np.inf,np.inf] #bet-1
        pkCoef[3,] = [26,-np.inf,np.inf] #sig-0
        pkCoef[4,] = [0.000,-np.inf,np.inf] #sig-1
        pkCoef[5,] = [0.000,-np.inf,np.inf] #sig-2
        pkCoef[6,] = [0.000,-np.inf,np.inf] #gam-0
        pkCoef[7,] = [0.000,-np.inf,np.inf] #gam-1
        pkCoef[8,] = [0.000,-np.inf,np.inf] #gam-2
        pkCoef[9,] = [0.000,-np.inf,np.inf] #g1ec
        pkCoef[10,] = [0.000,-np.inf,np.inf] #g2ec
        pkCoef[11,] = [0.000,-np.inf,np.inf] #gsf
        pkCoef[12,] = [0.000,-np.inf,np.inf] #rstr  
        pkCoef[13,] = [0.000,-np.inf,np.inf] #rsta
        pkCoef[14,] = [0.000,-np.inf,np.inf] #rsca
        pkCoef[15,] = [0.000,-np.inf,np.inf] #L11
        pkCoef[16,] = [0.000,-np.inf,np.inf] #L22
        pkCoef[17,] = [0.000,-np.inf,np.inf] #L33
        pkCoef[18,] = [0.000,-np.inf,np.inf] #L12
        pkCoef[19,] = [0.000,-np.inf,np.inf] #L13
        pkCoef[20,] = [0.000,-np.inf,np.inf] #L23
        pkCoef[21,] = [0.000,0,1] #eta (Psuedovoigt mixing parameter)
        pkCoef[22,] = [0.000,-np.inf,np.inf] #spare
        pkCoef[23,] = [0.000,-np.inf,np.inf] #spare
        pkCoef[24,] = [0.000,-np.inf,np.inf] #spare
        pkCoef[25,] = [0.000,-np.inf,np.inf] #spare
        pkCoef[26,] = [1.000,1,7] #regwid1
        pkCoef[27,] = [1.000,1,7] #regwid2
        pkCoef[28,] = [0.000,-np.inf,np.inf] #
        pkCoef[29,] = [0.000,-np.inf,np.inf] #
        pkCoef[30,] = [0.000,-np.inf,np.inf] #
                              
        difa = 0.1e-10  # of order e-10
        relsf = 1.01  # default value
        lsqAlg = 'leastsq' #default is leastsq

        epsilon = np.zeros(shape=(3,3),dtype=float)
        epsilon[0,0] = 1.0
        epsilon[1,1] = 1.0
        epsilon[2,2] = 1.0
    
        epsilon[0,1] = 0.0
        epsilon[0,2] = 0.0
        epsilon[1,2] = 0.0

        # Customize constraints
        minang = -1.5 #minimum and maximum values of setting angles in degrees
        maxang = +1.5
        fxsamediam = 1  # ==1 fix intensities for given hkl to be identical for both diamonds
        #    fixmult = 0  # if ==1 peak multipliers are fixed during refinement

        function_verbose = 'n'

        # constraint notifications
        if fxsamediam ==1:
            print('*diamonds constrained to have same relative dip intensity*\n')
        else:
            print('*diamonds allowed to have different dip intensities!*')

        # Get Input Files...
        # Build input filenames
        runNumber = self.getProperty('runNumber').value
        directory = self.getProperty('directory').value
        #directory = '/Users/malcolmguthrie/Documents/SNAP/ipts22004/data/'
        fullfilename_ub1 = directory + insTag + str(runNumber) + 'UB1.mat' 
        fullfilename_ub2 = directory + insTag + str(runNumber) + 'UB2.mat'
        ioioName = insTag + str(runNumber) + '_monitors_ioio' 

        #read Transmission data from Mantid
        try:
            ws = mtd[ioioName]
            #CloneWorkspace(InputWorkspace=ioioName, OutputWorkspace='tmp_peak_ws')
        except:
            print('error opening transmission workspace:')
            print(ioioName)
            print('have you created this?')
        #ioioName = self.getProperty('ioioName').value
        #fullfilename_trans =  directory + ioioName 
        TOF1 = ws.readX(0)
        yin = ws.readY(0)
        ein = ws.readE(0)
        #TOF, yin, ein = getMANTIDdat_keepbinning(fullfilename_trans)
        print('front end of TOF in: ',TOF1[0],TOF1[1])
        print('back end of TOF in: ',TOF1[-2],TOF1[-1])
        print('data read: ')
        TOF = TOF1[0:-1,] #correct some not-understood bug that mantid is passing a TOF array with one extra datapoint
                            #EDIT: this was a histogram vs point data issue
        print('TOF shape: ',TOF.shape)
        print('yin shape: ',yin.shape)
        print('ein shape: ',ein.shape)

        # get UBs from file for both diamonds
        UB1 = np.zeros([3,3])
        isawUB1 = np.zeros([3,3])
        f = open(fullfilename_ub1,'r')
        for i in range(3):
            line = f.readline()
            line = line.strip()
            columns = line.split()
            isawUB1[i,0] = columns[0]
            isawUB1[i,1] = columns[1]
            isawUB1[i,2] = columns[2]
        f.close()

        
         
        UB2 = np.zeros([3,3])
        isawUB2 = np.zeros([3,3])
        f = open(fullfilename_ub2,'r')
        for i in range(3):
            line = f.readline()
            line = line.strip()
            columns = line.split()
            isawUB2[i,0] = columns[0]
            isawUB2[i,1] = columns[1]
            isawUB2[i,2] = columns[2]
        f.close()

        isawUB1 = isawUB1.transpose()
        isawUB2 = isawUB2.transpose()

        UB1 = isawUB1
        UB2 = isawUB2
        
        TI2M = np.array([[0,1,0],[0,0,1],[1,0,0]]) #array that transforms ISAW convention to MANTID convention
        UB1 = np.dot(TI2M,UB1)#MANTID UB
        UB2 = np.dot(TI2M,UB2)#MANTID UB
        

        print('UB matrices read from ISAW files habe been rotated to mantid convention (z along beam)')
        print(('Starting refinement for: ' + ioioName))



        #####################
        # Start work...
        #####################

        # rebin transmission data
        lam = 0.0039558 * TOF / (L1 + initL2)

        print(('wavelength limits: ' +
               str(lam[0]) + ' and ' + str(lam[len(lam) - 1])))
        minlam = 0.862#1.5#0.8
        maxlam = 3#3.6
        if minlam < lam[0]:
            minlam = lam[0]
        
        if maxlam > lam[-1]:
            maxlam = lam[-1]
            
        
        imin = np.where(lam >= minlam)[0][0]
        imax = np.where(lam >= maxlam)[0][0]
        lam = lam[imin:imax + 1]
        TOF = TOF[imin:imax + 1]  # this will be the TOF range used in fit
        y = yin[imin:imax + 1]
        e = ein[imin:imax + 1]

        #rebin temp peaks workspace to match this range of TOF
        

        # generate all allowed diamond hkls:
        allhkl = allowedDiamRefs(-7, 7, -7, 7, -7, 7)
        
        # initial conditions for crystal 1
        print('UB1\n',np.array_str(UB1, precision=4, suppress_small=True))
        
        setang1=np.zeros(3)# rotation angles applied to refined UB
        a, b, c, q = pkposcalc_strain(allhkl, UB1, setang1,epsilon)
        pkpars1 = np.column_stack((a, b, c))

        # initial conditions for crystal 2
        print('UB2\n',np.array_str(UB2, precision=4, suppress_small=True))

        setang2 = np.zeros(3)
        a, b, c, q = pkposcalc_strain(allhkl, UB2, setang2,epsilon)
        pkpars2 = np.column_stack((a, b, c))

        # purge all reflections that don't satisfy the Bragg condition and that are
        # out of wavelength calculation range...

        laminlim = lam[0]
        lamaxlim = lam[len(lam) - 1]

        nref = len(allhkl)

        k1 = 0
        k2 = 0
        hkl1 = np.zeros(shape=(0, 3))
        hkl2 = np.zeros(shape=(0, 3))
        for i in range(nref):
            
            if laminlim <= pkpars1[i][0] <= lamaxlim:  # reflection in range
                hkl1 = np.vstack([hkl1, allhkl[i]])
                k1 += 1

            if laminlim <= pkpars2[i][0] <= lamaxlim:  # reflection in range
                hkl2 = np.vstack([hkl2, allhkl[i]])
                k2 += 1

        print(('There are: ' + str(k1) + ' expected dips due to Crystal 1'))
        print(('There are: ' + str(k2) + ' expected dips due to Crystal 2'))

        # determine equivalents
        # returns array with same dim as input labelling equivs
        eqvlab1, neqv1 = findeqvs(hkl1)
        eqvlab2, neqv2 = findeqvs(hkl2)
        
        print('number of inequivs diamond 1: ',neqv1)
        print('number of inequivs diamond 2: ',neqv2)
        

        # pkpars1 = np.zeros(shape=(k, 6))   #empty array
        a, b, c, q = pkposcalc_strain(hkl1, UB1, setang1,epsilon)
        pkpars1 = np.column_stack((a, b, c))
        # Calculated ref intensities
        #    pkcalcint1 = pkintread(hkl1, (pkpars1[:, 0:3]))
        #    pkcalcint1 *= 1e-6

        # pkpars2 = np.zeros(shape=(l, 6))   #empty array
        a, b, c, q = pkposcalc_strain(hkl2, UB2, setang2,epsilon)
        pkpars2 = np.column_stack((a, b, c))
        # Calculated ref intensities
        #    pkcalcint2 = pkintread(hkl2, (pkpars2[:, 0:3]))
        #    pkcalcint2 *= 1e-6
        
        #######################################################################################
        # build and initialise lmfit parameter library
        #
        # 22/12/2020 added option to read previous fit parameters from file here.
        #
        #check if a parameter file exists, then ask if it should be used
       
        
        fit_params = Parameters()
        FitParLogName = directory + insTag + str(runNumber) + 'FitTransPars.log'
        if os.path.isfile(FitParLogName):
            input = QAppThreadCall(mgutilities.workbench_input_fn)
            useParFile = input('Notice','Previous parameter file found, use this? ([y]/n)', 'str')
            if useParFile == 'n':
                print('Found parameter file, but not using. Initialising fit parameters')              
                fit_params.add('sf', value= sf, min = 0.0)
                fit_params.add('relsf',value = relsf, min = 0.8, max = 1.2) #limits should reflect pathlength through anvils.
                fit_params.add('setang1_alp', value = 0.0, min = minang, max = maxang)
                fit_params.add('setang1_bet', value = 0.0, min = minang, max = maxang)
                fit_params.add('setang1_gam', value = 0.0, min = minang, max = maxang)
                fit_params.add('setang2_alp', value = 0.0, min = minang, max = maxang)
                fit_params.add('setang2_bet', value = 0.0, min = minang, max = maxang)
                fit_params.add('setang2_gam', value = 0.0, min = minang, max = maxang)

                fit_params.add('pkCoef00',value=pkCoef[0,0], min = pkCoef[0,1], max = pkCoef[0,2])
                fit_params.add('pkCoef01',value=pkCoef[1,0], min = pkCoef[1,1], max = pkCoef[1,2])
                fit_params.add('pkCoef02',value=pkCoef[2,0], min = pkCoef[2,1], max = pkCoef[2,2])
                fit_params.add('pkCoef03',value=pkCoef[3,0], min = pkCoef[3,1], max = pkCoef[3,2])
                fit_params.add('pkCoef04',value=pkCoef[4,0], min = pkCoef[4,1], max = pkCoef[4,2])
                fit_params.add('pkCoef05',value=pkCoef[5,0], min = pkCoef[5,1], max = pkCoef[5,2])
                fit_params.add('pkCoef06',value=pkCoef[6,0], min = pkCoef[6,1], max = pkCoef[6,2])
                fit_params.add('pkCoef07',value=pkCoef[7,0], min = pkCoef[7,1], max = pkCoef[7,2])
                fit_params.add('pkCoef08',value=pkCoef[8,0], min = pkCoef[8,1], max = pkCoef[8,2])
                fit_params.add('pkCoef09',value=pkCoef[9,0], min = pkCoef[9,1], max = pkCoef[9,2])
                fit_params.add('pkCoef10',value=pkCoef[10,0], min = pkCoef[10,1], max = pkCoef[10,2])
                fit_params.add('pkCoef11',value=pkCoef[11,0], min = pkCoef[11,1], max = pkCoef[11,2])
                fit_params.add('pkCoef12',value=pkCoef[12,0], min = pkCoef[12,1], max = pkCoef[12,2])
                fit_params.add('pkCoef13',value=pkCoef[13,0], min = pkCoef[13,1], max = pkCoef[13,2])
                fit_params.add('pkCoef14',value=pkCoef[14,0], min = pkCoef[14,1], max = pkCoef[14,2])
                fit_params.add('pkCoef15',value=pkCoef[15,0], min = pkCoef[15,1], max = pkCoef[15,2])
                fit_params.add('pkCoef16',value=pkCoef[16,0], min = pkCoef[16,1], max = pkCoef[16,2])
                fit_params.add('pkCoef17',value=pkCoef[17,0], min = pkCoef[17,1], max = pkCoef[17,2])
                fit_params.add('pkCoef18',value=pkCoef[18,0], min = pkCoef[18,1], max = pkCoef[18,2])
                fit_params.add('pkCoef19',value=pkCoef[19,0], min = pkCoef[19,1], max = pkCoef[19,2])
                fit_params.add('pkCoef20',value=pkCoef[20,0], min = pkCoef[20,1], max = pkCoef[20,2])
                fit_params.add('pkCoef21',value=pkCoef[21,0], min = pkCoef[21,1], max = pkCoef[21,2])
                fit_params.add('pkCoef22',value=pkCoef[22,0], min = pkCoef[22,1], max = pkCoef[22,2])
                fit_params.add('pkCoef23',value=pkCoef[23,0], min = pkCoef[23,1], max = pkCoef[23,2])
                fit_params.add('pkCoef24',value=pkCoef[24,0], min = pkCoef[24,1], max = pkCoef[24,2])
                fit_params.add('pkCoef25',value=pkCoef[25,0], min = pkCoef[25,1], max = pkCoef[25,2])
                fit_params.add('pkCoef26',value=pkCoef[26,0], min = pkCoef[26,1], max = pkCoef[26,2])
                fit_params.add('pkCoef27',value=pkCoef[27,0], min = pkCoef[27,1], max = pkCoef[27,2])
                fit_params.add('pkCoef28',value=pkCoef[28,0], min = pkCoef[28,1], max = pkCoef[28,2])
                fit_params.add('pkCoef29',value=pkCoef[29,0], min = pkCoef[29,1], max = pkCoef[29,2])
                fit_params.add('pkCoef30',value=pkCoef[30,0], min = pkCoef[30,1], max = pkCoef[30,2])                
                
                fit_params.add('L2',value = initL2, min = 0.0, max = 2.5)
                fit_params.add('difa',value = difa, min = -3e-10, max = +3e-10)
                fit_params.add('Estr11',value=1,min = 0.95, max = 1.05)
                fit_params.add('Estr22',value=1,min = 0.95, max = 1.05)
                fit_params.add('Estr33',value=1,min = 0.95, max = 1.05)
                fit_params.add('Estr12',value=0,min = -0.05, max = 0.05)
                fit_params.add('Estr13',value=0,min = -0.05, max = 0.05)
                fit_params.add('Estr23',value=0,min = -0.05, max = 0.05)
                fit_params.add('bgd001',value=1)
                for i in range(23):
                    fit_params.add('bgd'+str(i+2).zfill(3),value = 0)
                for i in range(neqv1):
                    fit_params.add('pkmult1_'+str(i+1).zfill(3),value = 1, min = 0.0, max = 10.0)
                for i in range(neqv2):
                    fit_params.add('pkmult2_'+str(i+1).zfill(3),value = 1, min = 0.0, max = 10.0)
            else:
                print('Loading fit parameters from file')
                with open(FitParLogName,'r') as f:
                    word = f.readline().split()
                    print('iteration',word[0],word[1])
                    
                    word= f.readline().split()
                    print('chi2 is: ',word[1])
                    
                    word= f.readline().split()
                    fit_params.add('sf', value= float(word[1]), min = 0.0)
                    
                    word= f.readline().split()
                    if int(word[1])!=fxsamediam:
                        print('WARNING: fxsamediam updated to: ',word[1])
                        fxsamediam = int(word[1])
                    
                    word= f.readline().split()
                    fit_params.add('relsf',value = float(word[1]), min = 0.8,max = 1.2) #limits should reflect ratio of pathlength through anvils.
                    
                    word= f.readline().split()
                    fit_params.add('setang1_alp', value = float(word[1]), min = minang, max = maxang)
                    
                    word= f.readline().split()
                    fit_params.add('setang1_bet', value = float(word[1]), min = minang, max = maxang)
                    
                    word= f.readline().split()
                    fit_params.add('setang1_gam', value = float(word[1]), min = minang, max = maxang)
                    
                    word= f.readline().split()
                    fit_params.add('setang2_alp', value = float(word[1]), min = minang, max = maxang)
                    
                    word= f.readline().split()
                    fit_params.add('setang2_bet', value = float(word[1]), min = minang, max = maxang)
                    
                    word= f.readline().split()
                    fit_params.add('setang2_gam', value = float(word[1]), min = minang, max = maxang)
                    
                    word= f.readline().split()
                    if int(word[1])!=pktype:
                        print('WARNING: pktype in log file:', word[1],' is different from request: ', pktype)
                        #pktype = int(word[1])
                    for i in range(31):
                        word= f.readline().split()
                        fit_params.add('pkCoef'+str(i).zfill(2),value = float(word[1]), min = pkCoef[i,1], max = pkCoef[i,2])
                    
                    word= f.readline().split()
                    fit_params.add('L2',value = float(word[1]), min = 0.0, max = 2.5)
                    
                    word= f.readline().split()
                    fit_params.add('difa',value = float(word[1]), min = -3e-10, max = +3e-10)
                    
                    word= f.readline().split()
                    fit_params.add('Estr11',value=float(word[1]),min = 0.95, max = 1.05)
                    
                    word= f.readline().split()
                    fit_params.add('Estr22',value=float(word[1]),min = 0.95, max = 1.05)
                    
                    word= f.readline().split()
                    fit_params.add('Estr33',value=float(word[1]),min = 0.95, max = 1.05)
                    
                    word= f.readline().split()
                    fit_params.add('Estr12',value=float(word[1]),min = -0.05, max = 0.05)
                    
                    word= f.readline().split()
                    fit_params.add('Estr13',value=float(word[1]),min = -0.05, max = 0.05)
                    
                    word= f.readline().split()
                    fit_params.add('Estr23',value=float(word[1]),min = -0.05, max = 0.05)
                    
                    word= f.readline().split()
                    print('bgdtype: ',word[1])
                    if int(word[1])!=bgdtype:
                        print('WARNING: background type changed to: ',word[1])
                        bgdtype = int(word[1])
                    
                      
                    word= f.readline().split()
                    print('nbgd: ',word[1])  
                    if int(word[1])!=nbgd:
                        print('WARNING: number of background terms changed to: ',nbgd)
                        #nbgd = int(word[1])
                    
                    print('bgd terms...')
                    for i in range(24):
                        word= f.readline().split()
                        print('bgd'+str(i+1).zfill(3),' ',word[1])
                        fit_params.add('bgd'+str(i+1).zfill(3),value = float(word[1]))
                    
                    
                    word= f.readline().split()
                    if int(word[1])!=neqv1:
                        print('Warning: number of inequivalent reflections for diamond 1 changed: ')
                        print('Cannot use this file! Exiting')
                        sys.exit()
                        
                    for i in range(neqv1):
                        word= f.readline().split()
                        fit_params.add('pkmult1_'+str(i+1).zfill(3),value = float(word[1]), min = 0.0, max = 10.0)
                        print(i,word[1])
                    if fxsamediam !=1:
                        word= f.readline().split()
                        if int(word[1])!=neqv2:
                            print('Warning: number of inequivalent reflections for diamond 2 changed: ')
                            print('Cannot use this file! Exiting')
                            sys.exit()
                        for i in range(neqv2):
                            word= f.readline().split()
                            fit_params.add('pkmult2_'+str(i+1).zfill(3),value = float(word[1]), min = 0.0, max = 10.0)
                    else:
                        for i in range(neqv2):
                            fit_params.add('pkmult2_'+str(i+1).zfill(3),value = 1, min = 0.0, max = 10.0)
            
        # check starting point
        else: 
                print('No parameter file found. Initialising fit parameters')              
                fit_params.add('sf', value= sf, min = 0.0)
                fit_params.add('relsf',value = relsf, min = 0.8, max = 1.2) #limits should reflect pathlength through anvils.
                fit_params.add('setang1_alp', value = 0.0, min = minang, max = maxang)
                fit_params.add('setang1_bet', value = 0.0, min = minang, max = maxang)
                fit_params.add('setang1_gam', value = 0.0, min = minang, max = maxang)
                fit_params.add('setang2_alp', value = 0.0, min = minang, max = maxang)
                fit_params.add('setang2_bet', value = 0.0, min = minang, max = maxang)
                fit_params.add('setang2_gam', value = 0.0, min = minang, max = maxang)

                fit_params.add('pkCoef00',value=pkCoef[0,0], min = pkCoef[0,1], max = pkCoef[0,2])
                fit_params.add('pkCoef01',value=pkCoef[1,0], min = pkCoef[1,1], max = pkCoef[1,2])
                fit_params.add('pkCoef02',value=pkCoef[2,0], min = pkCoef[2,1], max = pkCoef[2,2])
                fit_params.add('pkCoef03',value=pkCoef[3,0], min = pkCoef[3,1], max = pkCoef[3,2])
                fit_params.add('pkCoef04',value=pkCoef[4,0], min = pkCoef[4,1], max = pkCoef[4,2])
                fit_params.add('pkCoef05',value=pkCoef[5,0], min = pkCoef[5,1], max = pkCoef[5,2])
                fit_params.add('pkCoef06',value=pkCoef[6,0], min = pkCoef[6,1], max = pkCoef[6,2])
                fit_params.add('pkCoef07',value=pkCoef[7,0], min = pkCoef[7,1], max = pkCoef[7,2])
                fit_params.add('pkCoef08',value=pkCoef[8,0], min = pkCoef[8,1], max = pkCoef[8,2])
                fit_params.add('pkCoef09',value=pkCoef[9,0], min = pkCoef[9,1], max = pkCoef[9,2])
                fit_params.add('pkCoef10',value=pkCoef[10,0], min = pkCoef[10,1], max = pkCoef[10,2])
                fit_params.add('pkCoef11',value=pkCoef[11,0], min = pkCoef[11,1], max = pkCoef[11,2])
                fit_params.add('pkCoef12',value=pkCoef[12,0], min = pkCoef[12,1], max = pkCoef[12,2])
                fit_params.add('pkCoef13',value=pkCoef[13,0], min = pkCoef[13,1], max = pkCoef[13,2])
                fit_params.add('pkCoef14',value=pkCoef[14,0], min = pkCoef[14,1], max = pkCoef[14,2])
                fit_params.add('pkCoef15',value=pkCoef[15,0], min = pkCoef[15,1], max = pkCoef[15,2])
                fit_params.add('pkCoef16',value=pkCoef[16,0], min = pkCoef[16,1], max = pkCoef[16,2])
                fit_params.add('pkCoef17',value=pkCoef[17,0], min = pkCoef[17,1], max = pkCoef[17,2])
                fit_params.add('pkCoef18',value=pkCoef[18,0], min = pkCoef[18,1], max = pkCoef[18,2])
                fit_params.add('pkCoef19',value=pkCoef[19,0], min = pkCoef[19,1], max = pkCoef[19,2])
                fit_params.add('pkCoef20',value=pkCoef[20,0], min = pkCoef[20,1], max = pkCoef[20,2])
                fit_params.add('pkCoef21',value=pkCoef[21,0], min = pkCoef[21,1], max = pkCoef[21,2])
                fit_params.add('pkCoef22',value=pkCoef[22,0], min = pkCoef[22,1], max = pkCoef[22,2])
                fit_params.add('pkCoef23',value=pkCoef[23,0], min = pkCoef[23,1], max = pkCoef[23,2])
                fit_params.add('pkCoef24',value=pkCoef[24,0], min = pkCoef[24,1], max = pkCoef[24,2])
                fit_params.add('pkCoef25',value=pkCoef[25,0], min = pkCoef[25,1], max = pkCoef[25,2])
                fit_params.add('pkCoef26',value=pkCoef[26,0], min = pkCoef[26,1], max = pkCoef[26,2])
                fit_params.add('pkCoef27',value=pkCoef[27,0], min = pkCoef[27,1], max = pkCoef[27,2])
                fit_params.add('pkCoef28',value=pkCoef[28,0], min = pkCoef[28,1], max = pkCoef[28,2])
                fit_params.add('pkCoef29',value=pkCoef[29,0], min = pkCoef[29,1], max = pkCoef[29,2])
                fit_params.add('pkCoef30',value=pkCoef[30,0], min = pkCoef[30,1], max = pkCoef[30,2]) 
                
                fit_params.add('L2',value = initL2, min = 0.0, max = 2.2)
                fit_params.add('difa',value = difa, min = -3e-10, max = +3e-10)
                fit_params.add('Estr11',value=1,min = 0.95, max = 1.05)
                fit_params.add('Estr22',value=1,min = 0.95, max = 1.05)
                fit_params.add('Estr33',value=1,min = 0.95, max = 1.05)
                fit_params.add('Estr12',value=0,min = -0.05, max = 0.05)
                fit_params.add('Estr13',value=0,min = -0.05, max = 0.05)
                fit_params.add('Estr23',value=0,min = -0.05, max = 0.05)
                fit_params.add('bgd001',value=1)
                for i in range(23):
                    fit_params.add('bgd'+str(i+2).zfill(3),value = 0)
                for i in range(neqv1):
                    fit_params.add('pkmult1_'+str(i+1).zfill(3),value = 1, min = 0.0, max = 10.0)
                for i in range(neqv2):
                    fit_params.add('pkmult2_'+str(i+1).zfill(3),value = 1, min = 0.0, max = 10.0)

        print('Input parameters initialised, calling SimTrans3_tof')
        tmp_peak_ws = CreateWorkspace(DataX=TOF, DataY=y, NSpec=1,UnitX='TOF')# ws needed to store peakshape 4        
        LoadInstrument(tmp_peak_ws, RewriteSpectraMap = True, InstrumentName = 'SNAP')
        #res = minimize(SimTrans3_tof, fit_params, method=lsqAlg, args=(TOF, y, e),iter_cb=iteration_output,maxfev=1 )
        res = SimTrans3_tof(fit_params,TOF,y,e)
        #chi2 = res.redchi
        chi2 = np.sum(res ** 2) / res.size
        
        # display starting point in mantid
        
        #ModelFit = CreateSampleWorkspace()
        ModelInputData = CreateWorkspace(DataX=TOF, DataY=y, DataE=e, NSpec=1,UnitX='TOF')
        ModelFit = CreateWorkspace(DataX=TOF, DataY=ttot, NSpec=1,UnitX='TOF')
        ModelResidual = CreateWorkspace(DataX=TOF, DataY=y-ttot, NSpec=1,UnitX='TOF')

        #statplot = plotSpectrum(source=[ModelFit,ModelResidual,ModelInputData],indices=[0],type = 0)
        #statplot = plotSpectrum([ModelFit,ModelResidual,ModelInputData],indices=[0],type = 0)
        do_stat_plot([ModelFit,ModelResidual,ModelInputData])
        print(('Initial chi^2 is: ' + str(chi2)))
        nit = 0
        allnit = np.array([nit])
        allchi2 = np.array([10])
        currentnit = np.array([0])
        currentchi2 = np.array([10])
#        chi2vals = CreateWorkspace(DataX=allnit,DataY=allchi2,NSpec=1,UnitX='Iteration')
        chi2valsCurrentIter = CreateWorkspace(DataX=currentnit,DataY=currentchi2,NSpec=1,UnitX='Iteration')
#        chiplot = plotSpectrum([chi2vals],indices=[0],type = 0)
        #chiplot2 = plotSpectrum([chi2valsCurrentIter],indices=[0],type = 0)
        chiplot2 = do_CHI2_plot([chi2valsCurrentIter])
#######################################################################################################
# The following allows refinement to proceed within a loop that continues until user exits
# Rather than full control of all variables, the user switches on/off pre-defined sets of variables
# More control would be nice, but I think only practical with a much more sophisticated GUI than I have
# 
# At the moment, boundaries are also hardwired on most parameters.
#######################################################################################################

        RefineLoop = True
        bestFitPars = fit_params #initialise best fit to starting point
        initFitPars = fit_params #keep a copy of starting point in case reset is required.
        
        input =  QAppThreadCall(input_qinputdialog)

        while RefineLoop:                                       
            ask_user = QAppThreadCall(ask_user_impl) #request refinement mode from user
            try: 
                print("found existing refinement settings")
                values = ask_user([values["Refine Scale Factor"],values["Refine Background"],values["Refine L2"], \
                values["Refine Positional Parameters"],values["Refine Setting Angles (diam1)"],values["Refine Setting Angles (diam2)"],values["Refine Peak Widths (Sample)"], \
                values["Refine Peak Widths (Instr.)"],values["Refine Dip Intensities"],values["Max. Iterations"],values["Reset parameters"]])
            except:
                print("No existing refinement settings, using defaults")
                values = ask_user([True,False,False,False,False,False,False,False,False,200,False])
            maxiter = int(values["Max. Iterations"])
            print('max number of cycles is:',maxiter)

            if values["Reset parameters"]:
                fit_params = initFitPars #reset values
                bestFitPars = fit_params
                fit_params = set_refinement_flags(values, fit_params)
                res = SimTrans3_tof(fit_params,TOF,y,e)
                chi2 = np.sum(res ** 2. / (2 * e ** 2)) / res.size
                ModelFit = CreateWorkspace(DataX=TOF, DataY=ttot, NSpec=1,UnitX='TOF')
                ModelResidual = CreateWorkspace(DataX=TOF, DataY=y-ttot, NSpec=1,UnitX='TOF')
                print('Fit parameters have been reset')
            elif values["Exit refinement"]:
                print('Finished refining')
                RefineLoop = False
            else:
                fit_params = bestFitPars #update values but not constraints/boundaries
                fit_params = set_refinement_flags(values, fit_params)
                currentnit = np.array([0])# Reset chi2 plot for current cycle
                try: 
                   currentchi2 = np.array([res.redchi])# Reset chi2 plot for current cycle
                except:
                   currentchi2 = np.array([10])# 
                
                res = minimize(SimTrans3_tof, fit_params, method=lsqAlg, args=(TOF, y, e),iter_cb=iteration_output,maxfev=maxiter )
                print('Chi2: ',res.redchi)
                ModelFit = CreateWorkspace(DataX=TOF, DataY=ttot, NSpec=1,UnitX='TOF')
                ModelResidual = CreateWorkspace(DataX=TOF, DataY=y-ttot, NSpec=1,UnitX='TOF')
                wtres = np.square(res.residual)#np.divide(y-ttot,e)#residual divided by errors
                ModelResidualWeighted = CreateWorkspace(DataX=TOF, DataY=wtres, NSpec=1,UnitX='TOF')
                #print('sigma_0**2 is: ',res.params['pkCoef03'].value)
                #print('sigma_1**2 is: ',res.params['pkCoef04'].value)
                #print('sigma_2**2 is: ',res.params['pkCoef05'].value)
                
                # create workspace with FWHM and standard deviation 
                sdev = res.params['pkCoef03'].value**2 + res.params['pkCoef04'].value**2*np.square(TOF)\
                + res.params['pkCoef05'].value**2*np.power(TOF,4)
                sdev = np.sqrt(sdev)
                FWHM = 2*np.sqrt(2*np.log(2))*sdev
                fitwidthsdata = np.concatenate((sdev,FWHM))
                fitwidths = CreateWorkspace(DataX=TOF, DataY=fitwidthsdata, NSpec=2,UnitX='TOF')
                bestFitPars = res.params
              

        print('Final L2 is: ', res.params['L2'].value, 'min: ',res.params['L2'].min, 'max: ',res.params['L2'].max)
        print('Final Chi2: ',res.redchi)
        
        #output final fit to mantid for focusing and applying
        lamf = tof2lam(TOF,L1+res.params['L2'].value,res.params['difa'].value)
        t1WorkspaceName = insTag + runNumber + '_trns_diam1'
        t2WorkspaceName = insTag + runNumber + '_trns_diam2'
        ttWorkspaceName = insTag + runNumber + '_trns_total'
        #take fitted model and extend to 10 Ang
        #minWavOut = lamf[1]
        #maxWavOut = 10;
        #stpWavOut = 
        lam_cal_tr1 = CreateWorkspace(OutputWorkspace=t1WorkspaceName,DataX=lamf, DataY=t1, NSpec=1,UnitX='Wavelength')
        lam_cal_tr2 = CreateWorkspace(OutputWorkspace=t2WorkspaceName,DataX=lamf, DataY=t2, NSpec=1,UnitX='Wavelength')
        lam_cal_tt = CreateWorkspace(OutputWorkspace=ttWorkspaceName,DataX=lamf, DataY=ttot, NSpec=1,UnitX='Wavelength')

        lam_cal_totalTrans = CreateWorkspace(DataX=lamf, DataY=t1*t2, NSpec=1,UnitX='Wavelength')
        lam_resid = CreateWorkspace(DataX=lamf, DataY=y-ttot, NSpec=1,UnitX='Wavelength')
        lam_input = CreateWorkspace(DataX=lamf, DataY=y, NSpec=1,UnitX='Wavelength')
        #finalplot = plotSpectrum([lam_cal_tr1,lam_cal_tr2,lam_cal_totalTrans,lam_resid],indices=[0],type = 0)        
        tr1 = ADS.retrieve(t1WorkspaceName)
        tr2 = ADS.retrieve(t2WorkspaceName)
        tt = ADS.retrieve(ttWorkspaceName)
        ttrs = ADS.retrieve('lam_cal_totalTrans')
        intrs = ADS.retrieve('lam_input')
        resd = ADS.retrieve('lam_resid')
        ticks1 = ADS.retrieve('ticks1')
        ticks2 = ADS.retrieve('ticks2')
        fig, axes = plt.subplots(edgecolor='#ffffff', num='results', subplot_kw={'projection': 'mantid'})
        axes.plot(tr1, color='#ef2929', label='transmission_diam1', specNum=1)
        axes.plot(tr2, color='#000000', label='transmission_diam2', specNum=1)
        axes.plot(tt, color='#ffaa00', label='total transmission', specNum=1)
        axes.plot(intrs, color='#1f77b4', label='transmission_total', specNum=1)
        axes.plot(resd, color='#000000', label='fit_residual', specNum=1)
        axes.plot(ticks1, color='#ef2929', label='ticks1', linestyle='None', marker=2, markersize=12.0, specNum=1)
        axes.plot(ticks2, color='#000000', label='ticks2', linestyle='None', marker=3, markersize=12.0, specNum=1)
        axes.set_title('Results')
        axes.set_xlabel('Wavelength ($\\AA$)')
       
        
        plt.show()
        
        
        showRes = input('Show fit results y/[n]: ')
        if showRes.lower() == 'y':
            print(fit_report(res))
            

AlgorithmFactory.subscribe(FitTrans)

#if __name__ == "__main__":
#    FitTrans()
