########################################################################################
# M. Guthrie 15 Jan 2020
# some functions to enable calculation of diamond UBs by choosing two reflections
#
# 26 Feb 2020 M. Guthrie replaced np.matmul with np.dot for compatibility with  older SNS numpy version
# 3 Mar 2020 M. Guthrie: improved exception handling for case when only enough reflections to get one UB
#22 April 2020 M. Guthrie: added Antonio's find_diamond_peaks algorithm
#24 July 2020 M. Guthrie changed FindAUB so that it only attempts to index peaks in the allowed
# list (i.e. those reflections that haven't been previously indexed). This was a significant bug
# Also fixed bug where UB wasn't properly written to output peaks workspace for UB2
# 14 Jan 2021 M. Guthrie: extracted last part of Jacobsen function, which ensures UB orientation
# follows a standard convension, into a separate function. This allows it to be re-called
# easily. I realised this is necessary as, the way that, findUBUsingLatticeParameters works
# means that the orientation is scrambled.
########################################################################################

    
def findDiamUB(din,Qin,Lamin,intin):
    
  import numpy as np

  #established option to run in verbose mode during beta testing, but this can be disabled here
  verb = 1 # set equal to 0 to disable verbose mode
  # start work...
  #sort input according to order of descending d-spacing
  print('test 03/03/2020!')
  srtIndx = np.argsort(din)
  #print('about to run flip')
#  srtIndx = np.fliplr(srtIndx) #flipped to give order as increasing d-spacing
  srtIndx = srtIndx[::-1]
  din = din[srtIndx]
  Qin = Qin[srtIndx,:]
  nRefin = din.size
  Hin = np.zeros((nRefin,3),dtype=int) #empty array to store indices
  
  # check for minimum d-spacing
  #for i in range(nRef):
  #    if d[i]<=0.73:
  #        print('WARNING: can''t work with reflections shorter than [422] (=0.728 Ang)')
  #        print('these will be ignored')
  #        break
  print(nRefin,' reflections read in')
  print('Filtering peaks not matching diamond d-spacings (n.b. the 511/333 at 0.687 will also be excluded')
  #At present, the code can't handle different hkl's with having the same d-spacings   
  #
  Hin = guessIndx(din,0.03) #guess indexing based on d-spacing
  dfilt = []
  Qfilt = []
  Lamfilt = []
  intfilt = []
  for i in range(din.size):
      if np.any(Hin[i,]):
          dfilt.append(din[i])
          Qfilt.append(Qin[i])
          Lamfilt.append(Lamin[i])
          intfilt.append(intin[i])
  d = np.array(dfilt)       # subsequently only operate with these filtered arrays 
  Q = np.array(Qfilt)       #
  Lam = np.array(Lamfilt)   #
  inten = np.array(intfilt)
  nRef = d.size             #
  print('This leaves: ',nRef,' usable reflections')
  ubLab = np.zeros(nRef,) # a label to identify unique crystals (each with their own UB's)
  H = np.zeros((nRef,3),dtype=int) #empty array to store indices

  print('calculating UB1:')
  UB1,ubLab,Hr1 = findAUB(H,Q,d,ubLab,1,verb)#allows user to choose two reflection, calc UB, then attempts to index
                                             #entire (filtered) peaks list with this.
  if ~np.any(UB1):
      print('Error in findAUB: not enough reflections to fit UB1')
  print('determine second UB with remaining reflections:')
  UB2,ubLab,Hr2 = findAUB(H,Q,d,ubLab,2,verb)
  if ~np.any(UB2):
      print('Error in findAUB: not enough reflections to fit UB2')
  
  print(' ')
  print(' REF|     h      k      l|     lam|   dspac||        int|  UB')
  for i in range(nRef):
      if ubLab[i]==1:
        print('%4i|%6.3f %6.3f %6.3f| %7.4f| %7.4f|| %10.0f| %3.0f' %(i,Hr1[i,0],Hr1[i,1],Hr1[i,2],Lam[i],d[i],inten[i],ubLab[i]))
      elif ubLab[i]==2:
        print('%4i|%6.3f %6.3f %6.3f| %7.4f| %7.4f|| %10.0f| %3.0f' %(i,Hr2[i,0],Hr2[i,1],Hr2[i,2],Lam[i],d[i],inten[i],ubLab[i]))
      else:
        print('%4i|%6.3f %6.3f %6.3f| %3.0f' %(i,0,0,0,0))
      #,Q[i,0],Q[i,1],Q[i,2]
  return UB1,UB2,ubLab,Hr1,Hr2,d,Q,Lam,inten

#################################################################
def chooseIntFromArray(anArray,string):
  import numpy as np
  #ensure that anArray contains integers
  #anArray = int(anArray)
  while True:
      try: 
          val = int(input(string))#first check that input is an int
          chk = anArray-val
          inList = np.where(chk==0)[0];
          allowed = inList.size      
          (1/allowed)
          break #division by zero means value not found in list and
      except:
          print('Error: Input must be an integer in this list:')
          print(anArray)

  return val

################################################################
def jacobsen(h1,Xm_1,h2,Xm_2):
  import numpy as np
#Jacobsen - Implementation of method articulated in RA Jacobsen 
#Zeitschrift fur Krystallographie vol. 212 pp 99-102 (1997).

# updated 12/01/2017
#
# Note 1: !!!! mantidplot scales reciprocal space coordinates with a factor of
# 2pi. In contrast, ISAWev does not.This algorithm assumes the mantidplot
# convention!!!!
#
# update 11/02/2020: added code to impose convention that diamond a axis is anti-
# parallel to the beam.
#

#h1 and h2 are vertical 3x1 coordinate matrices containing h,k,l for two
#reflections
#Xm_1 and Xm_2 are the corresponding coordinate matrices measured in the
#(Cartesian) diffractometer frame Xm (these are known by mantid for 
#a calibrated instrument)

# Note 2: UB's generated using above have diamond a-axis anti parallel to
# z, however, indexing has been chosen so that it should be parallel to x.
# possible explanation: the coordinate systen used by mantid has z anti-parallel
# to the beam? If this is the case, then transforming Xm_1 and Xm_2 by a 
# rotation of +90 deg about y should correct for this do this here:

# yang = 90; %angle of rotation about y
# Ry = [cosd(yang) 0 sind(yang);0 1 0; -sind(yang) 0 cosd(yang)];
# Xm_1 = Ry*Xm_1;
# Xm_2 = Ry*Xm_2;

#First check indexing of input reflections by checking angle: 
#angle between reciprocal lattice vectors

  if np.array_equal(h1,h2):
      return np.zeros((3,3))

#calculate alp, the expected angle between reflections
  alp = np.arccos(np.dot(h1,h2)/(np.linalg.norm(h1)*np.linalg.norm(h2)))
  alp = np.degrees(alp)
#calculate  bet, observed angle between reflections
  bet = np.arccos(np.dot(Xm_1,Xm_2)/(np.linalg.norm(Xm_1)*np.linalg.norm(Xm_2)))
  bet = np.degrees(bet)
#  print('expected angle (deg): ',alp)
#  print('observed angle (deg): ',bet)
  pi = 3.1415926535
  a = 3.567 # diamond lattice parameter
  ast = 2*pi/a #reciprocal lattice parameter (mantid convention)

#use the conventional definition of the b transformation matrix to give
#Jacobsons arbitrary cartesian frame. This is defined, for example, in
#Giacovazzo, Eqn 2.31a:
#B matrix: [1/a 0 0; -cos(gam)/(a sin(gam)) 1/(b sin(gam)) 0;
#         a*cos(bet*) b*cos(alp*) c*
#
# this is nice and simple if we're cubic, so alp=bet=gam=alp*=bet*=gam*
# and a = b = x
#
#B = [1/a 0 0; 0 1/a 0; 0 0 1/a] = [a* 0 0; 0 a* 0; 0 0 a*];
#Note Giacovazzo doesn't use the 2pi factor
#
#
#14 MARCH 2017 modified B to match MANTID convention  

  B = np.array([[ast,0.,0.],[0.,ast,0.],[0.,0.,ast]])
  

# Need 3 vectors do define U matrix. Have two measured ones, take cross
# product of these as third matrix. 

  Xm_g = np.cross(Xm_1,Xm_2)
  
  #concatenate into a single matrix:

  Xm = np.array([Xm_1,Xm_2,Xm_g]);
  Xm = np.transpose(Xm)
  
  
  #Vector Q1 is described in reciprocal space by its coordinate matrix h1.
  #By definition, operating on h1 with B gives its coordinates X1_a in the 
  #arbitrary Cartesian frame "Xa". Likewise for Q2,h2 and Xa_2
  Xa_1 = np.dot(B,h1)
  #
  Xa_2 = np.dot(B,h2)  # previously used matmul, but this doesn't seem to be suppported
  # again calculate cross product to give the third vector
  Xa_g = np.cross(Xa_1,Xa_2)
  #
  #and concatenated into a single matrix:
  Xa = np.array([Xa_1,Xa_2,Xa_g])
  Xa = np.transpose(Xa)

#
# These coordinates are related to the lab frame by a rotation R 
# given by:
  R = np.dot(Xa,np.linalg.inv(Xm)) #using dot instead of np.matmul
  U = np.linalg.inv(R)
  
  UB = np.dot(U,B)
  return UB
#  print('')
#  print('/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/')
#  print('Jacobsen calculated UB:')
#  print(UB)
#  print('/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/')
  
  
def UBStandardOrient(UB):
  import numpy as np
# now that UB is obtained, figure out orientation of diamond unit cell axes to beam
# and impose convention that a axis is pointing anti-parallel to beam
# nb this is arbitrary, but it will be helpful to consistently have the same orientation
# in order to easily map calculated dips to observed diamond reflections in the 
# diffraction detectors
# print('Diamond unit vectors along diamond cell axes:')
  
  a = 3.567 # diamond lattice parameter
  ast = 2*np.pi/a #reciprocal lattice parameter (mantid convention)
 
  B = np.array([[ast,0.,0.],[0.,ast,0.],[0.,0.,ast]])
  Binv = np.linalg.inv(B)
  U = np.dot(UB,Binv) #UB*B*B-1 should be U
  a = np.zeros([3,3]) #3x3 array to contain unit vectors of cell axes
  a[0,:] = np.dot(UB,[1,0,0])/np.linalg.norm(np.dot(UB,[1,0,0]))
  a[1,:] = np.dot(UB,[0,1,0])/np.linalg.norm(np.dot(UB,[0,1,0]))
  a[2,:] = np.dot(UB,[0,0,1])/np.linalg.norm(np.dot(UB,[0,0,1]))
#  print('a axis: ',a[0,:])
#  print('b axis: ',a[1,:])
#  print('c axis: ',a[2,:])
  
#interested to find the vector that is closest to parallel to the z-axis (beam)
#this will have the largest absolute value of dot product with [1,0,0]
  zDot = np.zeros([3,]) #3x1 array to contain z-coordinates of each unit cell axis
  cartV = np.array([[0,0,1],[0,0,1],[0,0,1]]) 
  for i in range(3):
    zDot[i] = np.dot(a[i,:],cartV[i,:]) #dot product of all lattice vectors with z axis
# find largest absolute value of zDot
  zMax = np.amax(np.absolute(zDot))
  anti = zMax<=0; #is true if axis is antiparallel to beam 
#  print('Anti is: ',anti)
#  print('zmax: ',zMax)
  k = np.where(np.absolute(zDot) == zMax) #find index of axis vector closest to z-axis 
#  print('index is: ',k[0])  
  anti = zDot[k[0]]<=0 #is true if axis is antiparallel to beam 
#  print('Anti is: ',anti)

  
  if k[0]==0:
      if anti:
#          print('diamond a-axis is closest to beam and is antiparallel') #desired case
          UB = UB #already what I want
      else:
#          print('diamond a-axis is closest to beam and is parallel')#
          # rotate  about either y or z axis by 180 degrees
          R = cartRotDeg(180,[0,1,0])
          RB = np.dot(R,B)
          UB = np.dot(U,RB)
  elif k[0]==1: 
      if anti:
#          print('diamond b-axis is closest to beam and is antiparallel')
          # rotate about z axis -90 degrees
          R = cartRotDeg(90,[0,0,1])
          RB = np.dot(R,B)
          UB = np.dot(U,RB)
      else:
#          print('diamond b-axis is closest to beam and is parallel')
          # rotate about z axis +90 degrees
          R = cartRotDeg(-90,[0,0,1])
          RB = np.dot(R,B)
          UB = np.dot(U,RB)
  elif k[0]==2:
      if anti:
#          print('diamond c-axis is closest to beam and is antiparallel')
          # rotate about y axis -90 degrees
          R = cartRotDeg(-90,[0,1,0])
          RB = np.dot(R,B)
          UB = np.dot(U,RB)
      else:
#          print('diamond c-axis is closest to beam and is parallel')
          # rotate about y axis +90 degrees
          R = cartRotDeg(90,[0,1,0])
          RB = np.dot(R,B)
          UB = np.dot(U,RB)
  print('')
  print('/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/')
  print('UB with conventional orientation:')
  print(UB)
  print('/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/')
  a[0,:] = np.dot(UB,[1,0,0])/np.linalg.norm(np.dot(UB,[1,0,0]))
  a[1,:] = np.dot(UB,[0,1,0])/np.linalg.norm(np.dot(UB,[0,1,0]))
  a[2,:] = np.dot(UB,[0,0,1])/np.linalg.norm(np.dot(UB,[0,0,1]))
  print('In ICS, a axis is: ',a[0,:])
  print('In ICS, b axis is: ',a[1,:])
  print('In ICS, c axis is: ',a[2,:])
  return UB
  
def cartRotDeg(ang,vect):
# returns a rotation matrix to rotate by an angle ang, specified in degrees, about an arbitrary
# vector vect. matrix taken from wikipedia:
# https://en.wikipedia.org/wiki/Rotation_matrix#Basic_rotations

  import numpy as np
  t= np.radians(ang);
  ux = vect[0]/np.linalg.norm(vect)
  uy = vect[1]/np.linalg.norm(vect)
  uz  = vect[2]/np.linalg.norm(vect)
  cost  = np.cos(t)
  sint = np.sin(t)

  R = np.array([[cost   +  ux**2*(1-cost), ux*uy*(1-cost)-uz*sint, ux*uz*(1-cost)+uy*sint],
  [uy*ux*(1-cost)+uz*sint, cost+uy**2*(1-cost), uy*uz*(1-cost)-ux*sint],
  [uz*ux*(1-cost)-uy*sint, uz*uy*(1-cost)+ux*sint, cost+uz**2*(1-cost)]]);

  return R

def equivMatch(refh,hkl,gam,tol):
#equivMatch: accepts two reflection hkl indices: refh a fixed index and hkl, a generic index, the observed angle between these two reflections, gam, and a tolerance, tol, that defines an acceptable difference between observed and calculated reflections 
  import numpy as np  

  allEqv,nEqv = m3mEquiv(hkl)

  match = np.zeros(nEqv)
  nmatch = 0
  for i in range(nEqv): #run over all sets of equivalent indices, calculating angle between these and the reference reflection (refh)
      h1 = allEqv[i];
      bet = np.arccos( np.dot(refh,h1)/(np.linalg.norm(refh)*np.linalg.norm(h1)))
      bet = np.degrees(bet)
      dif = np.absolute(bet-gam)
#      print('Equiv ',i,' is ',h1, 'making angle ',bet,' with reference reflection ',refh, ' obs angle is ',gam,' dif: ',dif)
      if dif<=tol:
          match[i]=1
          nmatch = nmatch+1
#          print('this matches!')
          
  if nmatch==0:
      hklmatch = np.array([0,0,0])
  else:
      ind = np.where(match==1)
      hklmatch = allEqv[ind]
#  print('type hklmatch: ', type(hklmatch))
  return hklmatch
  
def m3mEquiv(hkl):
#hkl is a single 1x3 vector containing an h,k,l set for a diamond reflection
#the general 48 symmetry equivalent reflections for the m-3m Point 
#group are calculated and then purged so that only distinct
#equivalents remain and these are returned
        
#lastly, if apar is true, the convention that the diamond a is 
#parallel to the beam is hardwired in.
  import numpy as np
  h = hkl[0]
  k = hkl[1]
  l = hkl[2]
  all = [h,k,l,
  -h,-k,l,
  -h,k,-l,
  h,-k,-l,
  k,h,-l,
  -k,-h,-l,
  k,-h,l,
  -k,h,l,
  
  l,h,k,
  l,-h,-k,
  -l,-h,k,
  -l,h,-k,
  -l,k,h,
  -l,-k,-h,
  l,k,-h,
  l,-k,h,
  
  k,l,h,
  -k,l,-h,
  k,-l,-h,
  -k,-l,h,
  h,-l,k,
  -h,-l,-k,
  -h,l,k,
  h,l,-k,
  
  -h,-k,-l,
  h,k,-l,
  h,-k,l,
  -h,k,l,
  -k,-h,l,
  k,h,l,
  -k,h,-l,
  k,-h,-l,
  
  -l,-h,-k,
  -l,h,k,
  l,h,-k,
  l,-h,k,
  l,-k,-h,
  l,k,h,
  -l,-k,h,
  -l,k,-h,
  
  -k,-l,-h,
  k,-l,h,
  -k,l,h,
  k,l,-h,
  -h,l,-k,
  h,l,k,
  h,-l,-k,
  -h,-l,k]
  
  all = np.reshape(all,(48,3))
  lab = -np.ones((48,), dtype=int) #labels unique reflections
  nlab = -1
  for i in range(48):
      
      if lab[i]==-1: #then reflection hasn't been checked yet, so check it...
        nlab = nlab + 1
        lab[i] = nlab
        for j in range(i+1,48): #check all remaining reflections
          dif = all[i]-all[j]
          if np.linalg.norm(dif)==0: #reflection is identical to reflection i
            lab[j]=lab[i]           
            
              #show labelling of reflection
  for i in range(48):
#      print('ref %3i:%3i %3i %3i %3i' %(i,all[i,0],all[i,1],all[i,2],lab[i]))
  

  # create reduced array containing only unique hkls
    eqvs = np.zeros((nlab+1,3), dtype=int)
    for i in range(nlab+1):
#     print('checking reflections sharing label: ',i) 
     k = np.where(lab==i) #indices of all 
#     print('these have indices: ',k)
#     print('first value of k is',k[0][0])
     eqvs[i,:] = all[k[0][0],:]
     del k
  return eqvs,nlab+1
  
def guessIndx(d,tol):
# accepts n d-spacings and returns guess at indexing just by looking at d-spacing values and finding expected diamond reflections that match within defined tolerance
  import numpy as np
  
  nref = d.size
  dref = np.array([2.0593,1.2611,1.0754,0.8917,0.8183,0.7281,0.6305,0.6029]) # truncated list of refs
  href = np.array([[-1,1,1],[-2,2,0],[-3,1,1],[-4,0,0],[-3,3,1],[-4,2,2],[-4,4,0],[-5,3,1]])
  
  #note: [5,1,1] has same d-spacing as [-3,3,3] need to check during next step...
  
  h = np.zeros([nref,3])
  for i in range(nref):
      reltol = tol*d[i] #relative tolerance scales linearly with d-spacing
      delta = np.absolute(dref-d[i]) #array with same dimension as dref
      hit = np.array(np.where(delta<reltol))
      if hit.size==0:
          h[i,:]=[0,0,0]
      elif hit.size>1:
          print('Error in guessIndx: tolerance is too large!')
          print('failed on reflection: ',i)
          print('d-spacing: ',d[i])
          exit()
      elif hit.size==1:
          h[i,:]=href[hit,:]
          
  return h 
          
def findAUB(Hin,Qin,din,ubLab,crysNo,verb):
  import numpy as np
  from mantidqt.utils.qt.qappthreadcall import QAppThreadCall
  input = QAppThreadCall(workbench_input_fn)
  
  # software will work to index currently unindexed reflection i.e. those with ublab==0
  # On successful completion, it will label indexed reflections with crysNo
  #
  # first step is to count unindexed reflections 
  # 
  nIn = din.size
  refNoIn = np.array(range(nIn))
  nRef = 0
  for i in range(nIn):
      if ubLab[i]==0: #then reflection is unindexed, so it should be processed
          nRef = nRef + 1

  # check there are sufficient reflections and stop if not

  print('of ',nIn,' input reflections, will fit UB to ',nRef,' of these')

        
  if nRef <2:
      print('insufficient available reflections to calculate UB!')
      UB1 = np.zeros([3,3])
      hindx = np.zeros([nIn,3])
      return UB1,ubLab,hindx #return empty UB1 for subsequent processing

         
  #now create arrays to store these (I'm sure this can be done in one loop, but don't know how)          
  H = np.zeros((nRef,3),dtype=int)
  refNo = np.zeros((nRef,),dtype=int)
  Q = np.zeros((nRef,3),dtype=float)
  d = np.zeros((nRef,),dtype=float)
  
  # now populate these arrays
  nRef = 0
  for i in range(nIn):
      if ubLab[i]==0:
          nRef = nRef+1
          H[nRef-1,:]=Hin[i,:]
          Q[nRef-1,:]=Qin[i,:]
          d[nRef-1]=din[i]
          refNo[nRef-1]=refNoIn[i]
 
  H = guessIndx(d,0.03) #guess indexing based on d-spacing
  print('Input reflections with provisional indexing:')
  print(' REF|  h   k   l| d-spac(Ang)|  UB|Q')
  for i in range(nRef):
      print('%4i|%3i %3i %3i| %11.4f|%4i|%9.3f%7.3f%7.3f' %(refNo[i],H[i,0],H[i,1],H[i,2],d[i],ubLab[refNo[i]],Q[i,0],Q[i,1],Q[i,2]))
  #VERBOSE
  if verb==1:
    #print('Calling Qt:')
    nRef1 = input('Choose a reflection','Enter reflection number','int')
    #print('Qt returned value of: ',nRef1) 
    #nRef1 = chooseIntFromArray(refNo,'Enter reflection number from list:') # will only accept value from refNo array
  else:
    nRef1 = 0 #default is longest d-spacing reflection
  #need to convert to nRef1 to correspond to correct index in reduced array that's being fitted
  nRef1 = np.where(refNo==nRef1)
  #print('correct index: ',nRef1[0][0])

  # check for indexing consistent with reference reflection and, in verbose mode, print output
  beta = np.zeros(nRef)  # will contain observed angles between a given reflection and reference reflection

  if verb==1:
      print(' ')
      print('Re-indexed for consistency with reference reflection:')
      print(' ')
      print(' REF|  h   k   l|    d(Ang)|    obs|  calc|')
  for i in range(nRef):
  #    print('checking ref: ',i)  #
      if i==nRef1[0][0]:
          beta[i] = 0.0
          if verb==1:
                    print('%4i|%3i %3i %3i| %9.4f| %6.1f|%6.1f| REFERENCE' %(refNo[i],H[i,0],H[i,1],H[i,2],d[i],beta[i],0.0))
      else:
          beta[i]=np.arccos(np.dot(Q[nRef1],Q[i])/(np.linalg.norm(Q[nRef1])*np.linalg.norm(Q[i])))
          beta[i]=np.degrees(beta[i])
          hklHit = equivMatch(H[nRef1],H[i],beta[i],4.0)
          #print('hklHit is:')
          #print(hklHit)
  # array includes all candidate indices, here we arbirarily chose the first of these.
  # because of how equivalents are generated, this should always give systematic results
          if np.linalg.norm(hklHit)==0:
              H[i]=[0,0,0]# no consistent indexing found
              if verb==1:
                  print('%4i|%3i %3i %3i| %9.4f| %6.1f|' %(refNo[i],H[i,0],H[i,1],H[i,2],d[i],beta[i]))
          else:
              calcang = np.arccos( np.dot(H[nRef1],hklHit[0,:])/(np.linalg.norm(H[nRef1])*np.linalg.norm(hklHit[0,:])))
              calcang = np.degrees(calcang)
              H[i]=hklHit[0,:]
              if verb==1:
                  print('%4i|%3i %3i %3i| %9.4f| %6.1f|%6.1f| poss. index' %(refNo[i],H[i,0],H[i,1],H[i,2],d[i],beta[i],calcang))

  if verb==1:
    nRef2 = input('Choose a reflection','Enter reflection number','int')
    #nRef2 = chooseIntFromArray(refNo,'Choose 2nd reflection to calculate UB:') # will only accept value from refNo array
  else:
      print('non-verbose mode doesn''t work yet!')
      stop()

  #need to convert to nRef1 to correspond to correct index in reduced array that's being fitted
  nRef2 = np.where(refNo==nRef2)
  print('correct index: ',nRef2[0][0])
  

  UB1 = jacobsen(H[nRef1[0][0],:],Q[nRef1[0][0],:],H[nRef2[0][0],:],Q[nRef2[0][0],:])

   
  #apply UB to all reflections
  invUB1 = np.linalg.inv(UB1)
  Qint = Qin.transpose()
  hindx = np.dot(invUB1,Qint) #replaced matmul with dot
  hindx = hindx.transpose() #contains indices of all reflections using UB1
  #
  # On the basis of a defined tolerance, decided which reflections in list are correctly
  # indexed

  indTol = 0.12 #:
  nIndx1 = 0
  for i in range(nIn): 
      if ubLab[i]==0: 
          dif = np.sum(np.absolute(hindx[i,:]-np.around(hindx[i,:])))
          if dif<=3*indTol:  # all indices are within tolerance
              ubLab[refNoIn[i]]=crysNo
              nIndx1 = nIndx1+1

  print(' ')
  print('UB calculated and ',nIndx1,'reflections indexed within tolerance of.', indTol)
  print(' ')
#  print(' REF|     h      k      l|  UB')
#  for i in range(nIn):
#      print('%4i|%6.3f %6.3f %6.3f| %3.0f' %(refNoIn[i],hindx[i,0],hindx[i,1],hindx[i,2],ubLab[refNoIn[i]]))
  
  return UB1,ubLab,hindx            

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


def find_diamond_peaks(wks_d, dens_thresh): # from Antonio
    
    ConvertToMD(InputWorkspace=wks_d, QDimensions='Q3D', dEAnalysisMode='Elastic', Q3DFrames='Q_lab', OutputWorkspace='test_MD')
    FindPeaksMD(InputWorkspace='test_MD', PeakDistanceThreshold=0.25, MaxPeaks=50, DensityThresholdFactor=dens_thres , OutputWorkspace='Peaks_MD')
    IntegratePeaksMD(InputWorkspace='test_MD', PeakRadius=0.12, BackgroundInnerRadius=0.14, BackgroundOuterRadius=0.17, PeaksWorkspace='Peaks_MD', OutputWorkspace='Peaks_MD_Integrated')
    FilterPeaks(InputWorkspace='Peaks_MD_Integrated', OutputWorkspace='Peaks_MD_Integrated_filtered', FilterVariable='Signal/Noise', FilterValue=40, Operator='>')
   
    return Peaks_MD_Integrated_filtered  
