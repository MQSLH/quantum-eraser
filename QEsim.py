
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange , complex64, float32,float64,complex128




def polarizerMatrix(phi,arm):

    phi = np.deg2rad(phi)

    if arm == 'upper':
        M = np.array([[np.cos(phi)**2,(np.cos(phi)*np.sin(phi)),0,0],
                    [(np.cos(phi)*np.sin(phi)),np.sin(phi)**2,0,0],
                    [0,0,1,0],
                    [0,0,0,1]],'complex128')

    elif arm == 'lower':

        M = np.array([[1,0,0,0],
                      [0,1,0,0],
                      [0,0,np.cos(phi)**2,(np.cos(phi)*np.sin(phi))],
                      [0,0,(np.cos(phi)*np.sin(phi)),np.sin(phi)**2]],'complex128')
    return M

@njit(parallel=True)
def QEResponse(delay,powerPDF,phiUpper,phiLower,phiOutput,outputPolarizer,PU,PL,Pout):

    Nphase = np.shape(delay)[0]
    Npolarizations= np.shape(powerPDF)[0]
    intensity = np.zeros((Nphase,4),'complex128')
    photonAngles = np.linspace(0,2*np.pi,np.shape(powerPDF)[0])
    

    
    BS1 = np.array([[-1,0,1,0],
                    [0,-1,0,1],
                    [1,0,1,0],
                    [0,1,0,1]],'complex128')*(1/np.sqrt(2))

    BS2 = np.array([[1,0,-1,0],
                    [0,1,0,-1],
                    [1,0,1,0],
                    [0,1,0,1]],'complex128')*(1/np.sqrt(2))

    M = np.array([[-1,0,0,0],
                    [0,-1,0,0],
                    [0,0,-1,0],
                    [0,0,0,-1]],'complex128')
    
    for ii in prange(Nphase):
        for jj in range(Npolarizations):
            
            Initial = np.array([np.cos(photonAngles[jj]),np.sin(photonAngles[jj]),0.,0.],'complex128')
            Initial = Initial/np.linalg.norm(Initial)
            
            tmp = np.exp(1j*delay[ii])
            PS = np.array([[0,0,0,0],
                            [0,0,0,0],
                            [0,0,1,0],
                            [0,0,0,1]],'complex128')
            
            PS[0,0] = tmp
            PS[1,1] = tmp


            f = powerPDF[jj]*Pout@BS2@PS@M@PL@PU@BS1@Initial
            I = (np.abs(f)**2)*(1/Npolarizations)
            intensity[ii,:] = intensity[ii,:] + I

    return intensity



if __name__== "__main__":

    # number of phase shifts
    Nphase = 100
    Nangles = 100

    #phase shift vector
    delay = np.linspace(0,2*np.pi,Nphase)

    F = np.zeros((4,Nphase),'complex128')
    visibility = np.zeros(Nangles)

    #polarizer angles in degrees
    phiUpper = np.linspace(0,180,Nangles)
    phiLower = 0
    phiOutput = 45

    polarizationExtinction = 20
    laserPratio = 1/(10**(polarizationExtinction/10))
    
    polarizationAngles = np.linspace(0,2*np.pi,100)
    ang = np.arange(-3/4,3.3,1/2)*np.pi
    pow = np.array([1., laserPratio,1., laserPratio,1., laserPratio,1., laserPratio,1.])
    powerPDF = np.interp(polarizationAngles,ang,pow)
    powerPDF = powerPDF.astype('complex128')
    #3rd polarizer in upper path
    outputPolarizer = False


    for ii in range(len(phiUpper)):
        
        PU = polarizerMatrix(phiUpper[ii],'upper')
        PL = polarizerMatrix(phiLower,'lower')
        
        if outputPolarizer:
            Pout = polarizerMatrix(phiOutput,'upper')
        else:
            Pout = np.eye(4)   
        
        intensity = QEResponse(delay,powerPDF,phiUpper[ii],phiLower,phiOutput,outputPolarizer,PU,PL,Pout)

        top = intensity[:,0]+intensity[:,1]
        bottom = intensity[:,2]+intensity[:,3]
        
        visibility[ii] = (np.max(top)-np.min(top))/(np.max(top)+np.min(top))



    fig, axs = plt.subplots(2,1)
    
    axs[0].plot(delay,top)
    axs[0].plot(delay,bottom)

    axs[0].set_ylim([0,1])

    axs[0].set_xlabel('phase difference [rad]')
    axs[0].set_ylabel('detection probability')
    axs[0].set_xticks(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])*(np.pi/4))
    axs[0].set_xticklabels(['$0$','$\pi/4$','$\pi/2$','$3\pi/4$','$\pi$','$5\pi/4$','$3\pi/2$','$7\pi/4$','$2\pi$'])
    axs[0].legend(('top screen','bottom screen'))

    axs[1].plot(phiUpper,visibility)
    axs[1].set_ylim([0,1])

    axs[1].set_xlabel('top polarizer [rad]')
    axs[1].set_ylabel('visibility')
    plt.show()