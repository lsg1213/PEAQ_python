import numpy as np
import time


'''
Original code: https://github.com/stephencwelch/Perceptual-Coding-In-Python/tree/master/PEAQPython
'''
class PQEval(object):
    def __init__(self, Amax = 1, Fs= 48000, NF= 2048):
        #Amax is maximum signal amplitude, Fs is sampling frequency
        #Setup parameters and precompute quantities we'll need.
        self.Fs = Fs
        self.NF = NF

        #Hardcode the louness scalling params:
        fcLoudness = 1019.5
        Lp = 92

        #Set up the window (including all gains)
        self.GL = self.PQ_GL(NF= self.NF, Amax = Amax, fcN = fcLoudness/self.Fs, Lp = Lp)

        #Precompute hann window:
        self.hw = self.GL*self.PQHannWin(self.NF)

        #Precompute frequency vector:
        self.f = np.linspace(0, self.Fs//2, self.NF//2+1)

        #Outer and middle ear weighting:
        self.W2 = self.PQWOME (self.f)

        #Critical band constants:
        self.Nc, self.fc, self.fl, self.fu, self.dz = self.PQCB()

        #Internal Noise:
        self.EIN = self.PQIntNoise(self.fc)

        #Precompute normalization for frequency spreading:
        self.Bs = self.PQ_SpreadCB(np.ones(self.Nc), np.ones(self.Nc))

        # Allocate storage
        self.Eb = np.zeros((2, self.Nc))
        self.Xw2 = np.zeros((2, self.NF//2+1))
        self.XwN2 = np.zeros(self.NF//2+1)
        self.E = np.zeros(self.Eb.shape)
        self.Es = np.zeros((2, self.Nc))

        #Precompute for PQ Group:
        self.df = float(self.Fs) / self.NF
        self.Emin = 1e-12
        
        self.U = np.zeros((self.NF//2+1, self.Nc))

        for k in range(self.NF//2+1):
            for i in range(self.Nc):
                temp = (np.amin([self.fu[i], (k+0.5)*self.df]) - np.amax([self.fl[i], (k-0.5)*self.df])) / self.df
                self.U[k, i] = np.amax([0, temp])

        # check FLAG, False means first operation
        self.check_PQmodPatt = False
                
    def PQDFTFrame(self, x):
        # Window the data
        xw = self.hw * x

        # DFT (output is real followed by imaginary)
        X = self.PQRFFT(xw, self.NF, 1)

        # Squared magnitude
        X2 = self.PQRFFTMSq(X, self.NF)
        
        return X2

    def PQ_excitCB(self, X2):
        # Critical band grouping and frequency spreading

        # Outer and middle ear filtering
        self.Xw2[0,:] = self.W2 * X2[0,0:self.NF//2+1]
        self.Xw2[1,:] = self.W2 * X2[1,0:self.NF//2+1]

        # Form the difference magnitude signal
        self.XwN2 = self.Xw2[0,:] - 2*np.sqrt(self.Xw2[0,:]*self.Xw2[1,:]) + self.Xw2[1,:]
        
        # Group into partial critical bands
        self.Eb[0,:] = self.PQgroupCB(self.Xw2[0,:])
        self.Eb[1,:] = self.PQgroupCB(self.Xw2[1,:])
        self.EbN     = self.PQgroupCB(self.XwN2)

        # Add the internal noise term => "Pitch patterns"
        self.E[0,:] = self.Eb[0,:] + self.EIN
        self.E[1,:] = self.Eb[1,:] + self.EIN

        # Critical band spreading => "Unsmeared (in time) excitation patterns"
        self.Es[0,:] = self.PQspreadCB(self.E[0,:])
        self.Es[1,:] = self.PQspreadCB(self.E[1,:])
        
        return self.EbN, self.Es

    def PQgroupCB(self, X2):
        # Group a DFT energy vector into critical bands
        # X2 - Squared-magnitude vector (DFT bins)
        # Eb - Excitation vector (fractional critical bands)

        Eb = np.dot(X2,self.U)
        Eb[Eb<self.Emin] = self.Emin
        
        return Eb

    def PQspreadCB(self, E):
        # Spread an excitation vector (pitch pattern) - FFT model
        # Both E and Es are powers	    
        Es = self.PQ_SpreadCB(E, self.Bs)
        
        return Es

    def PQ_SpreadCB(self, E, Bs):
        e = 0.4 # Commonly used power value
        
        # Initialize arrays for storage. These values are used
        # in each iteration (summed over, multiplied, raised to
        # powers, etc.) when computing the spread Bark-domain
        # energy Es.
        #
        # aUCEe is for the product of bin-dependent (index l)
        # term aC, energy-dependent (E) term aE, and
        # term aU.
        #
        # Ene is (E[l]/A(l,E[l]))^e, stored for each index l
        #
        # Es is the overall spread Bark-domain energy
        #

        aUCEe = np.zeros(self.Nc)
        Ene = np.zeros(self.Nc)
        Es = np.zeros(self.Nc)
        
        # Calculate energy-dependent terms
        aL = 10**(2.7*self.dz)

        for l in range(self.Nc):
            aUC = 10**((-2.4 - 23/self.fc[l])*self.dz)
            aUCE = aUC * (E[l]**(0.2*self.dz))
            gIL = (1 - aL**(-1*(l+1))) / (1 - aL**(-1))
            gIU = (1 - (aUCE)**(self.Nc-l)) / (1 - aUCE)
            En = E[l] / (gIL + gIU - 1)
            aUCEe[l] = aUCE**(e)
            Ene[l] = En**(e)
        
        # Lower spreading
        Es[self.Nc-1] = Ene[self.Nc-1]
        aLe = aL**(-1*e)
        for i in range((self.Nc-2),-1,-1):
            Es[i] = aLe*Es[i+1] + Ene[i]
        
        
        # Upper spreading (i > m)
        for i in range(0,(self.Nc-1)):
            r = Ene[i]
            a = aUCEe[i]
            for l in range((i+1),self.Nc):
                r = r*a
                Es[l] = Es[l] + r
                
        # Normalize the values by the normalization factor
        for i in range(0,self.Nc):
            Es[i] = (Es[i]**(1/e)) / Bs[i]
            
        return Es

    def PQ_timeSpread(self, Es, Ef):
        
        Nadv = self.NF//2
        Fss = float(self.Fs)/Nadv
        tau_100 = 0.030
        tau_min = 0.008
        alpha, beta = self.PQtConst(tau_100, tau_min, self.fc, Fss)
        
        # Allocate storage
        Ehs = np.zeros(self.Nc)
        # Time domain smoothing
        for i in range(self.Nc):
            Ef[i] = alpha[i]*Ef[i] + (1-alpha[i])*Es[i]
            Ehs[i] = max(Ef[i],Es[i])
        
        return Ehs, Ef

    def PQtConst(self, tau_100, tau_min, fc, Fss):
        # Tau values in units of seconds
        #tau_100 = 0.030
        #tau_min = 0.008
        
        tau = np.zeros(len(fc))
        alpha = np.zeros(len(fc))
        
        tau = tau_min + (np.divide(float(100),fc))*(tau_100 - tau_min)
        alpha = np.exp(np.divide(-1./Fss,tau))
        beta = 1. - alpha

        return alpha, beta

    #Internal noise:
    def PQIntNoise (self, f):
        INdB = 1.456 * (f / 1000.)**(-0.8)
        EIN = 10**(INdB / 10.)
        return EIN

    #Method to make hanning window, given lenth of window:	
    def PQHannWin(self, NF):
        n = np.arange(0, NF)
        hw = 0.5*(1-np.cos(2*np.pi*n/(NF-1)))
        return hw 

    def PQRFFT (self, x, N, ifn):
        # Calculate the DFT of a real N-point sequence or the inverse
        # DFT corresponding to a real N-point sequence.
        # ifn > 0, forward transform
        #          input x(n)  - N real values
        #          output X(k) - The first N/2+1 points are the real
        #            parts of the transform, the next N/2-1 points
        #            are the imaginary parts of the transform. However
        #            the imaginary part for the first point and the
        #            middle point which are known to be zero are not
        #            stored.
        # ifn < 0, inverse transform
        #          input X(k) - The first N/2+1 points are the real
        #            parts of the transform, the next N/2-1 points
        #            are the imaginary parts of the transform. However
        #            the imaginary part for the first point and the
        #            middle point which are known to be zero are not
        #            stored. 
        #          output x(n) - N real values


        if (ifn > 0):
            X = np.fft.fft (x, N)
            XR = np.real(X[0:N//2+1])
            XI = np.imag(X[1:N//2-1+1])
            X = np.concatenate([XR, XI])
            return X
        else:
            raise Exception('ifft Not Implemented Yet -SW')

    def PQRFFTMSq(self, X, N):
        # Calculate the magnitude squared frequency response from the
        # DFT values corresponding to a real signal (assumes N is even)

        X2 = np.zeros(N//2+1)

        X2[0] = X[0]**2
        for k in range(N//2-1):
            X2[k+1] = X[k+1]**2 + X[N//2+k+1]**2

        X2[N//2] = X[N//2]**2
        return X2

    def PQ_GL(self, NF=2048, Amax=1, fcN=1019.5/48000., Lp=92.):
        #Scaled Hann window, including loudness scaling
        # Calculate the gain for the Hann Window
        #  - level Lp (SPL) corresponds to a sine with normalized frequency
        #    fcN and a peak value of Amax
        
        W = NF-1
        gp = self.PQ_gp(fcN, NF, W)
        GL = 10**(Lp/20.) / (gp *Amax/4 *W)
        return GL
        
    def PQ_gp(self, fcN, NF, W):
        # Calculate the peak factor. The signal is a sinusoid windowed with
        # a Hann window. The sinusoid frequency falls between DFT bins. The
        # peak of the frequency response (on a continuous frequency scale) falls
        # between DFT bins. The largest DFT bin value is the peak factor times
        # the peak of the continuous response.
        # fcN - Normalized sinusoid frequency (0-1)
        # NF  - Frame (DFT) length samples
        # NW  - Window length samples

        #Distance to the nearest DFT bin
        df = 1./NF
        k = np.floor(fcN/df)
        
        dfN = np.amin([(k+1)*df - fcN, fcN -k*df])
        
        dfW = dfN*W
        gp = np.sin(np.pi*dfW) / (np.pi*dfW*(1-dfW**2))
        return gp

    def PQWOME(self, f):
        # Generate the weighting for the outer & middle ear filtering
        # Note: The output is a magnitude-squared vector
        
        N = len(f)
        W2 = np.zeros(N)
        
        for k in range(N-1):
            fkHz = float(f[k+1])/1000
            AdB = -2.184 * fkHz**(-0.8) + 6.5 * np.exp(-0.6 * (fkHz - 3.3)**2) - 0.001 * fkHz**(3.6)
            W2[k+1] = 10**(AdB / 10)
        return W2

    def PQCB(self):
        #Critical band parameters for the FFT model, for Basic Version:
        dz = 1./4
        
        #I don't see why we can't hardcode this:
        Nc = 109

        fl = np.array([80.000, 103.445, 127.023, 150.762, 174.694, \
            198.849, 223.257, 247.950, 272.959, 298.317, \
            324.055, 350.207, 376.805, 403.884, 431.478, \
            459.622, 488.353, 517.707, 547.721, 578.434, \
            609.885, 642.114, 675.161, 709.071, 743.884, \
            779.647, 816.404, 854.203, 893.091, 933.119, \
            974.336, 1016.797, 1060.555, 1105.666, 1152.187, \
            1200.178, 1249.700, 1300.816, 1353.592, 1408.094, \
            1464.392, 1522.559, 1582.668, 1644.795, 1709.021, \
            1775.427, 1844.098, 1915.121, 1988.587, 2064.590, \
            2143.227, 2224.597, 2308.806, 2395.959, 2486.169, \
            2579.551, 2676.223, 2776.309, 2879.937, 2987.238, \
            3098.350, 3213.415, 3332.579, 3455.993, 3583.817, \
            3716.212, 3853.817, 3995.399, 4142.547, 4294.979, \
            4452.890, 4616.482, 4785.962, 4961.548, 5143.463, \
            5331.939, 5527.217, 5729.545, 5939.183, 6156.396, \
            6381.463, 6614.671, 6856.316, 7106.708, 7366.166, \
            7635.020, 7913.614, 8202.302, 8501.454, 8811.450, \
            9132.688, 9465.574, 9810.536, 10168.013, 10538.460, \
            10922.351, 11320.175, 11732.438, 12159.670, 12602.412, \
            13061.229, 13536.710, 14029.458, 14540.103, 15069.295, \
            15617.710, 16186.049, 16775.035, 17385.420])
        fc = np.array([91.708, 115.216, 138.870, 162.702, 186.742, \
            211.019, 235.566, 260.413, 285.593, 311.136, \
            337.077, 363.448, 390.282, 417.614, 445.479, \
            473.912, 502.950, 532.629, 562.988, 594.065, \
            625.899, 658.533, 692.006, 726.362, 761.644, \
            797.898, 835.170, 873.508, 912.959, 953.576, \
            995.408, 1038.511, 1082.938, 1128.746, 1175.995, \
            1224.744, 1275.055, 1326.992, 1380.623, 1436.014, \
            1493.237, 1552.366, 1613.474, 1676.641, 1741.946, \
            1809.474, 1879.310, 1951.543, 2026.266, 2103.573, \
            2183.564, 2266.340, 2352.008, 2440.675, 2532.456, \
            2627.468, 2725.832, 2827.672, 2933.120, 3042.309, \
            3155.379, 3272.475, 3393.745, 3519.344, 3649.432, \
            3784.176, 3923.748, 4068.324, 4218.090, 4373.237, \
            4533.963, 4700.473, 4872.978, 5051.700, 5236.866, \
            5428.712, 5627.484, 5833.434, 6046.825, 6267.931, \
            6497.031, 6734.420, 6980.399, 7235.284, 7499.397, \
            7773.077, 8056.673, 8350.547, 8655.072, 8970.639, \
            9297.648, 9636.520, 9987.683, 10351.586, 10728.695, \
            11119.490, 11524.470, 11944.149, 12379.066, 12829.775, \
            13294.850, 13780.887, 14282.503, 14802.338, 15341.057, \
            15899.345, 16477.914, 17077.504, 17690.045])
        fu = np.array([103.445, 127.023, 150.762, 174.694, 198.849, \
            223.257, 247.950, 272.959, 298.317, 324.055, \
            350.207, 376.805, 403.884, 431.478, 459.622, \
            488.353, 517.707, 547.721, 578.434, 609.885, \
            642.114, 675.161, 709.071, 743.884, 779.647, \
            816.404, 854.203, 893.091, 933.113, 974.336, \
            1016.797, 1060.555, 1105.666, 1152.187, 1200.178, \
            1249.700, 1300.816, 1353.592, 1408.094, 1464.392, \
            1522.559, 1582.668, 1644.795, 1709.021, 1775.427, \
            1844.098, 1915.121, 1988.587, 2064.590, 2143.227, \
            2224.597, 2308.806, 2395.959, 2486.169, 2579.551, \
            2676.223, 2776.309, 2879.937, 2987.238, 3098.350, \
            3213.415, 3332.579, 3455.993, 3583.817, 3716.212, \
            3853.348, 3995.399, 4142.547, 4294.979, 4452.890, \
            4643.482, 4785.962, 4961.548, 5143.463, 5331.939, \
            5527.217, 5729.545, 5939.183, 6156.396, 6381.463, \
            6614.671, 6856.316, 7106.708, 7366.166, 7635.020, \
            7913.614, 8202.302, 8501.454, 8811.450, 9132.688, \
            9465.574, 9810.536, 10168.013, 10538.460, 10922.351, \
            11320.175, 11732.438, 12159.670, 12602.412, 13061.229, \
            13536.710, 14029.458, 14540.103, 15069.295, 15617.710, \
            16186.049, 16775.035, 17385.420, 18000.000])
        
        return Nc, fc, fl, fu, dz

    def PQmodPatt(self):
        Nadv = self.NF//2
        Fss = float(self.Fs)/Nadv
        tau_100 = 0.050
        tau_min = 0.008
        alpha, beta = self.PQtConst(tau_100, tau_min, self.fc, Fss)
        if self.check_PQmodPatt == False:
            self.DE = np.zeros((2, self.Nc))
            self.Ese = np.zeros((2, self.Nc))
            self.Eavg = np.zeros((2, self.Nc))
            self.check_PQmodPatt = True
        
        e = 0.3
        Ee = self.Es ** e
        alpha, beta = alpha[None], beta[None]
        self.DE = alpha * self.DE + beta * Fss * np.abs(Ee - self.Ese)
        self.Eavg = alpha * self.Eavg + beta * Ee
        self.Ese = Ee
        M = self.DE / (1 + self.Eavg / e)
        ERavg = self.Eavg[0]
        return M, ERavg

    def PQloud(self, Ehs, mod='FFT'):
        if mod != 'FFT':
            raise ValueError(f'Only FFT mod support, you choose {mod}')

        c = 1.07664
        e = 0.23
        E0 = 1e4
        self.Et = self.PQ_enThresh(self.fc)
        s = self.PQ_exIndex(self.fc)
        Ets = c * (self.Et / (s * E0)) ** e

        
        sN = np.sum(np.maximum(Ets * ((1 - s + s * Ehs / self.Et) ** e - 1), 0))
        Ntot = (24 / self.Nc) * sN
        return Ntot

    @staticmethod
    def PQ_enThresh(fc):
        return 10 ** ((3.64 * (fc / 1000) ** -0.8) / 10)

    @staticmethod
    def PQ_exIndex(fc):
        return 10**((-2 - 2.05 * np.arctan(fc / 4000) - 0.75 * np.arctan((fc / 1600) ** 2)) / 10)

    def PQmovModDiffB(self, M, ERavg):
        e = 0.3
        Ete = self.EIN ** e
        negWt2B = 0.1
        offset1B = 1.0
        offset2B = 0.01
        levWt = 100
        
        cond = M[0] > M[1]
        num1B = np.where(cond, M[0] - M[1], M[1] - M[0])
        num2B = np.where(cond, negWt2B * num1B, num1B)
        MD1B = num1B / (offset1B + M[0])
        MD2B = num2B / (offset2B + M[0])
        s1B = np.sum(MD1B)
        s2B = np.sum(MD2B)
        Wt = np.sum(ERavg / (ERavg + levWt * Ete))

        return (100 / self.Nc) * s1B, (100 / self.Nc) * s2B, Wt

    def PQmovPD(self, EhsR, EhsT):
        c = [-0.198719, 0.0550197, -0.00102438, 5.05622e-6, 9.01033e-11]
        d1 = 5.95072
        d2 = 6.39468
        g = 1.71332
        bP = 4
        bM = 6

        EdBR = 10 * np.log10(EhsR)
        EdBT = 10 * np.log10(EhsT)
        edB = EdBR - EdBT

        cond = edB > 0
        L = np.where(cond, 0.3 * EdBR + 0.7 * EdBT, EdBT)
        b = np.where(cond, bP, bM)

        cond = L > 0
        s = np.where(cond, d1 * (d2 / L) ** g + c[0] + L * (c[1] + L * (c[2] + L * (c[3] + L * c[4]))), 1e30)

        PD_p = 1 - 0.5 ** ((edB / s) ** b)
        PD_q = np.abs(edB.astype(np.int)) / s
        return PD_p, PD_q


class PEAQ(object):
    def __init__(self, Amax = 1, Fs = 48000, NF = 2048):
        # Amax = maximum signal amplitude
        # Fs = sampling frequency
        # NF = Length of analysis window

        self.NF = NF
        self.Fs = Fs
        self.Amax = Amax

        #Step forward in half window lengths:
        self.Nadv = self.NF // 2

        #Number of critical bands:
        self.Nc = 109
        self.P = np.zeros((2, self.Nc))
        self.Rn = np.zeros((self.Nc))
        self.Rd = np.zeros((self.Nc))
        self.PC = np.zeros((2, self.Nc))

    def process(self, referenceSignal, testSignal):
        #Preform basic procssing (Section 2 in Kabal.)
        # sigR = reference signal	
        # sigT = test signal

        sigR = referenceSignal
        sigT = testSignal

        #Number of frames:
        self.Np = (np.floor(len(sigR)/self.Nadv)).astype(np.int32)
        
        #Scale audio:
        if np.amax(abs(sigR)) != self.Amax:
            # sigRS = self.Amax*sigR/float(np.amax(abs(sigR)))
            # sigTS = self.Amax*sigT/float(np.amax(abs(sigT)))
            sigRS = sigR
            sigTS = sigT
            print ('Signals scaled, max reference value = ' + str(np.amax(abs(sigRS))) + ',')
            print ('and max test value = ' + str(np.amax(abs(sigTS))) +'.')

        #Instantiate Object to process single frames of data:
        self.PQE = PQEval(Amax = self.Amax, Fs = self.Fs, NF = self.NF)

        print('Processing Audio...')
        
        #Create empty matrices:
        X2 = np.zeros((2,self.NF//2+1))

        self.X2MatR = np.zeros((self.Np, self.NF//2+1))
        self.X2MatT = np.zeros((self.Np, self.NF//2+1))

        self.EbNMat = np.zeros((self.Np, self.Nc))
        self.EsMatR = np.zeros((self.Np, self.Nc))
        self.EsMatT = np.zeros((self.Np, self.Nc))

        self.EhsR = np.zeros((self.Np, self.Nc))
        self.EhsT = np.zeros((self.Np, self.Nc))

        previousFrameR = np.zeros(self.Nc)
        previousFrameT = np.zeros(self.Nc)

        #Maybe take this out later, but useful in debugging:
        self.xMatR = np.zeros((self.Np, self.NF))
        self.xMatT = np.zeros((self.Np, self.NF))

        self.loud_NRef = np.zeros((self.Np,))
        self.loud_NTest = np.zeros((self.Np,))

        self.BWRef = np.zeros((self.Np,))
        self.BWTest = np.zeros((self.Np,))

        self.PD_p = np.zeros((self.Np))
        self.PD_q = np.zeros((self.Np))
        self.MDiff_Mt1B = np.zeros((self.Np))
        self.MDiff_Mt2B = np.zeros((self.Np))
        self.MDiff_Wt = np.zeros((self.Np))
        self.NLoud_NL = np.zeros((self.Np))
        
        self.EHS = np.zeros((self.Np,))

        startS = 0

        startTime = time.time()

        for i in np.arange(self.Np):
            xR = sigRS[startS:self.NF+startS]
            xT = sigTS[startS:self.NF+startS]
            if xR.shape[-1] < self.NF:
                xR = np.pad(xR, (0, self.NF - xR.shape[-1]))
            if xT.shape[-1] < self.NF:
                xT = np.pad(xT, (0, self.NF - xT.shape[-1]))
            startS = startS+self.Nadv

            #Store unmodified windows of audio:
            self.xMatR[i, :] = xR
            self.xMatT[i, :] = xT
            
            #Process Frame: 
            X2[0,:] = self.PQE.PQDFTFrame(xR)
            X2[1,:] = self.PQE.PQDFTFrame(xT)
            self.X2MatR[i,:] = X2[0,:]
            self.X2MatT[i,:] = X2[1,:]
            
            # Critical band grouping and frequency spreading
            self.EbN, self.Es = self.PQE.PQ_excitCB(X2)
            
            self.EbNMat[i,:] = self.EbN
            self.EsMatR[i,:] = self.Es[0,:]
            self.EsMatT[i,:] = self.Es[1,:]
            
            #Time domain spreading
            self.EhsR[i,:], previousFrameR = self.PQE.PQ_timeSpread(self.EsMatR[i,:], previousFrameR)
            self.EhsT[i,:], previousFrameT = self.PQE.PQ_timeSpread(self.EsMatT[i,:], previousFrameT)

            EP = self.PQadapt(self.EhsR[i], self.EhsT[i], 'FFT')
            M, ERavg = self.PQE.PQmodPatt()
            self.loud_NRef[i] = self.PQE.PQloud(self.EhsR[i,:])
            self.loud_NTest[i] = self.PQE.PQloud(self.EhsT[i,:])

            self.MDiff_Mt1B[i], self.MDiff_Mt2B[i], self.MDiff_Wt[i] = self.PQE.PQmovModDiffB(M, ERavg)

            self.NLoud_NL[i] = self.PQmovNLoudB(M, EP)

            self.BWRef[i], self.BWTest[i] = self.computeBW(self.X2MatR[i], self.X2MatT[i])

            PD_p, PD_q = self.PQE.PQmovPD(self.EhsR[i,:], self.EhsT[i,:])
            self.PD_p[i], self.PD_q[i] = self.PQ_ChanPD(PD_p, PD_q)

            self.EHS[i] = self.PQmovEHS(xR, xT, X2)
        self.NMRavg, self.NMRmax = self.computeNMR(self.EbNMat, self.EhsR)

    def PQ_ChanPD(self, p, q):
        Pr = 1
        Qc = 0
        for m in range(self.Nc):
            Pr *= 1 - p[m]
            Qc += q[m]
        Pc = 1 - Pr
        return Pc, Qc

    def get(self):
        return {'Ntot': {'NRef': self.loud_NRef, 'NTest': self.loud_NTest},
                'ModDiff': {'Mt1B': self.MDiff_Mt1B, 'Mt2B': self.MDiff_Mt2B, 'Wt': self.MDiff_Wt},
                'NL': self.NLoud_NL,
                'BW': {'BWRef': self.BWRef, 'BWTest': self.BWTest},
                'NMR': {'NMRavg': self.NMRavg, 'NMRmax': self.NMRmax},
                'PD': {'p': self.PD_p, 'q': self.PD_q},
                'EHS': self.EHS}

    def PQadapt(self, EhsR, EhsT, Mod='FFT'):
        if Mod != 'FFT':
            raise ValueError(f'Mod only supports FFT, but {Mod}')
        
        Fs = 48000
        Fss = Fs / self.Nadv
        t100 = 0.050
        tmin = 0.008
        a, b = self.PQE.PQtConst(t100, tmin, self.PQE.fc, Fss)
        M1, M2 = 3, 4

        EP = np.zeros((2, self.Nc))
        R = np.zeros((2, self.Nc))

        self.P = np.expand_dims(a,-2) * self.P + np.expand_dims(b,-2) * np.stack([EhsR, EhsT])
        sn = np.sum(np.sqrt(self.P[...,0,:] * self.P[...,1,:]), -1)
        sd = np.sum(self.P[...,1,:], -1)

        CL = (sn / sd) ** 2
        cond = CL > 1
        EP[0] = np.where(cond, EhsR / CL, EhsR)
        EP[1] = np.where(cond, EhsT, EhsT * CL)

        self.Rn = a * self.Rn + EP[1] * EP[0]
        self.Rd = a * self.Rd + EP[0] ** 2

        cond = self.Rn >= self.Rd
        R[0] = np.where(cond, 1, self.Rn / self.Rd)
        R[1] = np.where(cond, self.Rd / self.Rn, 1)
        
        for m in range(self.Nc):
            iL = max(m - M1, 0)
            iU = min(m + M2, self.Nc-1)
            s1 = np.sum(R[0,iL:iU+1], -1)
            s2 = np.sum(R[1,iL:iU+1], -1)

            self.PC[0,m] = a[m] * self.PC[0,m] + b[m] * s1 / (iU-iL+1)
            self.PC[1,m] = a[m] * self.PC[1,m] + b[m] * s2 / (iU-iL+1)

            EP[0,m] *= self.PC[0,m]
            EP[1,m] *= self.PC[1,m]
        return EP

    def avg_get(self):
        self.avgBWRef, self.avgBWTest = self.PQ_avgBW(self.BWRef, self.BWTest)
        self.totalNMRB, self.relDistFramesB = self.PQ_avgNMRB(self.NMRavg, self.NMRmax)

        tdel = 0.5
        Fss = self.Fs / self.Nadv
        N500ms = np.ceil(tdel * Fss)
        Nwup = 0
        Ndel = np.maximum(np.zeros_like(N500ms), N500ms - Nwup)
        tex = 0.05
        
        self.WinModDiff1B, self.AvgModDiff1B, self.AvgModDiff2B = self.PQ_avgModDiffB(Ndel, self.MDiff_Mt1B, self.MDiff_Mt2B, self.MDiff_Wt)
        self.ADBB, self.MFPDB = self.PQ_avgPD(self.PD_p, self.PD_q)

        N50ms = np.ceil(tex * Fss)
        Nloud = self.PQloudTest(self.loud_NRef, self.loud_NTest)
        Ndel = max(Nloud + N50ms, Ndel)
        self.RmsNoiseLoudB = self.PQ_avgNLoudB(Ndel, self.NLoud_NL)
        self.EHSB = self.PQ_avgEHS(self.EHS)
        self.ODG = self.PQnNetB([self.avgBWRef, self.avgBWTest, self.totalNMRB, self.WinModDiff1B, self.ADBB, self.EHSB, self.AvgModDiff1B, self.AvgModDiff2B, self.RmsNoiseLoudB, self.MFPDB, self.relDistFramesB])
        return {'BW': {'BWRef': self.avgBWRef, 'BWTest': self.avgBWTest},
                'NMR': {'totalNMRB': self.totalNMRB, 'relDistFramesB': self.relDistFramesB},
                'WinModDiff1B': self.WinModDiff1B,
                'AvgModDiff1B': self.AvgModDiff1B,
                'AvgModDiff2B': self.AvgModDiff2B,
                'ODG': self.ODG
                }

    def PQnNetB(self, MOV):
        output = self.NNetPar('Basic')
        amin, amax, wx, wxb, wy, wyb, bmin, bmax = list(map(np.array, output))
        MOV = np.array(MOV)
        I, J = wx.shape

        MOVx = (MOV - amin) / (amax - amin)
        DI = wyb
        for j in range(J):
            arg = wxb[j]
            for i in range(I):
                arg += wx[i,j] * MOVx[i]
            DI += wy[j] * self.sigmoid(arg)
        ODG = bmin + (bmax - bmin) * self.sigmoid(DI)
        return ODG
        
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def NNetPar(Version):
        if Version == 'Basic':
            amin = [393.916656, 361.965332, -24.045116, 1.110661, -0.206623, 0.074318, 1.113683, 0.950345, 0.029985, 0.000101, 0]
            amax = [921, 881.131226, 16.212030, 107.137772, 2.886017,
                13.933351, 63.257874, 1145.018555, 14.819740, 1, 
                1]
            wx = [[-0.502657, 0.436333, 1.219602],
                [4.307481, 3.246017, 1.123743],
                [4.984241, -2.211189, -0.192096],
                [0.051056, -1.762424, 4.331315],
                [2.321580, 1.789971, -0.754560],
                [-5.303901, -3.452257, -10.814982],
                [2.730991, -6.111805, 1.519223],
                [0.624950, -1.331523, -5.955151],
                [3.102889, 0.871260, -5.922878],
                [-1.051468, -0.939882, -0.142913],
                [-1.804679, -0.503610, -0.620456]]
            wxb = [-2.518254, 0.654841, -2.207228]
            wy = [-3.817048, 4.107138, 4.629582]
            wyb = -0.307594
            bmin = -3.98
            bmax = 0.22
        else:
            amin = [13.298751, 0.041073, -25.018791, 0.061560, 0.024523]
            amax = [2166.5, 13.24326, 13.46708, 10.226771, 14.224874]
            wx = [[21.211773, -39.913052, -1.382553, -14.545348, -0.320899],
                [-8.981803, 19.956049, 0.935389, -1.686586, -3.238586],
                [1.633830, -2.877505, -7.442935, 5.606502, -1.783120],
                [6.103821, 19.587435, -0.240284, 1.088213, -0.511314],
                [11.556344, 3.892028, 9.720441, -3.287205, -11.031250]]
            wxb = [1.330890, 2.686103, 2.096598, -1.327851, 3.087055]
            wy = [-4.696996, -3.289959, 7.004782, 6.651897, 4.009144]
            wyb = -1.360308
            bmin = -3.98
            bmax = 0.22
        return amin, amax, wx, wxb, wy, wyb, bmin, bmax

    def PQ_avgEHS(self, EHS):
        s = np.sum(self.PQ_LinPosAvg(EHS), -1)
        return 1000 * s

    @staticmethod
    def PQ_LinPosAvg(x):
        return np.mean(x[x >= 0])

    def PQloudTest(self, loud_NRef, loud_NTest):
        Thr = 0.1
        Ndel = self.Np
        Ndel = min(Ndel, self.PQ_Lthresh(Thr, loud_NRef, loud_NTest))
        return Ndel

    def PQ_Lthresh(self, Thr, loud_NRef, loud_NTest):
        for i in range(self.Np):
            if loud_NRef[i] > Thr and loud_NTest[i] > Thr:
                return i
        return self.Np

    def PQ_avgNLoudB(self, Ndel, NLoud):
        x = NLoud[int(Ndel):self.Np]
        if len(x) == 0:
            return 0
        return (np.sum(x ** 2, -1) / len(x)) ** 0.5

    def PQ_avgPD(self, PD_p, PD_q):
        c0 = 0.9
        c1 = 1
        N = PD_p.shape[-1]
        nd = 0
        Qsum = 0
        Pcmax = 0
        Phc = 0
        for i in range(N):
            Phc = c0 * Phc + (1 - c0) * PD_p[...,i]
            Pcmax = max(Pcmax * c1, Phc)
            if PD_p[i] > 0.5:
                nd += 1
                Qsum += PD_q[i]

        if nd == 0:
            ADBB = 0
        elif Qsum > 0:
            ADBB = np.log10(Qsum / nd)
        else:
            ADBB = -0.5
        
        MFPDB = Pcmax
        return ADBB, MFPDB

    def PQmovNLoudB(self, M, EP):
        alpha = 1.5
        TF0 = 0.15
        S0 = 0.5
        NLmin = 0
        e = 0.23
        s = 0

        sref = TF0 * M[0] + S0
        test = TF0 * M[1] + S0
        beta = np.exp(-alpha * (EP[1] - EP[0]) / EP[0])
        tmp = test * EP[1] - sref * EP[0]
        a = np.maximum(tmp, np.zeros_like(tmp))
        b = self.PQE.EIN + sref * EP[0] * beta
        s = np.sum((self.PQE.EIN / test) ** e * ((1 + a / b) ** e - 1))
        NL = (24 / self.Nc) * s
        if NL < NLmin:
            return 0
        return NL

    def computeBW(self, X2MatR, X2MatT):
        fx = 21586
        kx = int(round(self.NF * float(fx)/self.Fs)) # 921
        fl = 8109
        kl = int(round(self.NF * float(fl)/self.Fs)) # 346
        FRdB = 10 # Ref. signal to exceed threshold level by 10dB
        FR = 10**(FRdB/10.) #added dot to make floating point - SW
        FTdB = 5 # Test signal to exceed threshold level by 5dB
        FT = 10**(FTdB/10.) #added dot to make floating point - SW
        
        Xth = np.amax(X2MatT[...,kx:-1], -1)
        XthR = FR * Xth
        cond = X2MatR[...,kl+1:kx] >= XthR[...,None]
        BWRef = (np.arange(kl + 1, cond.shape[-1] + kl + 1)[None] * cond).max(-1) + 1

        XthT = FT * Xth
        cond = X2MatT[...,:int(BWRef-1)] >= XthT[...,None]
        BWTest = (np.arange(cond.shape[-1])[None] * cond).max(-1) + 1
        return BWRef, BWTest

    def computeNMR(self, EbNMat, EhsR):
        #Kabal Section
        #Compute NRM for whole time series.

        NMRavg = np.zeros(self.Np)
        NMRmax = np.zeros(self.Np)

        for i in range(int(self.Np)):
            NMR = self.PQmovNMRB(EbNMat[i,:], EhsR[i,:])
            NMRavg[i] = NMR['NMRavg']
            NMRmax[i] = NMR['NMRmax']

        return NMRavg, NMRmax

    def PQmovNMRB(self, EbN, Ehs):
        NMR = dict()
        
        Nc, fc, fl, fu, dz = self.PQE.PQCB()
        gm = self.PQ_MaskOffset(dz, Nc)
        
        NMRmax = 0
        NMRm = 0
        s = 0

        R_NM = np.zeros(Nc)
        
        for k in range(Nc):
            NMRm = EbN[k] / (gm[k] * Ehs[k])
            R_NM[k] = NMRm # Remove later!
            s = s + NMRm
            
            if (NMRm > NMRmax):
                NMRmax = NMRm
                
        NMR['NMRmax'] = NMRmax
        NMR['NMRavg'] = float(s)/Nc
        
        return NMR

    def PQ_MaskOffset(self, dz, Nc):
        gm = np.zeros(Nc)
        for k in range(Nc):
            if (k <= 12./dz):
                mdB = 3
            else:
                mdB = 0.25*k*dz  
            gm[k] = 10**(-1*float(mdB)/10) 
        return gm

    def PQmovEHS(self, xR, xT, X2):
        NF = 2048
        Nadv = NF // 2
        Fs = 48000
        Fmax = 9000
        NL = 2**(self.PQ_log2(NF * Fmax / Fs))
        M = NL
        Hw = (1 / M) * (8 / 3) ** 0.5 * self.PQE.PQHannWin(M)

        EnThr = 8000
        kmax = NL + M - 1

        xR, xT = np.copy(xR).astype(np.float64), np.copy(xT).astype(np.float64)

        EnRef  = np.matmul(xR[Nadv:NF+1], xR[Nadv:NF+1].T)
        EnTest = np.matmul(xT[Nadv:NF+1], xT[Nadv:NF+1].T)

        if EnRef < EnThr and EnTest < EnThr:
            return -1

        D = np.log(X2[1] / X2[0])
        C = self.PQ_Corr(D, NL, M)

        Cn = self.PQ_NCorr(C, D, NL, M)
        Cnm = (1 / NL) * np.sum(Cn[:NL.astype(np.int)+1])

        Cw = Hw * (Cn - Cnm)

        cp = self.PQE.PQRFFT(Cw, NL.astype(np.int), 1)
        c2 = self.PQE.PQRFFTMSq(cp, NL.astype(np.int))

        EHS = self.PQ_FindPeak(c2, (NL/2+1).astype(np.int))
        return EHS

    def PQ_Corr(self, D, NL, M): # DFT-based operation in original matlab code
        M = M.astype(np.int)
        NL = NL.astype(np.int)

        C = np.zeros(NL)
        for i in range(NL):
            s = 0
            for j in range(M):
                s += D[...,j] * D[...,i+j]
            C[i] = s
        return C

    @staticmethod
    def PQ_log2(x):
        res = np.zeros_like(x)
        m = 1
        while m < x:
            res = res + 1
            m *= 2
        return res - 1

    def PQ_NCorr(self, C, D, NL, M):
        NL = NL.astype(np.int)
        M = M.astype(np.int)
        Cn = np.zeros((NL,))

        s0 = C[0]
        sj = s0
        Cn[0] = 1
        for i in range(1, NL):
            sj += (D[i+M-1] ** 2 - D[i-1] ** 2)
            d = s0 * sj
            if d <= 0:
                Cn[i] = 1
            else:
                Cn[i] = C[i] / d ** 0.5
        return Cn

    @staticmethod
    def PQ_FindPeak(c2, N):
        cprev = c2[0]
        cmax = 0
        for n in range(1, N):
            if c2[n] > cprev and c2[n] > cmax:
                cmax = c2[n]
        return cmax



    ## --------------- Averaging -------------------- ##
    ## Time averaging functions for MOVs
    ## Same naming convention as Kabal
    ##

    def PQ_avgBW(self, BWRef, BWTest):
        # I think this is just an average of all the 
        # positive values, as far as I can tell...
        # Our implementation is simpler too, becuase we aren't worried about stereo
        # Ok, so these values don't exactly match Octave, but they are pretty close (+)
        BandwidthRefB = np.mean(BWRef[BWRef >=0])
        BandwidthTestB = np.mean(BWTest[BWTest >=0])

        return BandwidthRefB, BandwidthTestB

    def PQ_avgNMRB(self, NMRavg, NMRmax):
        #Average NMR values, we also get another MOV here for free - RelDistFramesB
        #This has been validated against Octave, appears to match very well.

        totalNMRB = 10*np.log10(np.mean(NMRavg))

        #Threshold:
        Tr = 10**(1.5/10)
        relDistFramesB = np.mean(NMRmax>Tr)

        return totalNMRB, relDistFramesB

    def PQ_avgModDiffB(self, Ndel, Mt1B, Mt2B, Wt):
        NF = 2048
        Nadv = NF / 2
        Fs = 48000
        Ndel = int(Ndel)

        Fss = Fs / Nadv
        tavg = 0.1

        import pdb; pdb.set_trace()
        L = np.floor(tavg * Fss)
        WinModDiff1B = self.PQ_WinAvg(int(L), Mt1B[Ndel:])

        AvgModDiff1B = self.PQ_WtAvg(Mt1B[Ndel:], Wt[Ndel:])
        AvgModDiff2B = self.PQ_WtAvg(Mt2B[Ndel:], Wt[Ndel:])

        return WinModDiff1B, AvgModDiff1B, AvgModDiff2B

    @staticmethod
    def PQ_WinAvg(L, x):
        N = len(x)

        s = 0
        for i in range(L-1, N):
            t = 0
            for m in range(L):
                t = t + np.sqrt(x[i-m])
            s = s + (t / L) ** 4
        if (N >= L):
            s = np.sqrt(s / (N - L + 1))
        return s

    @staticmethod
    def PQ_WtAvg(x, W):
        N = len(x)

        s = 0
        sW = 0
        for i in range(N):
            s = s + W[i] * x[i]
            sW = sW + W[i]

        if (N > 0):
            s = s / sW
        return s

