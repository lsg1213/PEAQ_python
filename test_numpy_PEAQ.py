import torch
import torchaudio
from numpy_PEAQ import PEAQ


def print_as_frame(i):
    print('Ntot:\t', metrics['Ntot']['NRef'][i], metrics['Ntot']['NTest'][i])
    print('ModDiff:\t', metrics['ModDiff']['Mt1B'][i], metrics['ModDiff']['Mt2B'][i], metrics['ModDiff']['Wt'][i])
    # print('NL:\t', metrics['NL'][i])
    print('BW:\t', metrics['BW']['BWRef'][i], metrics['BW']['BWTest'][i])
    print('NMR:\t', metrics['NMR']['NMRavg'][i], metrics['NMR']['NMRmax'][i])
    # print('PD:\t', metrics['PD']['p'][i], metrics['PD']['q'][i]) # PD.pc, PD.qc ?
    print('EHS:\t', metrics['EHS'][i] * 1000)


def load(name):
    audio, rate = torchaudio.load(name, normalize=False)

    if audio.dtype == torch.float32:
        audio = audio * 32768.
    audio = audio.squeeze().numpy()
    return audio, rate


def main():
    ref, rate = load('test_clean.wav')
    test, rate = load('test_recons.wav')
    peaq = PEAQ(32768, Fs=rate)
    peaq.process(ref, test)
    metrics = peaq.get()
    print(peaq.PQ_avgBW(metrics['BW']['BWRef'], metrics['BW']['BWTest']))
    # MATLAB CODE OUTPUTS
    # Model Output Variables:
    #     BandwidthRefB: 841.045
    #     BandwidthTestB: 304.442
    #         Total NMRB: 12.1637
    #         WinModDiff1B: 65.3313
    #                 ADBB: 3.06878
    #                 EHSB: 7.32808
    #         AvgModDiff1B: 65.2213
    #         AvgModDiff2B: 125.292
    #     RmsNoiseLoudB: 5.72994
    #             MFPDB: 1
    #     RelDistFramesB: 1
    #     Objective Difference Grade: -3.875

if __name__ == '__main__':
    main()

