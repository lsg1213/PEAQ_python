import torch
import torchaudio
import numpy_PEAQ
import torch_PEAQ


def print_as_frame(metrics, i):
    print('Ntot:\t', metrics['Ntot']['NRef'][i], metrics['Ntot']['NTest'][i])
    print('ModDiff:\t', metrics['ModDiff']['Mt1B'][i], metrics['ModDiff']['Mt2B'][i], metrics['ModDiff']['Wt'][i])
    print('NL:\t', metrics['NL'][i])
    print('BW:\t', metrics['BW']['BWRef'][i], metrics['BW']['BWTest'][i])
    print('NMR:\t', metrics['NMR']['NMRavg'][i], metrics['NMR']['NMRmax'][i])
    print('PD:\t', metrics['PD']['p'][i], metrics['PD']['q'][i])
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
    
    # numpy version
    # nppeaq = numpy_PEAQ.PEAQ(32768, Fs=rate)
    # nppeaq.process(ref, test)
    # metrics_as_frame = nppeaq.get()
    # npmetrics = nppeaq.avg_get()
    # print('---------- numpy PEAQ ----------')
    # print(npmetrics)

    torchpeaq = torch_PEAQ.PEAQ(32768, Fs=rate)
    torchpeaq.process(ref, test)
    metrics_as_frame = torchpeaq.get()
    torchmetrics = torchpeaq.avg_get()
    print('---------- torch PEAQ ----------')
    print(torchmetrics)


    # pytorch version

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

