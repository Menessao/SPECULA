import numpy as np
import os
from specula.mmlib.yaml_overrides import write_yaml_overrides
from astropy.io import fits

from specula.mmlib.utils import shift_image, get_frame_pupil_centers

def get_meas_frame(im,idx=None,trim:bool=True):
    imstd = np.std(im,axis=1).reshape([240,240])
    ref_centers = get_frame_pupil_centers(imstd)
    savg_center = np.mean(ref_centers,axis=0)
    hsize = 120
    if idx is None:
        imframe = imstd
    else:
        imframe = im[:,idx].reshape([240,240])
    frimg = shift_image(imframe, shift=hsize-savg_center[1], axis=0)
    frimg = shift_image(frimg, shift=hsize-savg_center[0], axis=1)
    if trim:
        return frimg[60:180,60:180]
    else:
        return frimg

im = fits.getdata('/raid1/mmenessini/calibration/EKARUS/data/IntMat_20260815_194918.fits') #194038 #193320
img = get_meas_frame(im,trim=False)
# img = np.load('/raid1/mmenessini/calibration/EKARUS/data/OnBench_frame.npy')
ref_centers = get_frame_pupil_centers(img)
ref_dcenters = ref_centers - np.mean(ref_centers,axis=0)

def evaluate_metric():
    frame = fits.getdata('/raid1/mmenessini/results/EKARUS/frame.fits')[0]
    diff = img/np.mean(img)-frame/np.mean(frame)
    chi = np.sum(diff**2)
    return chi

def delta_centers(pup_sep:float):    
    frame = fits.getdata('/raid1/mmenessini/results/EKARUS/frame.fits')[0]
    centers = get_frame_pupil_centers(frame)
    dcenters = centers - np.mean(centers,axis=0)
    dtilts = (ref_dcenters - dcenters)/pup_sep
    pupids = np.array([3,2,1,0],dtype=int)
    dtilts[:2,:] *= -1
    return dtilts[pupids,:].T

# Parameters to optimize
diams = np.linspace(39,41,5)
dtlts = np.linspace(-0.1,0.1,21)
# pyr_tlt_ref = [[1.1328852114285715, 1.0876296414285715, 1.1665203742857142, 1.1357142857142857], 
#                [1.0607142857142857, 0.9845511985714286, 1.0239343157142857, 0.95]]
pyr_tlt_ref = [[1.1020833372833667, 1.1616588162548362, 1.0971916659819192, 1.166550487556285],
    [1.0678673571075679, 0.968967716126909, 1.0551018949820765, 0.9817331782524007]]
pyr_tlt_ref = [[1.1520833372833667, 1.1116588162548362, 1.1671916659819192, 1.146550487556285],
    [1.0678673571075679, 0.968967716126909, 1.0351018949820765, 0.9617331782524007]]


result_dir = '/raid1/mmenessini/calibration/EKARUS/scratch_bench_pupopt'
overrides_name = 'pup_bench_overrides'
main_config = 'config/EKARUS/ekarus_pupils_synim.yml'
results = []

if __name__ == '__main__':

    nominal_deviation = 60/2
    gain = 0.5

    best_pyr_tlt = np.array(pyr_tlt_ref.copy())
    for it in range(0):    
        ovdes = ("{"
            f"pyr.pup_diam: {np.max(diams):1.1f}, "
            f"pyr.pyr_tlt_coeff: {(best_pyr_tlt).tolist()}, "
            f"prop.inputs.common_layer_list: ['dmstop'], "
            "}")
        write_yaml_overrides(input_string=ovdes, temp_name=overrides_name)
        os.system(f'specula {main_config} {overrides_name}.yml')
        dtilts = delta_centers(nominal_deviation)
        best_pyr_tlt += dtilts * gain
        print(dtilts)

    vec = np.zeros(len(diams))
    for i,diam in enumerate(diams):
        ovdes = ("{"
                f"pyr.pup_diam: {diam:1.1f}, "
                f"pyr.pyr_tlt_coeff: {best_pyr_tlt.tolist()}, "
                f"prop.inputs.common_layer_list: ['dmstop'], "
                "}")
        write_yaml_overrides(input_string=ovdes, temp_name=overrides_name)
        os.system(f'specula {main_config} {overrides_name}.yml')
        vec[i] = evaluate_metric()
    np.savez(os.path.join(result_dir,'pup_sizes_metric.npz'),metric=vec,pup_diams=diams)
    diam = diams[np.argmin(vec)]
    print(f'Selected {diam:1.1f} pix diameter')

    # best_pyr_tlt = np.array(pyr_tlt_ref.copy())
    # for k in range(4):
    #     vec = np.zeros([len(dtlts),len(dtlts)])
    #     for i,dtltx in enumerate(dtlts):
    #         for j,dtlty in enumerate(dtlts):
    #             pyr_tlts = best_pyr_tlt.copy()
    #             pyr_tlts[0,k] = pyr_tlts[0,k] + dtltx
    #             pyr_tlts[1,k] = pyr_tlts[1,k] + dtlty
    #             ovdes = ("{"
    #                     f"pyr.pup_diam: {diam:1.1f}, "
    #                     f"pyr.pyr_tlt_coeff: {(pyr_tlts).tolist()}, "
    #                     f"prop.inputs.common_layer_list: ['dmstop'], "
    #                     "}")
    #             write_yaml_overrides(input_string=ovdes, temp_name=overrides_name)
    #             os.system(f'specula {main_config} {overrides_name}.yml')
    #             vec[i,j] = evaluate_metric()
    #     np.savez(os.path.join(result_dir,f'pup{k:1.0f}_tilts_metric.npz'),metric=vec,tiltsX=pyr_tlts[0,k]+dtlts,tiltsY=pyr_tlts[1,k]+dtlts)
    #     best_pyr_tlt[0,k] = best_pyr_tlt[0,k] + dtlts[np.unravel_index(vec.argmin(), vec.shape)[0]]
    #     best_pyr_tlt[1,k] = best_pyr_tlt[1,k] + dtlts[np.unravel_index(vec.argmin(), vec.shape)[1]]
    #     print(f'Selected {best_pyr_tlt[0,k]:1.2f},{best_pyr_tlt[1,k]:1.2f} for pupil {k}')

    overrides = ("{"
                f"pyr.pup_diam: {diam:1.1f}, "
                f"pyr.pyr_tlt_coeff: {best_pyr_tlt.tolist()}, "
                f"prop.inputs.common_layer_list: ['dmstop'], "
                "}")
    write_yaml_overrides(input_string=overrides, temp_name=overrides_name)
    os.system(f'specula {main_config} {overrides_name}.yml')
