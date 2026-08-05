import numpy as np
import os
from specula.mmlib.yaml_overrides import write_yaml_overrides
from astropy.io import fits

img = np.load('/raid1/mmenessini/calibration/EKARUS/data/OnBench_frame.npy')

def image_grid(shape, recenter:bool = False):
    ny, nx = shape
    cy, cx = (0,0)
    if recenter:
        cy, cx = ny//2, nx//2
    x = np.arange(nx, dtype=float) - cx
    y = np.arange(ny, dtype=float) - cy
    X,Y = np.meshgrid(x, y)
    return X,Y

def get_photocenter(image,offset:bool=False):
    X,Y = image_grid(image.shape)
    qy = np.sum(Y * image) / np.sum(image)
    qx = np.sum(X * image) / np.sum(image)
    if offset:
        qy += 0.5
        qx += 0.5
    return qx,qy 

def get_frame_pupil_centers(frame,thr=0.3,xhalf=100,yhalf=120):
    Y,X = image_grid(frame.shape)
    ll = (X<=xhalf) * (Y<=yhalf)
    lr = (X>xhalf) * (Y<=yhalf)
    ul = (X<=xhalf) * (Y>yhalf)
    ur = (X>xhalf) * (Y>yhalf)
    centers = np.zeros([4,2])
    mask = (frame*ll > thr*np.nanmax(frame*ll)).astype(bool)
    qx,qy = get_photocenter(mask)
    centers[0,:] = np.array([qx,qy])
    mask = (frame*lr > thr*np.nanmax(frame*lr)).astype(bool)
    qx,qy = get_photocenter(mask)
    centers[1,:] = np.array([qx,qy])
    mask = (frame*ul > thr*np.nanmax(frame*ul)).astype(bool)
    qx,qy = get_photocenter(mask)
    centers[2,:] = np.array([qx,qy])
    mask = (frame*ur > thr*np.nanmax(frame*ur)).astype(bool)
    qx,qy = get_photocenter(mask)
    centers[3,:] = np.array([qx,qy])
    return centers

ref_centers = get_frame_pupil_centers(img)
avg_center = np.mean(ref_centers,axis=0)

xmin = int(np.round(avg_center[0]))-60
xmax = int(np.round(avg_center[0]))+60
ymin = int(np.round(avg_center[1]))-60
ymax = int(np.round(avg_center[1]))+60

img = img[ymin:ymax,xmin:xmax]

def evaluate_metric():
    frame = fits.getdata('/raid1/mmenessini/results/EKARUS/frame.fits')[0]
    cframe = frame[60:180,60:180]
    diff = img/np.mean(img)-cframe/np.mean(cframe)
    chi = np.sum(diff**2)
    return chi

# Parameters to optimize
diams = np.linspace(38,40,5)
dtlts = np.linspace(-0.1,0.1,21)
pyr_tlt_ref = [[1.1328852114285715, 1.0876296414285715, 1.1665203742857142, 1.1357142857142857], [1.0607142857142857, 0.9845511985714286, 1.0239343157142857, 0.95]]

result_dir = '/raid1/mmenessini/calibration/EKARUS/scratch_bench_pupopt'
overrides_name = 'pup_bench_overrides'
main_config = 'config/EKARUS/ekarus_pupils_synim.yml'
results = []

if __name__ == '__main__':

    vec = np.zeros(len(diams))
    for i,diam in enumerate(diams):
        ovdes = ("{"
                f"pyr.pup_diam: {diam:1.1f}, "
                f"pyr.pyr_tlt_coeff: {pyr_tlt_ref}, "
                f"prop.inputs.common_layer_list: ['dmstop'], "
                "}")
        write_yaml_overrides(input_string=ovdes, temp_name=overrides_name)
        os.system(f'specula {main_config} {overrides_name}.yml')
        vec[i] = evaluate_metric()
    np.savez(os.path.join(result_dir,'pup_sizes_metric.npz'),metric=vec,pup_diams=diams)
    diam = diams[np.argmin(vec)]
    print(f'Selected {diam:1.1f} pix diameter')

    ovdes = ("{"
            f"pyr.pup_diam: {diam:1.1f}, "
            f"pyr.pyr_tlt_coeff: {pyr_tlt_ref}, "
            f"prop.inputs.common_layer_list: ['dmstop'], "
            "}")
    write_yaml_overrides(input_string=ovdes, temp_name=overrides_name)

    best_pyr_tlt = np.array(pyr_tlt_ref.copy())
    for k in range(4):
        vec = np.zeros([len(dtlts),len(dtlts)])
        for i,dtltx in enumerate(dtlts):
            for j,dtlty in enumerate(dtlts):
                pyr_tlts = best_pyr_tlt.copy()
                pyr_tlts[0,k] = pyr_tlts[0,k] + dtltx
                pyr_tlts[1,k] = pyr_tlts[1,k] + dtlty
                ovdes = ("{"
                        f"pyr.pup_diam: {diam:1.1f}, "
                        f"pyr.pyr_tlt_coeff: {(pyr_tlts).tolist()}, "
                        f"prop.inputs.common_layer_list: ['dmstop'], "
                        "}")
                write_yaml_overrides(input_string=ovdes, temp_name=overrides_name)
                os.system(f'specula {main_config} {overrides_name}.yml')
                vec[i,j] = evaluate_metric()
        np.savez(os.path.join(result_dir,f'pup{k:1.0f}_tilts_metric.npz'),metric=vec,tiltsX=pyr_tlts[0,k]+dtlts,tiltsY=pyr_tlts[1,k]+dtlts)
        best_pyr_tlt[0,k] = best_pyr_tlt[0,k] + dtlts[np.unravel_index(vec.argmin(), vec.shape)[0]]
        best_pyr_tlt[1,k] = best_pyr_tlt[1,k] + dtlts[np.unravel_index(vec.argmin(), vec.shape)[1]]
        print(f'Selected {best_pyr_tlt[0,k]:1.2f},{best_pyr_tlt[1,k]:1.2f} for pupil {k}')

    overrides = ("{"
                f"pyr.pup_diam: {diam:1.1f}, "
                f"pyr.pyr_tlt_coeff: {best_pyr_tlt.tolist()}, "
                f"prop.inputs.common_layer_list: ['dmstop'], "
                "}")
    write_yaml_overrides(input_string=overrides, temp_name=overrides_name)
    os.system(f'specula {main_config} {overrides_name}.yml')
