#%Import modules
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

import spiceypy as sp

import spacepy.irbempy as irb
import spacepy.time as spt

from planetary_coverage import TourConfig, et, utc
from planetary_coverage.misc import Segment
from planetary_coverage import EARTH, ESA_MK
from planetary_coverage.math import angle
from planetary_coverage import SpiceRef
from planetary_coverage import SpicePool
from planetary_coverage.spice import ocentric2ographic
from planetary_coverage.maps import Map
from planetary_coverage.math import hav_dist
from planetary_coverage.math.greatcircle import great_circle_patch
from planetary_coverage.ticks import deg_ticks, km_s_ticks, date_ticks, km_pix_ticks, km_ticks, UnitFormatter

#import cartopy.crs as ccrs
from geopack import geopack
from geopack import t89, t96, t01, t04

from scipy.optimize import least_squares

import chaosmagpy as cp
import pooch

re = 6371.2
rp = 6357
f = (re-rp) / re

def spicekernel(mk = 'ops', kernels_dir = 'JUICE-Data/kernels', download_kernels = True, spacecraft = 'JUICE', target = 'EARTH'):
    #%Load spice kernels for flybys
    tour_juice = TourConfig(
        mk = mk,
        kernels_dir = kernels_dir,
        download_kernels = download_kernels,
        spacecraft = spacecraft,
        target= target
        )
    
    return tour_juice

def flybys(spicekernel = spicekernel()):
    flybys = spicekernel.get_flybys()

    return flybys

def times(flybys_juice = flybys(spicekernel()), flyby=1):
    flyby_start_juice = flybys_juice[flyby].start
    flyby_stop_juice = flybys_juice[flyby].stop
    flybys_utc_juice = flybys_juice[flyby].utc
    lotlan = flybys_juice[flyby].lotlan

    ets = et(flybys_utc_juice) #utc to et conversion
    dates = [datetime for datetime in utc(ets)]

    return dates, ets, flyby_start_juice, flyby_stop_juice, lotlan

# Custom function to parse the datetime strings
def parse_datetime(dt_str):
    return datetime.strptime(dt_str, '%Y-%m-%dT%H:%M:%S.%fZ')

def b_val(ut, r, colat, lon, pos, mat = 'sca'):
    b_vec = np.zeros((n, 3))
    b_mod_vec = np.zeros(n)
    if mat == 'sca':
        pform = pform_magsca
    if mat == 'ibs':
        pform = pform_magibs
    if mat == 'obs':
        pform = pform_magobs
    for i in range(n):
        """
        model = cp.load_CHAOS_matfile(chaos_matfile)
        time = cp.data_utils.mjd2000(2023, 1, 1)  # modified Julian date
        B_radius, B_theta, B_phi = model(time, r[i], np.degrees(colat[i]), np.degrees(lon[i]), source_list=['tdep', 'static'])  # only internal sources
        B_x = (B_radius * np.sin(colat[i]) * np.cos(lon[i]) +
           B_theta * np.cos(colat[i]) * np.cos(lon[i]) -
           B_phi * np.sin(lon[i]))
    
        B_y = (B_radius * np.sin(colat[i]) * np.sin(lon[i]) +
           B_theta * np.cos(colat[i]) * np.sin(lon[i]) +
           B_phi * np.cos(lon[i]))
    
        B_z = (B_radius * np.cos(colat[i]) -
           B_theta * np.sin(colat[i]))

        b_car = np.array([B_x, B_y, B_z])
        
        """


        #xgsm,ygsm,zgsm = (pos[i][0]/re, pos[i][1]/re, pos[i][2]/re)
        #ps = geopack.recalc(ut[i])
        #b0xgsm,b0ygsm,b0zgsm = geopack.dip(xgsm,ygsm,zgsm)    		# calc dipole B in GSM.
        #dbxgsm,dbygsm,dbzgsm = t89.t89(2, ps, xgsm,ygsm,zgsm)       # calc T89 dB in GSM.
        #bxgsm,bygsm,bzgsm = [b0xgsm+dbxgsm,b0ygsm+dbygsm,b0zgsm+dbzgsm]
        #print(bxgsm,bygsm,bzgsm)
        ps = geopack.recalc(ut[i])
        xgsm, ygsm, zgsm = geopack.gsmgse(pos[i][0]/re, pos[i][1]/re, pos[i][2]/re, -1) #gse-to-gsm

        b0xgsm,b0ygsm,b0zgsm = geopack.igrf_gsm(xgsm,ygsm,zgsm)    		# calc dipole B in GSM.
        dbxgsm,dbygsm,dbzgsm = geopack.t01.t01([0.24,-10,0.8,-4.8,6,10], ps, xgsm,ygsm,zgsm)       # calc T89 dB in GSM.
        txgsm,tygsm,tzgsm = [b0xgsm + dbxgsm,b0ygsm + dbygsm,b0zgsm + dbzgsm]
        t_gsex, t_gsey, t_gsez = geopack.gsmgse(txgsm, tygsm, tzgsm, 1) #
        t_gse = np.array([t_gsex, t_gsey, t_gsez])

        b_geo = np.array(geopack.igrf_geo(r = r[i]/re, theta = (colat[i]), phi = lon[i]))
        b_car = np.array(geopack.bspcar(theta = colat[i], phi = lon[i], br = b_geo[0], btheta = b_geo[1], bphi = b_geo[2]))
        # Calculate the Cartesian components
        #b_car = np.array([bxgsm, bygsm, bzgsm])
        b_car_tr = np.dot(pform[i], t_gse)
        b_mod = np.sqrt(b_car_tr[0]**2 + b_car_tr[1]**2 + b_car_tr[2]**2)
        b_vec[i][0] = b_car_tr[0]
        b_vec[i][1] = b_car_tr[1]
        b_vec[i][2] = b_car_tr[2]
        b_mod_vec[i] = b_mod
    
    return (b_vec.T, b_mod_vec)

def sca_angle(b_2):
    return np.degrees(np.arccos(b_2[0][0] / b_2[1]))

def plot_mag_fields(dates_int, indices, b_0, b_1, b_2, angle_b_sca, filename = 'B-Field-Frame.png'):

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))

    axes[0][0].plot(dates_int, b_0[0][0][indices], label='$B_{x}$', color = 'red')
    axes[0][0].plot(dates_int, b_0[0][1][indices], label='$B_{y}$', color = 'lime')
    axes[0][0].plot(dates_int, b_0[0][2][indices], label='$B_{z}$', color = 'blue')
    axes[0][0].plot(dates_int, b_0[1][indices], label='$|B|$', color = 'black')
    axes[0][0].legend()
    axes[0][0].set_ylabel("MAG_OBS magnetic field (nT)")
    axes[0][0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


    axes[1][0].plot(dates_int, b_1[0][0][indices], label='$B_{x}$', color = 'red')
    axes[1][0].plot(dates_int, b_1[0][1][indices], label='$B_{y}$', color = 'lime')
    axes[1][0].plot(dates_int, b_1[0][2][indices], label='$B_{z}$', color = 'blue')
    axes[1][0].plot(dates_int, b_1[1][indices], label='$|B|$', color = 'black')
    axes[1][0].legend()
    axes[1][0].set_ylabel("MAG_IBS magnetic field (nT)")
    axes[1][0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[1][0].set_xlabel('UTC on 20th August 2024')


    axes[0][1].plot(dates_int, b_2[0][0][indices], label='$B_{x}$', color = 'red')
    axes[0][1].plot(dates_int, b_2[0][1][indices], label='$B_{y}$', color = 'lime')
    axes[0][1].plot(dates_int, b_2[0][2][indices], label='$B_{z}$', color = 'blue')
    axes[0][1].plot(dates_int, b_2[1][indices], label='$|B|$', color = 'black')
    axes[0][1].legend()
    axes[0][1].set_ylabel("MAG_SCA magnetic field (nT)")
    axes[0][1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


    axes[1][1].plot(dates_int, angle_b_sca[indices])
    axes[1][1].set_ylabel("SCA optical axis angle")
    axes[1][1].yaxis.set_major_formatter(FuncFormatter(degree_formatter))
    axes[1][1].set_ylim(0,180)
    axes[1][1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[1][1].set_xlabel('UTC on 20th August 2024')

    print(np.max(b_0[1]), np.max(b_1[1]), np.max(b_2[1]))
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

def degree_formatter(x, pos):
    return f'{x}°'

#Functions for alignment algorithm
def rotate_3d(b, axis, angle):
    """
    Rotate 3D coordinates around a specified axis by a given angle.
    
    Parameters:
    - x, y, z: Arrays or lists of x, y, z coordinates.
    - axis: 'x', 'y', or 'z', specifying the rotation axis.
    - angle: Rotation angle in degrees.
    
    Returns:
    - Rotated x, y, z coordinates.
    """
    # Convert angle to radians
    angle = np.radians(angle)
    
    # Define rotation matrices
    if axis == 'x':
        R = 1.1*np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    elif axis == 'y':
        R = 1.2*np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    elif axis == 'z':
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    
    # Stack x, y, z into a matrix
    coords = np.vstack([b[0][0], b[0][1], b[0][2]])

    rotated_coords = R @ coords
    
    magnitudes = np.sqrt(rotated_coords[0]**2 + rotated_coords[1]**2 + rotated_coords[2]**2)
    
    return rotated_coords, magnitudes

def extract_magnetic_fields(b_data):
    """
    Extract the (x, y, z) arrays from b_data and return as (n, 3) matrix.
    
    Parameters:
    - b_data: A tuple where the first element is (x, y, z) arrays and the second element is b_mod.
    
    Returns:
    - (n, 3) matrix of magnetic field vectors.
    """
    b_x, b_y, b_z = b_data[0]
    
    return np.vstack((b_x, b_y, b_z)).T

#Least squares

def rotation_matrix_from_angles(theta_x, theta_y, theta_z):
    """
    Create a rotation matrix from Euler angles (in degrees).
    
    Parameters:
    - theta_x, theta_y, theta_z: Rotation angles around x, y, and z axes in degrees.
    
    Returns:
    - Rotation matrix R.
    """
    theta_x, theta_y, theta_z = np.radians([theta_x, theta_y, theta_z])
    
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])
    
    R_y = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])
    
    R_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])
    
    return R_z @ R_y @ R_x

def residuals(angles, b0, b0_actual):
    """
    Compute the residuals for least squares fitting of rotation matrix.
    
    Parameters:
    - angles: Rotation angles to be optimized (theta_x, theta_y, theta_z in degrees).
    - b0: Theoretical data (n, 3) where each row is a 3D vector.
    - b0_actual: Actual data (n, 3) where each row is a 3D vector.
    
    Returns:
    - Residuals vector.
    """
    R = rotation_matrix_from_angles(*angles)
    b0_rotated = np.dot(b0, R.T)
    return (b0_rotated - b0_actual).ravel()

def jacobian_of_rotation_matrix(theta_x, theta_y, theta_z):
    """
    Compute the Jacobian of the rotation matrix with respect to the angles.
    
    Parameters:
    - theta_x, theta_y, theta_z: Rotation angles around the x, y, z axes in degrees.
    
    Returns:
    - A list of 3x3 Jacobian matrices for the rotation matrix with respect to each angle.
    """
    theta_x = np.radians(theta_x)
    theta_y = np.radians(theta_y)
    theta_z = np.radians(theta_z)
    
    # Derivatives with respect to theta_x
    dRx_dtheta_x = np.array([
        [0, 0, 0],
        [0, -np.sin(theta_x), -np.cos(theta_x)],
        [0, np.cos(theta_x), -np.sin(theta_x)]
    ])
    
    # Derivatives with respect to theta_y
    dRy_dtheta_y = np.array([
        [-np.sin(theta_y), 0, np.cos(theta_y)],
        [0, 0, 0],
        [-np.cos(theta_y), 0, -np.sin(theta_y)]
    ])
    
    # Derivatives with respect to theta_z
    dRz_dtheta_z = np.array([
        [-np.sin(theta_z), -np.cos(theta_z), 0],
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [0, 0, 0]
    ])
    
    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])
    
    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])
    
    # Jacobians with respect to each angle
    dR_dtheta_x = Rz @ Ry @ dRx_dtheta_x
    dR_dtheta_y = Rz @ dRy_dtheta_y @ Rx
    dR_dtheta_z = dRz_dtheta_z @ Ry @ Rx
    
    return [dR_dtheta_x, dR_dtheta_y, dR_dtheta_z]

def compute_rotation_matrix_least_squares(b0, b0_actual):
    """
    Compute the rotation matrix that best aligns b0 with b0_actual using least squares fitting.
    
    Parameters:
    - b0: Theoretical data (n, 3) where each row is a 3D vector.
    - b0_actual: Actual data (n, 3) where each row is a 3D vector.

    Returns:
    - Rotation matrix R.
    """
    # Initial guess for Euler angles (can be zeros or any reasonable guess)
    initial_angles = [0, 0, 0]
    
    # Use least squares to optimize the rotation angles
    result = least_squares(residuals, initial_angles, args=(b0, b0_actual))
    optimized_angles = result.x
    
    residuals_var = np.var(result.fun)

    print(optimized_angles)
    # Calculate the covariance matrix from the Jacobian
    J = result.jac  # Jacobian matrix from least squares
    cov_matrix = np.linalg.inv(J.T @ J) * residuals_var
    
    # Standard errors are the square roots of the diagonal elements of the covariance matrix
    standard_errors = np.sqrt(np.diag(cov_matrix))

    print(standard_errors)
    # Compute the rotation matrix from optimized angles
    R = rotation_matrix_from_angles(*optimized_angles)
     # Compute the Jacobians of the rotation matrix with respect to each angle
    
    jacobians_R = jacobian_of_rotation_matrix(*optimized_angles)
    
    # Compute the error propagation for each element in the rotation matrix
    R_errors = np.zeros_like(R)
    for i in range(3):
        for j in range(3):
            variance_Rij = sum(
                (jacobians_R[k][i, j] ** 2) * cov_matrix[k, k]
                for k in range(3)
            )
            R_errors[i, j] = np.sqrt(variance_Rij)

    print(R_errors)
    
    return R, R_errors, optimized_angles, standard_errors

def apply_rotation_matrix(b, R):
    """
    Apply the rotation matrix R to the data b.

    Parameters:
    - b: Data to be rotated (n, 3) where each row is a 3D vector.
    - R: Rotation matrix (3, 3).

    Returns:
    - Rotated data (n, 3).
    """
    b = np.array(b)
    R = np.array(R)
    
    # Ensure shapes are correct
    if b.shape[1] != 3 or R.shape != (3, 3):
        raise ValueError("Input data and rotation matrix have incorrect shapes.")
    
    return np.dot(b, R.T)

#SVD

def compute_rotation_matrix_svd(b0, b0_actual):
    """
    Compute the rotation matrix that best aligns b0 with b0_actual using Singular Value Decomposition (SVD).
    
    Parameters:
    - b0: Theoretical data (n, 3) where each row is a 3D vector.
    - b0_actual: Actual data (n, 3) where each row is a 3D vector.

    Returns:
    - Rotation matrix R.
    """
    # Ensure b0 and b0_actual have the same shape
    if b0.shape != b0_actual.shape:
        raise ValueError("b0 and b0_actual must have the same shape")

    # Compute the matrix H
    H = np.dot(b0.T, b0_actual)
    
    # Perform SVD
    U, _, Vt = np.linalg.svd(H)
    
    # Compute rotation matrix
    R = np.dot(U, Vt)
    
    # Ensure a proper rotation matrix with a positive determinant
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(U, Vt)

    return R.T

chaos_matfile = pooch.retrieve(
    "http://www.spacecenter.dk/files/magnetic-models/CHAOS-7/CHAOS-7.15.mat",
    known_hash="4a074dd48674eafd076fb95ac05cd1e3e034c2eb6bfe5ee3f566e3764c43bb80"
)

# Processing onboard data
data = np.loadtxt('JUICE-Data/2024-08-20_fob_1S_calibrated.dat', dtype=str)

str_utcs = data[:, 0]            # Time as strings
utcs = np.array([parse_datetime(t) for t in str_utcs])  # Convert to datetime objects if needed
utcs = np.array(data[:, 0], dtype='datetime64')
ets= et(utcs)

###Delete
#utcs = times()[0]
#str_utcs = [utcs[i].astype('datetime64[s]').astype(datetime).strftime('%Y-%m-%dT%H:%M:%SZ') for i in range(len(utcs))]
#ets = times()[1]
###

date = np.array(utcs, dtype='datetime64')
b_x = data[:, 1].astype(float)   # B_x values
b_y = data[:, 2].astype(float)   # B_y values
b_z = data[:, 3].astype(float)   # B_z values
modulus_b = data[:, 4].astype(float)  # Modulus of B values
print(np.max(modulus_b))
b_data = [np.array([b_x, b_y, b_z]), modulus_b] 

# Processing predicted data
METAKR = 'JUICE-Data/kernels/mk/juice_ops.tm'
sp.furnsh( METAKR )

et_conv = [sp.str2et( str_utcs[i] ) for i in range(len(utcs))]

state = [sp.spkezr( 'JUICE', et_conv[i],      'GSE',
                                      'none',   'EARTH'       ) for i in range(len(et_conv))]
# Example of applying the transformation to a state vector in J2000

ts = np.array(utcs)
t0 = np.datetime64(datetime(1970,1,1))
dt = (utcs- t0)
ut = (dt / np.timedelta64(1, 's')) 

n = len(state)
pos = np.array([(state[i][0][0:3]) for i in range(n)])
vel = np.array([state[i][0][3:7] for i in range(n)])
ltime = np.array([state[i][1] for i in range(n)])
r = np.array([sp.recsph(pos[i])[0] for i in range(n)])
colat = np.array([sp.recsph(pos[i])[1] for i in range(n)])
lat = (np.pi/2) - colat
lon = np.array([sp.recsph(pos[i])[2] for i in range(n)])

pform_magsca = [sp.pxform('GSE', 'JUICE_JMAG_MAGSCA_SCI', ets[i]) for i in range(n)]
pform_magobs = [sp.pxform('GSE', 'JUICE_JMAG_MAGOBS_SCI', ets[i]) for i in range(n)]
pform_magibs = [sp.pxform('GSE', 'JUICE_JMAG_MAGIBS_SCI', ets[i]) for i in range(n)]

b_0 = b_val(ut, r, colat, lon, pos, mat='obs')
b_1 = b_val(ut, r, colat, lon, pos, mat='ibs')
b_2 = b_val(ut, r,  colat, lon, pos, mat='sca')
angle_b_sca = sca_angle(b_2)

#Choose apt. time interval to plot
start_time = np.datetime64('2024-08-20T20:00:00')
end_time = np.datetime64('2024-08-21T00:00:00')
indices = np.where((utcs >= start_time) & (utcs <= end_time))[0]
date = np.array(utcs, dtype='datetime64')

plot_mag_fields(date[indices], indices, b_0, b_1, b_2, angle_b_sca, 'B-Field-Frame.png')

b_0_actual = b_data
# Extract magnetic field vectors
b_0_matrix = extract_magnetic_fields(b_0)
b_0_actual_matrix = extract_magnetic_fields(b_0_actual) #replace with b_data

#Least Squares rotation matrix computation
R, R_errors, euler_angles, euler_errors = compute_rotation_matrix_least_squares(b_0_matrix, b_0_actual_matrix)

# Apply the rotation matrix
b_0_corrected_matrix = apply_rotation_matrix(b_0_matrix, R)

# Convert back to (x, y, z) format
x_0_corrected, y_0_corrected, z_0_corrected = b_0_corrected_matrix.T

print("Least Squares Computed Rotation Matrix:")
print(R)

#SVD

# SVD rotation matrix computation
R = compute_rotation_matrix_svd(b_0_matrix, b_0_actual_matrix)

# Apply the rotation matrix
b_0_corrected_matrix = apply_rotation_matrix(b_0_matrix, R)

# Convert back to (x, y, z) format
x_0_corrected, y_0_corrected, z_0_corrected = b_0_corrected_matrix.T


# Calculate 5% error intervals
error_x = 0.05 * np.abs(b_x)
error_y = 0.05 * np.abs(b_y)
error_z = 0.05 * np.abs(b_z)

angles_formatted = [f"{angle:.2f}°" for angle in euler_angles]
errors_formatted = [f"± {error:.2f}°" for error in euler_errors]

# Create subplots: 2 rows, 1 column
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
font_size = 12
# First subplot: MAGOBS prediction
axs[0].plot(utcs, b_x, label='$B_{x}$ MAG_OBS', color='red')
axs[0].fill_between(utcs, b_x - error_x, b_x + error_x, color='red', alpha=0.2)
axs[0].plot(utcs, b_y, label='$B_{y}$ MAG_OBS', color='lime')
axs[0].fill_between(utcs, b_y - error_y, b_y + error_y, color='lime', alpha=0.2)
axs[0].plot(utcs, b_z, label='$B_{z} $ MAG_OBS', color='blue')
axs[0].fill_between(utcs, b_z - error_z, b_z + error_z, color='blue', alpha=0.2)
axs[0].plot(utcs, b_0[0][0], label='$B_{x}$ IGRF-13', linestyle='--', color='red')
axs[0].plot(utcs, b_0[0][1], label='$B_{y}$ IGRF-13', linestyle='--', color='lime')
axs[0].plot(utcs, b_0[0][2], label='$B_{z}$ IGRF-13', linestyle='--', color='blue')
axs[0].legend(fontsize=10)
axs[0].set_title("MAG_OBS before alignment", fontsize=font_size)

# Second subplot: MAGOBS with alignment
axs[1].plot(utcs, x_0_corrected, label='$B_{0,x}$', linestyle='--', color='red')
axs[1].plot(utcs, y_0_corrected, label='$B_{0,y}$', linestyle='--', color='lime')
axs[1].plot(utcs, z_0_corrected, label='$B_{0,z}$', linestyle='--', color='blue')
axs[1].plot(utcs, b_x, label='$B_{x}$', color='red')
axs[1].fill_between(utcs, b_x - error_x, b_x + error_x, color='red', alpha=0.2)
axs[1].plot(utcs, b_y, label='$B_{y}$', color='lime')
axs[1].fill_between(utcs, b_y - error_y, b_y + error_y, color='lime', alpha=0.2)
axs[1].plot(utcs, b_z, label='$B_{z}$', color='blue')
axs[1].fill_between(utcs, b_z - error_z, b_z + error_z, color='blue', alpha=0.2)
axs[1].set_xlabel('UTC on 20th August 2024', fontsize=font_size)
axs[1].set_title("MAG_OBS after alignment", fontsize=font_size)

calibration_text = (
    f"Calibration parameters:\n"
    f"$\\psi$: {angles_formatted[2]} {errors_formatted[2]}\n"
    f"$\\theta$: {angles_formatted[1]} {errors_formatted[1]}\n"
    f"$\\phi$: {angles_formatted[0]} {errors_formatted[0]}"
)
axs[1].text(0.02, 0.05, calibration_text, transform=axs[1].transAxes, fontsize=font_size,
            verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.8))

# Shared y-axis label
fig.text(0.04, 0.5, 'Magnetic Field Components (nT)', va='center', ha='center', rotation='vertical', fontsize=font_size)

# Format x-axis to show only hours (HH:MM)
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.tight_layout(rect=[0.05, 0.05, 1, 1])
plt.show()

print("SVD Computed Rotation Matrix:")
print(R)

d_x = b_0[0][0] - b_x
d_y = b_0[0][1] - b_y
d_z = b_0[0][2] - b_z

d_x_a = x_0_corrected - b_x
d_y_a = y_0_corrected - b_y
d_z_a = z_0_corrected - b_z

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axs[0].plot(utcs, d_x, label='$ΔB_{x}$', color='red')
axs[0].plot(utcs, d_y, label='$ΔB_{y}$', color='lime')
axs[0].plot(utcs, d_z, label='$ΔB_{z}$', color='blue')
axs[0].set_xlabel('UTC on 20th August 2024', fontsize=font_size)
axs[0].set_title("ΔMAG_OBS before alignment", fontsize=font_size)

axs[1].plot(utcs, d_x_a, label='$ΔB_{x}$', color='red')
axs[1].plot(utcs, d_y_a, label='$ΔB_{y}$', color='lime')
axs[1].plot(utcs, d_z_a, label='$ΔB_{z}$', color='blue')
axs[1].set_xlabel('UTC on 20th August 2024', fontsize=font_size)
axs[1].set_title("ΔMAG_OBS after alignment", fontsize=font_size)

# Shared y-axis label
fig.text(0.04, 0.5, 'Magnetic Field Components (nT)', va='center', ha='center', rotation='vertical', fontsize=font_size)

# Format x-axis to show only hours (HH:MM)
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.tight_layout(rect=[0.05, 0.05, 1, 1])
plt.savefig('delta-mag-obs-c.png', dpi=300)
plt.show()


fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)


delta_modulus = b_0[1] - modulus_b

axs[0].plot(utcs, modulus_b, label='$|B|$ MAG_OBS Data', color='orange')
axs[0].plot(utcs, b_0[1], label='$|B|$ MAG_OBS Data IGRF-13', color='blue')
axs[0].set_xlabel('UTC on 20th August 2024', fontsize=font_size)
axs[0].legend(fontsize=10)

axs[1].plot(utcs, delta_modulus, label='Difference in $|B|$ MAG_OBS', color='black')
axs[1].set_xlabel('UTC on 20th August 2024', fontsize=font_size)
axs[1].legend(fontsize=10)

fig.text(0.04, 0.5, 'Magnetic Field (nT)', va='center', ha='center', rotation='vertical', fontsize=font_size)

axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.tight_layout(rect=[0.05, 0.05, 1, 1])
plt.savefig("")
plt.show()
