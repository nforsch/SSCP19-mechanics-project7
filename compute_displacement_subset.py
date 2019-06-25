import os
import numpy as np
import dolfin as df
import pulse
import ldrb
import matplotlib.pyplot as plt
from scipy import spatial

from demo import load_geometry
pi = np.pi

def cart2prolate( focalLength, XYZ ):
    # Convert Cartesian XYZ to Prolate TML
    # TML[0] = theta, TML[1] = mu, TML[2] = lambda

    X = XYZ.T[0]
    Y = XYZ.T[1]
    Z = XYZ.T[2]

    r1 = np.sqrt( Y**2 + Z**2 + (X+focalLength)**2 )
    r2 = np.sqrt( Y**2 + Z**2 + (X-focalLength)**2 )

    lmbda = np.real( np.arccosh((r1+r2)/(2*focalLength)) )
    mu = np.real( np.arccos((r1-r2)/(2*focalLength)) )
    theta = np.arctan2(Z,Y)

    idx = theta<0
    theta[idx] = theta[idx] + 2*np.pi

    TML = np.concatenate(([theta], [mu], [lmbda]))
    return TML

def prolate2cart( focalLength, TML ):
    # Convert Prolate TML to Cartesian XYZ
    # XYZ[0] = X, XYZ[1] = Y, XYZ[2] = Z

    theta = TML[0]
    mu = TML[1]
    lmbda = TML[2]

    X = focalLength * np.cosh(lmbda) * np.cos(mu)
    Y = focalLength * np.sinh(lmbda) * np.sin(mu) * np.cos(theta)
    Z = focalLength * np.sinh(lmbda) * np.sin(mu) * np.sin(theta)

    XYZ = np.concatenate(([X],[Y],[Z]))
    return XYZ

def focal( a, b, c ):
    focalLength = np.sqrt( a**2 - (0.5*(b+c))**2 )
    return focalLength

def get_surface_points(marker):
    coordinates = []
    idxs = []
    # Loop over the facets
    for facet in df.facets(geometry.mesh):
        # If the facet markers matched that of ENDO
        if geometry.ffun[facet] == marker:
            # Loop over the vertices of that facets
            for vertex in df.vertices(facet):
                idxs.append(vertex.global_index())
                # coordinates.append(tuple(vertex.midpoint().array()))
    # Remove duplicates
    idxs = np.array(list(set(idxs)))
    coordinates = geometry.mesh.coordinates()[idxs]
    return coordinates, idxs

def fit_prolate( P ):
    # Sample nodes of mesh using prolate coordinates to get displacements for
    #   same number of points, similar regions across meshes
    # input P = TML from mesh endo/epi

    mu_max = np.amax(P[1]) # find max mu coordinate from mesh
    tree = spatial.KDTree(P[0:2].T) # setup tree for finding nearest point

    idx_match = []
    sample_points = []
    for theta in np.linspace(pi/2,2*pi,4): # theta range
        for mu in np.linspace(0,mu_max,5): # mu ranges from 0 to mu_max based on mesh
            sample_points.append([theta,mu]) # list of sampled [theta,mu] combinations
            distance, index = tree.query([theta,mu]) # find closest point
            idx_match.append(index) # store index of point in endo or epi

    return idx_match

# Define coordinates of ED mesh for endo and epi
geometry = load_geometry('ellipsoid.h5')
# Get nodes ENDO
marker_endo = geometry.markers['ENDO'][0]
endo_coordinates, endo_idxs = get_surface_points(marker_endo)
# Get nodes EPI
marker_epi = geometry.markers['EPI'][0]
epi_coordinates, epi_idxs = get_surface_points(marker_epi)

# convert Cartesian coordinates to Prolate, find maximum mu value
focalLength_endo = focal(4.1,1.6,1.6) # same parameters [a,b,c] used for mesh
focalLength_epi = focal(5,2.9,2.9) # same parameters [a,b,c] used for mesh
TML_endo = cart2prolate(focalLength_endo, endo_coordinates)
TML_epi = cart2prolate(focalLength_epi, epi_coordinates)
# XYZ_endo = prolate2cart(focalLength_endo,TML_endo) # check return XYZ from TML

# Find fit to closest node by varying theta, mu and fitting lambda (store index of node)
idx_match_endo = fit_prolate(TML_endo)
idx_match_epi = fit_prolate(TML_epi)
idx_node_endo = endo_idxs[idx_match_endo].tolist()
idx_node_epi = epi_idxs[idx_match_epi].tolist()
idx_nodes = idx_node_endo + idx_node_epi

# Get displacement between ES and ED using idx_nodes
print('Loading ED and ES mesh coordinates...')
ed_coordinates = np.loadtxt('coords_ED.txt',delimiter=',')
es_coordinates = np.loadtxt('coords_ES.txt',delimiter=',')
displacement = es_coordinates-ed_coordinates # calculate displacement between ED and ES
disp_out = displacement[idx_nodes] # get displacement for nodes in list idx_nodes
print('Saving displacements for %d points' %(len(idx_nodes)))
np.savetxt('displacement.txt',disp_out,fmt='%.8f',delimiter=',')

# from IPython import embed; embed()
