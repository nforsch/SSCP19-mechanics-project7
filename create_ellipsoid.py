import os
import numpy as np
import dolfin as df
import pulse
import ldrb


def create_geometry(h5name):
    """
    Create an lv-ellipsoidal mesh and fiber fields using LDRB algorithm

    An ellipsoid is given by the equation

    .. math::

        \frac{x^2}{a} + \frac{y^2}{b} + \frac{z^2}{c} = 1

    We create two ellipsoids, one for the endocardium and one
    for the epicardium and subtract them and then cut the base.
    For simplicity we assume that the longitudinal axis is in
    in :math:`x`-direction and as default the base is located
    at the :math:`x=0` plane.
    """

    # Number of subdivision (higher -> finer mesh)
    N = 13

    # Parameter for the endo ellipsoid
    a_endo = 1.5
    b_endo = 0.5
    c_endo = 0.5
    # Parameter for the epi ellipsoid
    a_epi = 2.0
    b_epi = 1.0
    c_epi = 1.0
    # Center of the ellipsoid (same of endo and epi)
    center = (0.0, 0.0, 0.0)
    # Location of the base
    base_x = 0.0

    # Create a lv ellipsoid mesh with longitudinal axis along the x-axis
    geometry = ldrb.create_lv_mesh(
        N=N,
        a_endo=a_endo,
        b_endo=b_endo,
        c_endo=c_endo,
        a_epi=a_epi,
        b_epi=b_epi,
        c_epi=c_epi,
        center=center,
        base_x=base_x
    )


    # Select fiber angles for rule based algorithm
    angles = dict(alpha_endo_lv=60,  # Fiber angle on the endocardium
                  alpha_epi_lv=-60,  # Fiber angle on the epicardium
                  beta_endo_lv=0,    # Sheet angle on the endocardium
                  beta_epi_lv=0)     # Sheet angle on the epicardium

    fiber_space = 'Lagrange_1'

    # Compte the microstructure
    fiber, sheet, sheet_normal = ldrb.dolfin_ldrb(mesh=geometry.mesh,
                                                  fiber_space=fiber_space,
                                                  ffun=geometry.ffun,
                                                  markers=geometry.markers,
                                                  **angles)

    # Compute focal point
    focal = np.sqrt(a_endo**2 - (0.5 * (b_endo + c_endo))**2)
    # Make mesh according to AHA-zons
    pulse.geometry_utils.mark_strain_regions(mesh=geometry.mesh, foc=focal)

    mapper = {'lv': 'ENDO', 'epi': 'EPI', 'rv': 'ENDO_RV', 'base': 'BASE'}
    m = {mapper[k]: (v, 2) for k, v in geometry.markers.items()}

    pulse.geometry_utils.save_geometry_to_h5(
        geometry.mesh, h5name, markers=m,
        fields=[fiber, sheet, sheet_normal]
    )

create_geometry('ellipsoid.h5')
