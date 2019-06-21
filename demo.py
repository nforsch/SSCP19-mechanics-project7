import os
import numpy as np
import dolfin as df
import pulse
import ldrb
import matplotlib.pyplot as plt


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


def load_geometry(h5name='ellipsoid.h5', recreate=False):

    if not os.path.exists(h5name) or recreate:
        create_geometry(h5name)

    geo = pulse.HeartGeometry.from_file(h5name)
    # Scale mesh to a realistic size
    geo.mesh.coordinates()[:] *= 4.5
    return geo


def save_geometry_vis(geometry, folder='geometry'):
    """
    Save the geometry as well as markers and fibers to files
    that can be visualized in paraview
    """
    if not os.path.isdir(folder):
        os.makedirs(folder)

    for attr in ['mesh', 'ffun', 'cfun']:
        print('Save {}'.format(attr))
        df.File('{}/{}.pvd'.format(folder, attr)) << getattr(geometry, attr)

    for attr in ['f0', 's0', 'n0']:
        ldrb.fiber_to_xdmf(getattr(geometry, attr),
                           '{}/{}'.format(folder, attr))


def get_strains(u, v, dx):

    F = pulse.kinematics.DeformationGradient(u)
    E = pulse.kinematics.GreenLagrangeStrain(F, isochoric=False)

    return df.assemble(df.inner(E*v, v) * dx) \
        / df.assemble(df.Constant(1.0) * dx)


def get_nodal_coordinates(u):

    mesh = df.Mesh(u.function_space().mesh())
    V = df.VectorFunctionSpace(mesh, "CG", 1)
    df.ALE.move(mesh, df.interpolate(u, V))
    return mesh.coordinates()


def postprocess(geometry):
    """
    Get strain at nodal values

    Arguments
    ---------
    filename : str
        Filname where to store the results
    """

    coords = [geometry.mesh.coordinates()]
    V = df.VectorFunctionSpace(geometry.mesh, "CG", 2)
    Ef = np.zeros((3, 17))

    u_ED = df.Function(V, "ED_displacement.xml")
    coords.append(get_nodal_coordinates(u_ED))
    for i in range(17):
        Ef[1, i] = get_strains(u_ED, geometry.f0, geometry.dx(i+1))
    EDV = geometry.cavity_volume(u=u_ED)

    u_ES = df.Function(V, "ES_displacement.xml")
    coords.append(get_nodal_coordinates(u_ES))
    for i in range(17):
        Ef[2, i] = get_strains(u_ES, geometry.f0, geometry.dx(i+1))
    ESV = geometry.cavity_volume(u=u_ES)
    # Stroke volume
    SV = EDV - ESV
    # Ejection fraction
    EF = SV / EDV
    print(("EDV: {EDV:.2f} ml\nESV: {ESV:.2f} ml\nSV: {SV:.2f}"
           " ml\nEF: {EF:.2f}").format(EDV=EDV, ESV=ESV, SV=SV, EF=EF))

    # Save nodes as txt at ED and ES
    np.savetxt('coords_ED.txt',coords[1],fmt='%.4f',delimiter=',')
    np.savetxt('coords_ES.txt',coords[2],fmt='%.4f',delimiter=',')

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    for i in range(17):
        j = i // 6
        # from IPython import embed; embed()
        # exit()
        ax[j].plot(Ef[:, i], label="region {}".format(i+1))

    ax[0].set_title("Basal")
    ax[1].set_title("Mid")
    ax[2].set_title("Apical")

    ax[0].set_ylabel("Fiber strain")
    for axi in ax:
        axi.set_xticks(range(3))
        axi.set_xticklabels(["", "ED", "ES"])
        axi.legend()

    plt.show()


def solve(
        geometry,
        EDP=1.0,
        ESP=15.0,
        Ta=60,
        material_parameters=None,
):
    """

    Arguments
    ---------
    EDP : float
        End diastolic pressure
    ESP : float
        End systolic pressure
    Ta : float
        Peak active tension (at ES)
    material_parameters : dict
        A dictionart with parameter in the Guccione model.
        Default:  {'C': 2.0, 'bf': 8.0, 'bt': 2.0, 'bfs': 4.0}
    filename : str
        Filname where to store the results

    """
    # Create model
    activation = df.Function(df.FunctionSpace(geometry.mesh, "R", 0))
    matparams = pulse.Guccione.default_parameters()
    if material_parameters is not None:
        matparams.update(material_parameters)
    material = pulse.Guccione(activation=activation,
                              parameters=matparams,
                              active_model="active_stress",
                              f0=geometry.f0,
                              s0=geometry.s0,
                              n0=geometry.n0)

    lvp = df.Constant(0.0)
    lv_marker = geometry.markers['ENDO'][0]
    lv_pressure = pulse.NeumannBC(traction=lvp,
                                  marker=lv_marker, name='lv')
    neumann_bc = [lv_pressure]

    # Add spring term at the base with stiffness 1.0 kPa/cm^2
    base_spring = 1.0
    robin_bc = [pulse.RobinBC(value=df.Constant(base_spring),
                              marker=geometry.markers["BASE"][0])]

    # Fix the basal plane in the longitudinal direction
    # 0 in V.sub(0) refers to x-direction, which is the longitudinal direction
    def fix_basal_plane(W):
        V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
        bc = df.DirichletBC(V.sub(0),
                            df.Constant(0.0),
                            geometry.ffun, geometry.markers["BASE"][0])
        return bc

    dirichlet_bc = [fix_basal_plane]

    # Collect boundary conditions
    bcs = pulse.BoundaryConditions(dirichlet=dirichlet_bc,
                                   neumann=neumann_bc,
                                   robin=robin_bc)

    # Create the problem
    problem = pulse.MechanicsProblem(geometry, material, bcs)

    xdmf = df.XDMFFile(df.mpi_comm_world(), 'output.xdmf')

    # Solve the problem
    print(("Do an initial solve with pressure = 0 kPa "
          "and active tension = 0 kPa"))
    problem.solve()
    u, p = problem.state.split()
    xdmf.write(u, 0.0)
    print("LV cavity volume = {} ml".format(geometry.cavity_volume(u=u)))

    # Solve for ED
    print(("Solver for ED with pressure = {} kPa and active tension = 0 kPa"
           "".format(EDP)))
    pulse.iterate.iterate(problem, lvp, EDP)

    u, p = problem.state.split(deepcopy=True)
    xdmf.write(u, 1.0)
    df.File("ED_displacement.xml") << u
    print("LV cavity volume = {} ml".format(geometry.cavity_volume(u=u)))

    # Solve for ES
    print(("Solver for ES with pressure = {} kPa and active tension = {} kPa"
           "".format(ESP, Ta)))
    pulse.iterate.iterate(problem, lvp, ESP, initial_number_of_steps=20)
    pulse.iterate.iterate(problem, activation, Ta, initial_number_of_steps=20)

    u, p = problem.state.split(deepcopy=True)
    xdmf.write(u, 2.0)
    df.File("ES_displacement.xml") << u
    print("LV cavity volume = {} ml".format(geometry.cavity_volume(u=u)))


def main():
    geometry = load_geometry(recreate=True)
    save_geometry_vis(geometry)
    solve(geometry,
          EDP=1.0,
          ESP=15.0,
          Ta=60,
          material_parameters=None)
    postprocess(geometry)





if __name__ == "__main__":
    main()
