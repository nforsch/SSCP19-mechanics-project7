import dolfin as df

from demo import load_geometry


geometry = load_geometry()

endo_coordinates = []
endo_marker = geometry.markers['ENDO'][0]
# Loop over the facets
for facet in df.facets(geometry.mesh):
    # If the facet markers matched that of ENDO
    if geometry.ffun[facet] == endo_marker:
        # Loop over the vertices of that facets
        for vertex in df.vertices(facet):
            endo_coordinates.append(tuple(vertex.midpoint().array()))

# Remove duplicates
endo_coordinates = set(endo_coordinates)
