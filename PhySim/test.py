from mayavi import mlab
f = mlab.figure()
f.scene.movie_maker.record = True
mlab.test_mesh_sphere_anim()

