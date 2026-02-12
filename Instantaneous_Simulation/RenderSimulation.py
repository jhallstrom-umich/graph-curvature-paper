# Modified this from a fresnel/hoomd tutorial, very janky
# This is not intended as a full tutorial on fresnel - see the
# fresnel user documentation (https://fresnel.readthedocs.io/) if you would like to learn more.
import io
import warnings

import fresnel
import packaging.version
import numpy
import PIL
import gsd.hoomd

device = fresnel.Device()
tracer = fresnel.tracer.Path(device=device, w=600, h=600)

rad = 10.0

FRESNEL_MIN_VERSION = packaging.version.parse("0.13.0")
FRESNEL_MAX_VERSION = packaging.version.parse("0.14.0")


def render(snapshot, sample=10):
    if ('version' not in dir(fresnel) or packaging.version.parse(
            fresnel.version.version) < FRESNEL_MIN_VERSION
            or packaging.version.parse(
                fresnel.version.version) >= FRESNEL_MAX_VERSION):
        warnings.warn(
            f"Unsupported fresnel version {fresnel.version.version} - expect errors."
        )


    vertices = [
        (0.5, 0.5, 0.5),
        (0.5, 0.5, -0.5),
        (0.5, -0.5, 0.5),
        (-0.5, 0.5, 0.5),
        (0.5, -0.5, -0.5),
        (-0.5, -0.5, 0.5),
        (-0.5, 0.5, -0.5),
        (-0.5, -0.5, -0.5),
    ]
    vertices = numpy.array(vertices) * 4
    poly_info = fresnel.util.convex_polyhedron_from_vertices(vertices)


    L = snapshot.configuration.box[0]
    scene = fresnel.Scene(device)
    # geometry = fresnel.geometry.Sphere(scene,
    #                                    N=len(snapshot.particles.position),
    #                                    radius=0.5)
    geometry = fresnel.geometry.ConvexPolyhedron(scene,
                                                 poly_info,
                                                 N=len(snapshot.particles.position))
    geometry.material = fresnel.material.Material(color=fresnel.color.linear(
        [252 / 255, 209 / 255, 1 / 255]),
                                                  roughness=1)
    geometry.position[:] = snapshot.particles.position[:]
    geometry.outline_width = 0.04
    box = fresnel.geometry.Box(scene, [L, L, L, 0, 0, 0], box_radius=.1)

    scene.lights = []
    scene.camera = fresnel.camera.Orthographic(position=(0, 0, L * 2),
                                               look_at=(0, 0, 0),
                                               up=(0, 1, 0),
                                               height=L * 1.4 + 1)
    scene.background_alpha = 1
    scene.background_color = (1, 1, 1)
    return IPython.display.Image(tracer.sample(scene, samples=sample)._repr_png_())


def render_2types(snapshot, sample=10):
    if ('version' not in dir(fresnel) or packaging.version.parse(
            fresnel.version.version) < FRESNEL_MIN_VERSION
            or packaging.version.parse(
                fresnel.version.version) >= FRESNEL_MAX_VERSION):
        warnings.warn(
            f"Unsupported fresnel version {fresnel.version.version} - expect errors."
        )
    central_color = fresnel.color.linear([252 / 255, 41 / 255, 0 / 255])
    constituent_color = fresnel.color.linear([93 / 255, 210 / 255, 252 / 255])

    L = snapshot.configuration.box[0]
    scene = fresnel.Scene(device)
    geometry = fresnel.geometry.Sphere(scene,
                                       N=len(snapshot.particles.position),
                                       radius=rad)
    geometry.material = fresnel.material.Material(color=[0, 0, 0],
                                                  roughness=0.5,
                                                  primitive_color_mix=1.0)
    geometry.position[:] = snapshot.particles.position[:]
    geometry.color[snapshot.particles.typeid[:] == 0] = central_color
    geometry.radius[snapshot.particles.typeid[:] == 0] = rad
    geometry.color[snapshot.particles.typeid[:] == 1] = constituent_color
    geometry.outline_width = 0.04
    box = fresnel.geometry.Box(scene, [L, L, 0, 0, 0, 0], box_radius=.02)

    scene.lights = []
    scene.camera = fresnel.camera.Orthographic(position=(0, 0, L * 2),
                                               look_at=(0, 0, 0),
                                               up=(0, 1, 0),
                                               height=L * 1.4 + 1)
    scene.background_alpha = 1
    scene.background_color = (1, 1, 1)
    print(type(tracer.sample(scene, samples=sample)._repr_png_()))
    bytes = tracer.sample(scene, samples=sample)._repr_png_()
    image = PIL.Image.open(io.BytesIO(bytes))
    print(save_string)
    image.save(save_string)
    return None


def render_frame(snapshot, particles=None, is_solid=None):
    if ('version' not in dir(fresnel) or packaging.version.parse(
            fresnel.version.version) < FRESNEL_MIN_VERSION
            or packaging.version.parse(
                fresnel.version.version) >= FRESNEL_MAX_VERSION):
        warnings.warn(
            f"Unsupported fresnel version {fresnel.version.version} - expect errors."
        )

    central_color = fresnel.color.linear([252 / 255, 41 / 255, 0 / 255])
    constituent_color = fresnel.color.linear([93 / 255, 210 / 255, 252 / 255])

    N = snapshot.particles.N
    L = snapshot.configuration.box[0]
    if particles is not None:
        N = len(particles)
    if is_solid is not None:
        N = int(numpy.sum(is_solid))

    scene = fresnel.Scene(device)
    geometry = fresnel.geometry.Sphere(scene,
                                       N=len(snapshot.particles.position),
                                       radius=rad)
    geometry.material = fresnel.material.Material(color=fresnel.color.linear(
        [0.01, 0.74, 0.26]),
                                                  roughness=1)
    if particles is None and is_solid is None:
        geometry.position[:] = snapshot.particles.position[:]
    elif particles is not None:
        geometry.position[:] = snapshot.particles.position[particles, :]
    elif is_solid is not None:
        geometry.position[:] = snapshot.particles.position[numpy.ix_(
            is_solid, [0, 1, 2])]

    geometry.color[snapshot.particles.typeid[:] == 0] = central_color
    geometry.radius[snapshot.particles.typeid[:] == 0] = rad
    geometry.color[snapshot.particles.typeid[:] == 1] = constituent_color

    geometry.outline_width = 0
    box = fresnel.geometry.Box(scene,
                               snapshot.configuration.box,
                               box_radius=.1)

    scene.lights = []
    scene.camera = fresnel.camera.Orthographic(position=(0, 0, L * 2),
                                               look_at=(0, 0, 0),
                                               up=(0, 1, 0),
                                               height=L * 1.4 + 1)
    scene.background_color = (1, 1, 1)
    return tracer.sample(scene, samples=1)

def render_movie(frames, particles=None, is_solid=None, total_time=10000, filename="out.gif"):
    if is_solid is None:
        is_solid = [None] * len(frames)
    a = render_frame(frames[0], particles, is_solid[0])

    im0 = PIL.Image.fromarray(a[:, :, 0:3], mode='RGB').convert(
        "P", palette=PIL.Image.Palette.ADAPTIVE)
    ims = []
    for i, f in enumerate(frames[1:]):
        a = render_frame(f, particles, is_solid[i])
        im = PIL.Image.fromarray(a[:, :, 0:3], mode='RGB')
        im_p = im.quantize(palette=im0)
        ims.append(im_p)

    blank = numpy.ones(shape=(im.height, im.width, 3), dtype=numpy.uint8) * 255
    im = PIL.Image.fromarray(blank, mode='RGB')
    im_p = im.quantize(palette=im0)
    ims.append(im_p)

    f = io.BytesIO()
    im0.save(filename, 'gif', save_all=True, append_images=ims, duration=numpy.round(total_time/len(ims)), loop=0)
    #im0.save(f, 'gif', save_all=True, append_images=ims, duration=numpy.round(total_time/len(ims)), loop=0)

    #size = len(f.getbuffer()) / 1024
    #if (size > 3000):
    #    warnings.warn(f"Large GIF: {size} KiB")
    #return IPython.display.display(IPython.display.Image(data=f.getvalue()))
    return None

save_string = "RandomizedConfiguration.png"
random = gsd.hoomd.open("randomized.gsd")
render_2types(random[0])

traj = gsd.hoomd.open('trajectory.gsd')
print(len(traj))
print(traj[0].particles.N)

for save_frame in [10, 50, 100, 150]:
    save_string = "frame{}.png".format(save_frame)
    render_2types(traj[save_frame])

print("GIF time")
#render_movie(traj[0:len(traj)], particles=list(range(traj[0].particles.N)), total_time=20000, filename="Trajectory.gif")
#render_movie(traj[len(traj)-2:len(traj)], particles=list(range(traj[0].particles.N)), total_time=100000, filename="LastTwo.gif")
render_movie(traj[:int(len(traj)//4):4], particles=list(range(traj[0].particles.N)), total_time=10000, filename="Fourths_1.gif")
render_movie(traj[int(1*len(traj)//4):int(2*len(traj)//4):4], particles=list(range(traj[0].particles.N)), total_time=10000, filename="Fourths_2.gif")
render_movie(traj[int(2*len(traj)//4):int(3*len(traj)//4):4], particles=list(range(traj[0].particles.N)), total_time=10000, filename="Fourths_3.gif")
render_movie(traj[int(3*len(traj)//4)::4], particles=list(range(traj[0].particles.N)), total_time=10000, filename="Fourths_4.gif")
