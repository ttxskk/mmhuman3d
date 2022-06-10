import torch

from mmhuman3d.core.visualization.renderer.mpr_renderer import utils
from mmhuman3d.core.visualization.renderer.mpr_renderer.cuda.rasterizer import \
    estimate_normals as estimate_normals_cuda  # noqa: E501
from mmhuman3d.core.visualization.renderer.mpr_renderer.cuda.rasterizer import \
    project_mesh as project_mesh_cuda  # noqa: E501


def estimate_normals(vertices, faces, pinhole, vertices_filter=None):
    if vertices_filter is None:
        utils.is_cuda_tensor(vertices)
        utils.check_shape_len(vertices, 2)
        n = vertices.shape[0]
        vertices_filter = torch.ones((n),
                                     dtype=torch.uint8,
                                     device=vertices.device)
    vertices = vertices.contiguous()
    vertices_ndc = pinhole.project_ndc(vertices)
    coords, normals = estimate_normals_cuda(vertices_ndc, faces, vertices,
                                            vertices_filter, pinhole.h,
                                            pinhole.w)
    return coords, normals


def project_mesh(vertices,
                 faces,
                 vertice_values,
                 pinhole,
                 vertices_filter=None):
    if vertices_filter is None:
        utils.is_cuda_tensor(vertices)
        utils.check_shape_len(vertices, 2)
        n = vertices.shape[0]
        vertices_filter = torch.ones((n),
                                     dtype=torch.uint8,
                                     device=vertices.device)
    vertices = vertices.contiguous()
    vertices_ndc = pinhole.project_ndc(vertices)
    return project_mesh_cuda(vertices_ndc, faces, vertice_values,
                             vertices_filter, pinhole.h, pinhole.w)
