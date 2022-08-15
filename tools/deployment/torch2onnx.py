# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import warnings

import numpy as np
import torch

from mmhuman3d.apis import init_model

try:
    import onnx
    import onnxruntime as rt
except ImportError as e:
    raise ImportError(f'Please install onnx and onnxruntime first. {e}')

try:
    from mmcv.onnx.symbolic import register_extra_symbolics
except ModuleNotFoundError:
    raise NotImplementedError('please update mmcv to version>=1.0.4')


def _convert_batchnorm(module):
    """Convert the syncBNs into normal BN3ds."""
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm3d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


def pytorch2onnx(model,
                 input_shape,
                 opset_version=12,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False,
                 output_names=['heatmap', 'smpl_pose', 'camera', 'smpl_beta'],
                 input_name = ['input']):
    """Convert pytorch model to onnx model.

    Args:
        model (:obj:`nn.Module`): The pytorch model to be exported.
        input_shape (tuple[int]): The input tensor shape of the model.
        opset_version (int): Opset version of onnx used. Default: 12.
        show (bool): Determines whether to print the onnx model architecture.
            Default: False.
        output_file (str): Output onnx model name. Default: 'tmp.onnx'.
        verify (bool): Determines whether to verify the onnx model.
            Default: False.
        output_names (list): Output names for the output nodes defined in
            the onnx model. If the exported model is pare, this list
            should be ['heatmap', 'smpl_pose', 'camera', 'smpl_beta'].
            Default: ['heatmap', 'smpl_pose', 'camera', 'smpl_beta'].
            
            Note: this list may affect the inference process (run_tensorrt_model)
                defined in `apis/inference.py`. If you want to deploy other models,
                please check the onnx output node and assign a corresponding output
                name list. For example, if the dimension of the elements in the
                output node list is `[[b, 3], [b, 24, 3, 3], [b, 10]` ,
                the corresponding output name list is
                `[camera, smpl_pose, smpl_beta]`.
                
        input_name (list): Input name for the input nodes defined in the onnx
            model. Default: ['input'].
    """
    model.cpu().eval()

    one_img = torch.randn(input_shape)

    register_extra_symbolics(opset_version)
    torch.onnx.export(
        model,
        one_img,
        output_file,
        input_names=input_name,
        output_names=output_names,
        export_params=True,
        keep_initializers_as_inputs=True,
        verbose=show,
        opset_version=opset_version)

    print(f'Successfully exported ONNX model: {output_file}')
    if verify:
        # check by onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # check the numerical value
        # get pytorch output
        pytorch_results = model(one_img)
        if isinstance(pytorch_results, dict):
            pytorch_results = list(pytorch_results.values())

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert len(net_feed_input) == 1
        sess = rt.InferenceSession(output_file)
        onnx_results = sess.run(None,
                                {net_feed_input[0]: one_img.detach().numpy()})

        # compare results
        assert len(pytorch_results) == len(onnx_results)
        for pt_result, onnx_result in zip(pytorch_results, onnx_results):
            assert np.allclose(
                pt_result.detach().cpu(), onnx_result, atol=1.e-5
            ), 'The outputs are different between Pytorch and ONNX'
        print('The numerical values are same between Pytorch and ONNX')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMHuman3D models to ONNX')
    parser.add_argument(
        '--config',
        default='configs/pare/hrnet_w32_conv_pare_coco_cache.py',
        help='test config file path')
    parser.add_argument(
        '--checkpoint',
        default='data/checkpoints/hrnet_w32_conv_pare_mosh.pth',
        help='checkpoint file')
    # parser.add_argument('--config', default='configs/hmr/resnet50_hmr_pw3d_e50_cache.py', help='test config file path')  # noqa: E501
    # parser.add_argument('--checkpoint', default='data/checkpoints/resnet50_hmr_pw3d-04f40f58_20211201.pth', help='checkpoint file')  # noqa: E501
    parser.add_argument(
        '--output-names',
        type=list,
        default=['heatmap', 'smpl_pose', 'camera', 'smpl_beta'],
        help="Output names for the output nodes defined in the onnx model." 
            "If the exported model is pare, this list should be"
            " [`heatmap`, `smpl_pose`, `camera`, `smpl_beta`].")
    parser.add_argument(
        '--input_name',
        type=list,
        default=['input'],
        help="Input name for the input nodes defined in the onnx"
            " model. Default: ['input'].")   
    parser.add_argument(
        '--show',
        default=True,
        help='show onnx graph')
    parser.add_argument(
        '--output-file',
        type=str,
        default='data/checkpoints/pare.onnx')
    parser.add_argument('--opset-version', type=int, default=12)
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1, 3, 224, 224],
        help='input size')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    assert args.opset_version == 12, 'MMPose only supports opset 12 now'

    # Following strings of text style are from colorama package
    bright_style, reset_style = '\x1b[1m', '\x1b[0m'
    red_text, blue_text = '\x1b[31m', '\x1b[34m'
    white_background = '\x1b[107m'

    msg = white_background + bright_style + red_text
    msg += 'DeprecationWarning: This tool will be deprecated in future. '
    msg += blue_text + 'Welcome to use the unified model deployment toolbox '
    msg += 'MMDeploy: https://github.com/open-mmlab/mmdeploy'
    msg += reset_style
    warnings.warn(msg)

    model, _ = init_model(args.config, args.checkpoint, device='cpu')
    model = _convert_batchnorm(model)

    # onnx.export does not support kwargs
    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'Please implement the forward method for exporting.')

    # convert model to onnx file
    pytorch2onnx(
        model,
        args.shape,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file,
        verify=True,
        output_names=args.output_names,
        input_name=args.input_name)  # args.verify)
