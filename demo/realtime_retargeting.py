# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import sys
import time
from collections import deque
from queue import Queue
from threading import Event, Lock, Thread

import torch
import socket
import json
import cv2
import numpy as np
import time as Time
import os

import mmcv
from mmcv.tensorrt import TRTWraper


sys.path.append("E:/sqp/mmdeploy/build/bin/Release")
from mmdeploy_python import Detector

from mmhuman3d.utils.transforms import rotmat_to_aa
from mmhuman3d.core.renderer.mpr_renderer.smpl_realrender import \
    VisualizerMeshSMPL  # noqa: E501
from tools.retargeting.fast_retargeting_singleton import retarget_one_frame as fast_retarget
from tools.retargeting.fast_retargeting_singleton import (
    XIAOTAO_NAME_TO_BONE,
    SRC_SKELETON_JSON,
    TGT_SKELETON_JSON
)

from mmhuman3d.models.body_models.builder import build_body_model
from mmhuman3d.utils.demo_utils import (
    StopWatch,
    convert_verts_to_cam_coord,
    build_smooth_func,
    smooth_process,
    process_mmdet_results,
)

from mmhuman3d.apis import run_tensorrt_model

try:
    # from mmdet.apis import inference_detector, init_detector
    from mmdet.core import bbox2result
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

try:
    import psutil
    psutil_proc = psutil.Process()
except (ImportError, ModuleNotFoundError):
    psutil_proc = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mesh-trt-file',
        type=str,
        default='data/checkpoints/pare.trt',
        help='MMHuman3d tensorRT file path')
    parser.add_argument('--cam-id', type=str, default='0')
    parser.add_argument(
        '--det-mmdeploy-model',
        type=str,
        default='../mmdeploy/work_dir/ssd',
        help='path of mmdeploy SDK model dumped by model converter')
    parser.add_argument(
        '--det-num-classes',
        type=int,
        default=80,
        help='class number which can be found in the mmdet config file')    
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--device', default='cuda', help='Device used for inference')
    parser.add_argument(
        '--body_model_dir',
        type=str,
        default='data/body_models/smpl',
        help='Body models file path')
    parser.add_argument(
        '--bbox_thr',
        type=float,
        default=0.6,
        help='Bounding box score threshold')
    parser.add_argument(
        '--smooth-type',
        type=str,
        default='savgol',
        help="Smooth the data through the specified type. "
        "Select in [None,'oneeuro', 'gaus1d', 'savgol','smoothnet', "
        "'smoothnet_windowsize8','smoothnet_windowsize16', "
        "'smoothnet_windowsize32','smoothnet_windowsize64']. ")
    parser.add_argument(
        '--show',
        type=str,
        default=True,
        help='whether to show visualizations. This may reduce the frame rate')
    parser.add_argument(
        '--window-size',
        type=int,
        default=21,
        help='The size of the filter window')
    parser.add_argument(
        '--out_video_file',
        type=str,
        default='./output.mp4',
        help='Record the video into a file. This may reduce the frame rate')
    parser.add_argument(
        '--out_video_fps',
        type=int,
        default=30,
        help='Set the FPS of the output video file.')
    parser.add_argument(
        '--buffer_size',
        type=int,
        default=1,
        help='Frame buffer size. If set -1, the buffer size will be '
        'automatically inferred from the display delay time. Default: -1')
    parser.add_argument(
        '--inference_fps',
        type=int,
        default=30,
        help='Maximum inference FPS. This is to limit the resource consuming '
        'especially when the detection and pose model are lightweight and '
        'very fast. Default: 10.')
    parser.add_argument(
        '--display_delay',
        type=int,
        default=0,
        help='Delay the output video in milliseconds. This can be used to '
        'align the output video and inference results. The delay can be '
        'disabled by setting a non-positive delay time. Default: 0')
    parser.add_argument(
        '--synchronous_mode',
        type=str,
        default=False,
        help='Enable synchronous mode that video I/O and inference will be '
        'temporally aligned. Note that this will reduce the display FPS.')
    parser.add_argument(
        '--target-ip',
        type=str,
        default="localhost",
        help='Target ip. Default: localhost')
    parser.add_argument(
        '--target-port',
        type=int,
        default=54321,
        help='Target port. Default: 54321')
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
    return parser.parse_args()


def read_camera():
    # init video reader
    print('Thread "input" started')
    cam_id = args.cam_id
    if cam_id.isdigit():
        cam_id = int(cam_id)
    vid_cap = cv2.VideoCapture(cam_id)
    if not vid_cap.isOpened():
        print(f'Cannot open camera (ID={cam_id})')
        exit()
    # vid_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # vid_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    while not event_exit.is_set():
        # capture a camera frame
        ret_val, frame = vid_cap.read()
        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if ret_val:
            ts_input = time.time()

            event_inference_done.clear()
            with input_queue_mutex:
                input_queue.append((ts_input, frame))

            if args.synchronous_mode:
                event_inference_done.wait()
            frame_buffer.put((ts_input, frame))
        else:
            # input ending signal
            frame_buffer.put((None, None))
            break

    vid_cap.release()


def inference_detection():
    print('Thread "det" started')
    stop_watch = StopWatch(window=10)
    min_interval = 1.0 / args.inference_fps
    _ts_last = None  # timestamp when last inference was done

    while True:
        while len(input_queue) < 1:
            time.sleep(0.001)
        with input_queue_mutex:
            ts_input, frame = input_queue.popleft()
        # inference detection
        with stop_watch.timeit('Det'):
            T = Time.time()
            for i in range(1):
                # mmdet_results = inference_detector(det_model, frame)
                det_bboxes, det_labels, _ = detector(frame)     
                mmdet_results = bbox2result(
                    det_bboxes,
                    det_labels,
                    args.det_num_classes
                    )
            print(f"Det: {Time.time() - T}")
            # mmdet_results = inference_detector(det_model, frame)
        t_info = stop_watch.report_strings()
        with det_result_queue_mutex:
            det_result_queue.append((ts_input, frame, t_info, mmdet_results))

        # limit the inference FPS
        _ts = time.time()
        if _ts_last is not None and _ts - _ts_last < min_interval:
            time.sleep(min_interval - _ts + _ts_last)
        _ts_last = time.time()


def inference_mesh():
    print('Thread "mesh" started')
    stop_watch = StopWatch(window=10)

    while True:
        while len(det_result_queue) < 1:
            time.sleep(0.001)
        with det_result_queue_mutex:
            ts_input, frame, t_info, mmdet_results = det_result_queue.popleft()

        with stop_watch.timeit('Mesh'):
            det_results = process_mmdet_results(
                mmdet_results, cat_id=args.det_cat_id, bbox_thr=args.bbox_thr)
            
            # smooth bbox
            if det_results and args.smooth_type is not None:
                smooth_bbox_queue.append(det_results[0]['bbox'])
                if len(smooth_bbox_queue) == args.window_size:
                    bbox = smooth_process(
                        np.array(smooth_bbox_queue)[:, None, :],
                        smooth_func)[-1].squeeze()
                    det_results[0]['bbox'] = bbox
            
            T = Time.time()
            for _ in range(1):
                mesh_results = run_tensorrt_model(
                    mesh_model,
                    frame,
                    det_results,
                    bbox_thr=args.bbox_thr,
                    format='xyxy')
                
            print(f"Mesh: {Time.time()-T}")
            
        t_info += stop_watch.report_strings()
        with mesh_result_queue_mutex:
            mesh_result_queue.append((ts_input, t_info, mesh_results))

        event_inference_done.set()


def retargeting():
    print('Thread "retargeting" started')
    stop_watch = StopWatch(window=10)
 
    # initialize result status
    ts_inference = None  # timestamp of the latest inference result
    fps_inference = 0.  # infenrece FPS
    t_delay_inference = 0.  # inference result time delay
    mesh_results = None
    t_info = []  # upstream time information (list[str])

    vid_out = None  # video writer    
    vid_out_ori = None
    while True:
        with stop_watch.timeit('_FPS_'):
            # acquire a frame from buffer
            ts_input, frame = frame_buffer.get()
            
            # input ending signal
            if ts_input is None:
                break
            img = frame.copy()
            # get mesh estimation results
            if len(mesh_result_queue) > 0:
                with mesh_result_queue_mutex:
                    _result = mesh_result_queue.popleft()
                    _ts_input, t_info, mesh_results = _result

                _ts = time.time()
                if ts_inference is not None:
                    fps_inference = 1.0 / (_ts - ts_inference)
                ts_inference = _ts
                t_delay_inference = (_ts - _ts_input) * 1000            
                        
            if mesh_results:
                bboxes_xyxy = mesh_results[0]['bbox']
                # show bounding boxes
                mmcv.imshow_bboxes(
                    frame,
                    bboxes_xyxy[None],
                    colors='green',
                    top_k=-1,
                    thickness=2,
                    show=False)                
                pred_cams = mesh_results[0]['camera']
                transl = np.concatenate([
                    pred_cams[..., [1]], pred_cams[..., [2]], 2 * 5000. /
                    (224 * pred_cams[..., [0]] + 1e-9)
                ], -1)            
                smpl_poses = mesh_results[0]['smpl_pose']
                
                # smooth smpl
                if mesh_results and args.smooth_type is not None:
                    # smpl_poses = mesh_results[0]['smpl_pose']
                    smooth_smpl_queue.append(smpl_poses)
                    if len(smooth_smpl_queue) == args.window_size:

                        smpl_poses = smooth_process(
                            np.array(smooth_smpl_queue).reshape(args.window_size, 24, 9),
                            smooth_func).reshape(args.window_size, 24, 3, 3)[[-1]]
                        
                if smpl_poses.shape[1:] == (24, 3, 3):
                    smpl_poses = rotmat_to_aa(smpl_poses)
                elif smpl_poses.shape[1:] == (24, 3):
                    smpl_poses = smpl_poses
                else:
                    raise (f'Wrong shape of `smpl_pose`: {smpl_poses.shape}')

                body_pose=smpl_poses[:, 1:]
                global_orient=smpl_poses[:, 0]
                smpl_beta = mesh_results[0]['smpl_beta']
                
                smpl_dict ={
                    'body_pose':body_pose,
                    'global_orient':global_orient,
                    'betas':smpl_beta,
                    # 'transl':transl
                    }
                
                # retargeting
                T = Time.time()
                for _ in range(1):
                    motion_data = fast_retarget(
                        smpl_dict,
                        tgt_name2bone=XIAOTAO_NAME_TO_BONE,
                        src_skeleton_json=SRC_SKELETON_JSON,
                        tgt_skeleton_json=TGT_SKELETON_JSON,
                    )   
                    
                print(f"Ret: {Time.time()-T}")
            
                motion_data.pop('transl')
                motion_data = {"XiaoTao":motion_data}
                motion_data = json.dumps(motion_data).encode('UTF-16LE')
                
                # send to ue 
                client.sendto(motion_data,(target_ip,target_port))
                
                if args.show:
                    T = time.time()
                    for _ in range(1):
                        result = body_model(
                            global_orient=torch.tensor(global_orient),
                            body_pose=torch.tensor(body_pose).view(-1, 69),
                            betas=torch.tensor(smpl_beta)
                            )
                    
                        verts = result['vertices'].detach().cpu().numpy()
                        verts, _ = convert_verts_to_cam_coord(
                            verts, pred_cams, bboxes_xyxy, focal_length=5000.)
                        if isinstance(verts, np.ndarray):
                            verts = torch.tensor(verts).to(args.device).squeeze()
                
                        frame = renderer(verts, frame)
                    print(time.time()-T)
            # delay control
            if args.display_delay > 0:
                t_sleep = args.display_delay * 0.001 - (time.time() - ts_input)
                print(t_sleep)
                if t_sleep > 0:
                    time.sleep(t_sleep)
            t_delay = (time.time() - ts_input) * 1000

            # show time information
            t_info_display = stop_watch.report_strings()  # display fps
            t_info_display.append(f'Inference FPS: {fps_inference:>5.1f}')
            t_info_display.append(f'Delay: {t_delay:>3.0f}')
            t_info_display.append(
                f'Inference Delay: {t_delay_inference:>3.0f}')
            t_info_str = ' | '.join(t_info_display + t_info)
            sys_info = [
                # f'RES: {img.shape[1]}x{img.shape[0]}',
                f'Buffer: {frame_buffer.qsize()}/{frame_buffer.maxsize}'
            ]
            if psutil_proc is not None:
                sys_info += [
                    f'CPU: {psutil_proc.cpu_percent():.1f}%',
                    f'MEM: {psutil_proc.memory_percent():.1f}%'
                ]
            sys_info_str = ' | '.join(sys_info)
            
            print(t_info_str)
            print(sys_info_str)
            # print(
            #     f'Inference FPS: {fps_inference:>5.1f}, '
            #     f'Inference Delay: {t_delay_inference:>3.0f} '
            #     f'Delay: {t_delay:>3.0f}'
            # )
            
            # save the output video frame
            if args.out_video_file is not None:
                if vid_out is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = args.out_video_fps
                    frame_size = (frame.shape[1], frame.shape[0])
                    vid_out = cv2.VideoWriter(args.out_video_file, fourcc, fps,
                                              frame_size)

                vid_out.write(frame)

            # save the output video frame
            if args.out_video_file is not None:
                if vid_out_ori is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = args.out_video_fps
                    frame_size = (img.shape[1], img.shape[0])
                    vid_out_ori = cv2.VideoWriter('./output_ori.mp4', fourcc, fps,
                                              frame_size)

                vid_out_ori.write(img)
            
            if args.show:
                cv2.imshow('realtime_demo', frame)
                keyboard_input = cv2.waitKey(1)
                if keyboard_input in (27, ord('q'), ord('Q')):
                    break
    if vid_out is not None:
        vid_out.release()
        vid_out_ori.release()
    if args.show:     
        cv2.destroyAllWindows()

                
def main():
    global args
    global frame_buffer
    global smooth_smpl_queue, smooth_bbox_queue, smooth_func
    global input_queue, input_queue_mutex
    global det_result_queue, det_result_queue_mutex
    global mesh_result_queue, mesh_result_queue_mutex
    global mesh_model, detector
    global event_exit, event_inference_done
    global body_model, renderer 
    global client, target_ip, target_port

    args = parse_args()
    assert has_mmdet, 'Please install mmdet to run the demo.'
    
    # set conn
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target_ip = args.target_ip
    target_port = args.target_port
    
    args.cam_id = 'demo/resources/single_person_demo.mp4'
    cam_id = args.cam_id
    if cam_id.isdigit():
        cam_id = int(cam_id)
    vid_cap = cv2.VideoCapture(cam_id)
    if not vid_cap.isOpened():
        print(f'Cannot open camera (ID={cam_id})')
        exit()
    _, frame = vid_cap.read()
    resolution = list(frame.shape[:2])
    vid_cap.release()

    # build smooth func
    if args.smooth_type is not None:
        smooth_func = build_smooth_func(
            smooth_type = args.smooth_type,
            window_size=args.window_size)
        
    # build body model for visualization
    body_model = build_body_model(
        dict(
            type='SMPL',
            gender='neutral',
            num_betas=10,
            model_path=args.body_model_dir))
    # build renderer
    renderer = VisualizerMeshSMPL(
        device=args.device, body_models=body_model, resolution=resolution)
    
    
    # build body model for visualization
    body_model = build_body_model(
        dict(
            type='SMPL',
            gender='neutral',
            num_betas=10,
            model_path=args.body_model_dir))

    # build detection model
    detector = Detector(
        model_path=args.det_mmdeploy_model, 
        device_name=args.device, 
        device_id=0)
    
    # build mmhuman3d models
    # input_names = ['input']
    # output_names = ['3245', '3401', '3322', '3323']
    mesh_model = TRTWraper(args.mesh_trt_file, args.input_name, args.output_names)

    # frame buffer
    if args.buffer_size > 0:
        buffer_size = args.buffer_size
    else:
        # infer buffer size from the display delay time
        # assume that the maximum video fps is 30
        buffer_size = round(30 * (1 + max(args.display_delay, 0) / 1000.))
    frame_buffer = Queue(maxsize=buffer_size)

    smooth_smpl_queue = deque(maxlen=args.window_size)
    smooth_bbox_queue = deque(maxlen=args.window_size)
    # queue of input frames
    # element: (timestamp, frame)
    input_queue = deque(maxlen=1)
    input_queue_mutex = Lock()

    # queue of detection results
    # element: tuple(timestamp, frame, time_info, det_results)
    det_result_queue = deque(maxlen=1)
    det_result_queue_mutex = Lock()

    # queue of detection/pose results
    # element: (timestamp, time_info, pose_results_list)
    mesh_result_queue = deque(maxlen=1)
    mesh_result_queue_mutex = Lock()

    try:
        event_exit = Event()
        event_inference_done = Event()
        t_input = Thread(target=read_camera, args=())
        t_det = Thread(target=inference_detection, args=(), daemon=True)
        t_mesh = Thread(target=inference_mesh, args=(), daemon=True)

        t_input.start()
        t_det.start()
        t_mesh.start()

        # run retargeting in the main thread
        retargeting()
        # join the input thread (non-daemon)
        t_input.join()

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
