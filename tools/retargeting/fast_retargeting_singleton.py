import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as spRotation
from mathutils import Matrix, Quaternion, Vector

PWD = Path(__file__).resolve().parent
SRC_SKELETON_JSON = PWD / "smpl_skeleton_info.json"
TGT_SKELETON_JSON = PWD / "xiaotao_skeleton_info.json"

BONE_NAMES = [
    'Pelvis',
    'Spine1',
    'Spine2',
    'Chest',
    'Neck',
    'Head',
    'Scapula_L',
    'Shoulder_L',
    'Elbow_L',
    'Wrist_L',
    # 'index_A_L',
    # 'index_B_L',
    # 'index_C_L',
    # 'middle_A_L',
    # 'middle_B_L',
    # 'middle_C_L',
    # 'pinky_A_L',
    # 'pinky_B_L',
    # 'pinky_C_L',
    # 'ring_A_L',
    # 'ring_B_L',
    # 'ring_C_L',
    # 'thumb_A_L',
    # 'thumb_B_L',
    # 'thumb_C_L',
    'Scapula_R',
    'Shoulder_R',
    'Elbow_R',
    'Wrist_R',
    # 'index_A_R',
    # 'index_B_R',
    # 'index_C_R',
    # 'middle_A_R',
    # 'middle_B_R',
    # 'middle_C_R',
    # 'pinky_A_R',
    # 'pinky_B_R',
    # 'pinky_C_R',
    # 'ring_A_R',
    # 'ring_B_R',
    # 'ring_C_R',
    # 'thumb_A_R',
    # 'thumb_B_R',
    # 'thumb_C_R',
    'Hip_L',
    'Knee_L',
    'Ankle_L',
    'Toes_L',
    'Hip_R',
    'Knee_R',
    'Ankle_R',
    'Toes_R',
]

XIAOTAO_NAME_TO_BONE = {
    "Ankle_L": "Ankle_L",
    "Ankle_R": "Ankle_R",
    "Chest": "Chest_M",
    "Elbow_L": "Elbow_L",
    "Elbow_R": "Elbow_R",
    "Head": "Head_M",
    "Hip_L": "Hip_L",
    "Hip_R": "Hip_R",
    "Knee_L": "Knee_L",
    "Knee_R": "Knee_R",
    "Neck": "Neck_M",
    "Pelvis": "Root_M",
    "Scapula_L": "Scapula_L",
    "Scapula_R": "Scapula_R",
    "Shoulder_L": "Shoulder_L",
    "Shoulder_R": "Shoulder_R",
    "Spine1": "Spine1_M",
    "Spine2": "Spine2_M",
    "Toes_L": "Toes_L",
    "Toes_R": "Toes_R",
    "Wrist_L": "Wrist_L",
    "Wrist_R": "Wrist_R",
    "index_A_L": "IndexFinger1_L",
    "index_A_R": "IndexFinger1_R",
    "index_B_L": "IndexFinger2_L",
    "index_B_R": "IndexFinger2_R",
    "index_C_L": "IndexFinger3_L",
    "index_C_R": "IndexFinger3_R",
    "middle_A_L": "MiddleFinger1_L",
    "middle_A_R": "MiddleFinger1_R",
    "middle_B_L": "MiddleFinger2_L",
    "middle_B_R": "MiddleFinger2_R",
    "middle_C_L": "MiddleFinger3_L",
    "middle_C_R": "MiddleFinger3_R",
    "pinky_A_L": "PinkyFinger1_L",
    "pinky_A_R": "PinkyFinger1_R",
    "pinky_B_L": "PinkyFinger2_L",
    "pinky_B_R": "PinkyFinger2_R",
    "pinky_C_L": "PinkyFinger3_L",
    "pinky_C_R": "PinkyFinger3_R",
    "ring_A_L": "RingFinger1_L",
    "ring_A_R": "RingFinger1_R",
    "ring_B_L": "RingFinger2_L",
    "ring_B_R": "RingFinger2_R",
    "ring_C_L": "RingFinger3_L",
    "ring_C_R": "RingFinger3_R",
    "thumb_A_L": "ThumbFinger1_L",
    "thumb_A_R": "ThumbFinger1_R",
    "thumb_B_L": "ThumbFinger2_L",
    "thumb_B_R": "ThumbFinger2_R",
    "thumb_C_L": "ThumbFinger3_L",
    "thumb_C_R": "ThumbFinger3_R"
}
SMPL_NAME_TO_BONE = {
    "Ankle_L": "left_ankle",
    "Ankle_R": "right_ankle",
    "Chest": "spine3",
    "Elbow_L": "left_elbow",
    "Elbow_R": "right_elbow",
    "Head": "head",
    "Hip_L": "left_hip",
    "Hip_R": "right_hip",
    "Knee_L": "left_knee",
    "Knee_R": "right_knee",
    "Neck": "neck",
    "Pelvis": "pelvis",
    "Scapula_L": "left_collar",
    "Scapula_R": "right_collar",
    "Shoulder_L": "left_shoulder",
    "Shoulder_R": "right_shoulder",
    "Spine1": "spine1",
    "Spine2": "spine2",
    "SpinePattern": "",
    "Toes_L": "left_foot",
    "Toes_R": "right_foot",
    "Wrist_L": "left_wrist",
    "Wrist_R": "right_wrist",
    "index_A_L": "left_index1",
    "index_A_R": "right_index1",
    "index_B_L": "left_index2",
    "index_B_R": "right_index2",
    "index_C_L": "left_index3",
    "index_C_R": "right_index3",
    "middle_A_L": "left_middle1",
    "middle_A_R": "right_middle1",
    "middle_B_L": "left_middle2",
    "middle_B_R": "right_middle2",
    "middle_C_L": "left_middle3",
    "middle_C_R": "right_middle3",
    "pinky_A_L": "left_pinky1",
    "pinky_A_R": "right_pinky1",
    "pinky_B_L": "left_pinky2",
    "pinky_B_R": "right_pinky2",
    "pinky_C_L": "left_pinky3",
    "pinky_C_R": "right_pinky3",
    "ring_A_L": "left_ring1",
    "ring_A_R": "right_ring1",
    "ring_B_L": "left_ring2",
    "ring_B_R": "right_ring2",
    "ring_C_L": "left_ring3",
    "ring_C_R": "right_ring3",
    "thumb_A_L": "left_thumb1",
    "thumb_A_R": "right_thumb1",
    "thumb_B_L": "left_thumb2",
    "thumb_B_R": "right_thumb2",
    "thumb_C_L": "left_thumb3",
    "thumb_C_R": "right_thumb3"
}


def calculate_one_bone_retargeting(
    is_pelvis: bool,
    # skeleton
    tgt_matrix_world: Matrix,
    src_matrix_world: Matrix,
    src2tgt_hip_align_scaling: Tuple[float, float, float],
    # rest pose
    mat_0_src: Matrix,
    mat_0_tgt: Matrix,
    # pose
    mat_src: Matrix,
) -> Matrix:
    """Do one bone retargeting."""
    src2tgt_scaling_mat = Matrix()
    for i in range(3):
        src2tgt_scaling_mat[i][i] = src2tgt_hip_align_scaling[i]

    tgt_matrix_world_inv = tgt_matrix_world.inverted()
    # consider scaling alignment
    src_matrix_world_inv = (src_matrix_world @ src2tgt_scaling_mat).inverted()

    if is_pelvis:
        # * in world space, src & tgt pelvis have the save transl and rot
        # ? tgt_matrix_world @ mat_tgt @ (tgt_matrix_world @ mat_0_tgt).inverted() == \
        # ?     src_matrix_world @ mat_src @ (src_matrix_world @ mat_0_src).inverted()
        mat_tgt = (
            tgt_matrix_world_inv
            @ src_matrix_world
            @ mat_src
            @ mat_0_src.inverted()
            @ src_matrix_world_inv
            @ tgt_matrix_world
            @ mat_0_tgt
        )
        # # tgt_bone.matrix_basis @ tgt_bone.matrix.inverted() @ mat_tgt
        # mat_basis = tgt_local_matrix.inverted() @ mat_tgt
    else:
        # * in armature space, src & tgt have the some pose Rotation
        # ? mat_tgt @ mat_0_tgt.inverted() == mat_src @ mat_0_src.inverted()
        mat_tgt = mat_src @ mat_0_src.inverted() @ mat_0_tgt
        # mat_basis = tgt_local_matrix.inverted() @ mat_tgt
        # # reset T to the rest pose
        # tgt_0_loc, _, _ = mat_0_tgt.decompose()
        # for i in range(3):
        #     mat_basis[i][3] = tgt_0_loc[i]
    return mat_tgt


class BoneInfo:
    __slot__ = ('pose_bone_info', 'name', 'bone_name', 'matrix', 'matrix_basis', 'matrix_local', 'location', 'rotation', 'parent_name', 'parent')

    """Static bone info at rest pose."""
    def __init__(self, pose_bone_info: dict):
        self.pose_bone_info = pose_bone_info
        self.name: str = self.pose_bone_info["name"]
        self.bone_name: str = self.pose_bone_info["bone_name"]
        self.matrix = Matrix(self.pose_bone_info["matrix"])
        self.matrix_basis = Matrix(self.pose_bone_info["matrix_basis"])
        # self._matrix_local = Matrix(self.pose_bone_info["matrix_local"])
        self.matrix_local = Matrix(self.pose_bone_info["matrix_local"])
        self.location = Vector(self.pose_bone_info["location"])
        self.rotation = Quaternion(self.pose_bone_info["rotation"])
        self.parent_name: str = self.pose_bone_info["parent_name"] or ''
        self.parent: Optional['BoneInfo'] = None

        # If not pose bone, the bone is not intended to have animation
        self.is_pose_bone: bool = not self.name.startswith('_')

    def __str__(self) -> str:
        return f'BoneInfo<name="{self.name}">'

    def __repr__(self) -> str:
        return self.__str__()


class SkeletonInfo:
    _instances = {}
    bone_names = BONE_NAMES

    @classmethod
    def from_json(cls, json_path) -> 'SkeletonInfo':
        if json_path not in cls._instances:
            cls._instances[json_path] = SkeletonInfo(json_path)
        instance = cls._instances[json_path]
        return instance

    def __init__(self, json_path):
        print("🐞 json_path :", json_path)
        with open(json_path) as f:
            self.data = json.load(f)
        self.pose_bones_info = self.data["pose_bones"]
        self.armature_matrix = Matrix(self.data["armature_matrix"])
        self.pelvis_height: float = (
            self.armature_matrix @ Matrix(self.pose_bones_info['Pelvis']['matrix'])
        ).decompose()[0].z
        # Setup bones and their relationship
        self.bones: Dict[str, BoneInfo] = OrderedDict()

        def _get_bone_info_(name: Optional[str]) -> Optional[BoneInfo]:
            if not name:
                return None
            elif name in self.bones:
                bone_info = self.bones[name]
            else:
                bone_info = BoneInfo(self.pose_bones_info[name])
                bone_info.parent = _get_bone_info_(bone_info.parent_name)
                self.bones[name] = bone_info
            return bone_info

        for name in self.pose_bones_info.keys():
            _get_bone_info_(name)


class SMPLMotion:
    PELVIS = 'Pelvis'
    SMPL_IDX_TO_NAME: Dict[int, str] = OrderedDict([
        (0, PELVIS),
        (1, 'Hip_L'),
        (2, 'Hip_R'),
        (3, 'Spine1'),
        (4, 'Knee_L'),
        (5, 'Knee_R'),
        (6, 'Spine2'),
        (7, 'Ankle_L'),
        (8, 'Ankle_R'),
        (9, 'Chest'),
        (10, 'Toes_L'),
        (11, 'Toes_R'),
        (12, 'Neck'),
        (13, 'Scapula_L'),
        (14, 'Scapula_R'),
        (15, 'Head'),
        (16, 'Shoulder_L'),
        (17, 'Shoulder_R'),
        (18, 'Elbow_L'),
        (19, 'Elbow_R'),
        (20, 'Wrist_L'),
        (21, 'Wrist_R'),
        (22, ''),
        (23, ''),
    ])
    NAME_TO_SMPL_IDX = OrderedDict([(v, k) for k, v in SMPL_IDX_TO_NAME.items() if v])
    BONE_NAMES = [x for x in SMPL_IDX_TO_NAME.values() if x]
    PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19, 20, 21]

    # In order to make the smpl head up to +z
    GLOBAL_ORIENT_ADJUSTMENT = spRotation.from_euler('xyz', np.deg2rad([180, 0, 0]))    # type: ignore

    def __init__(self, smpl_data, insert_rest_pose=True) -> None:
        self.smpl_data = smpl_data
        self.n_frames = smpl_data['body_pose'].shape[0]

        transl = smpl_data.get('transl', np.zeros([self.n_frames, 3], dtype=np.float32))
        global_orient = self.smpl_data['global_orient'].reshape(self.n_frames, 3)
        body_pose = smpl_data['body_pose'].reshape(self.n_frames, -1, 3)
        # Adjust in order to make the smpl head up to +z
        global_orient = (
            self.GLOBAL_ORIENT_ADJUSTMENT * spRotation.from_rotvec(global_orient)   # type: ignore
        ).as_rotvec()
        transl = np.concatenate(
            [transl[:, 0:1], -transl[:, 1:2], -transl[:, 2:3]],
            axis=1,
        )
        if insert_rest_pose:
            # insert the 0 frame as a T-Pose
            transl_0 = np.zeros([1, 3])
            global_orient_0 = np.zeros([1, 3])
            body_pose_0 = np.zeros([1, body_pose.shape[1], 3])
            transl = np.concatenate([transl_0, transl], axis=0)
            global_orient = np.concatenate([global_orient_0, global_orient], axis=0)
            body_pose = np.concatenate([body_pose_0, body_pose], axis=0)
        self.transl = transl.astype(np.float32)
        self.body_pose = body_pose.astype(np.float32)
        self.global_orient = global_orient.astype(np.float32)

    def get_transl(self, frame=0) -> np.ndarray:
        return self.transl[frame, :3]

    def get_global_orient(self, frame=0) -> np.ndarray:
        return self.global_orient[frame, :3]

    def get_bone_rotvec(self, bone_name, frame=0) -> np.ndarray:
        if bone_name == self.PELVIS:
            return self.get_global_orient(frame)
        else:
            idx = self.NAME_TO_SMPL_IDX[bone_name]
            return self.body_pose[frame, idx - 1, :3]

    def get_bone_rotation(self, bone_name, frame=0) -> spRotation:
        rotvec = self.get_bone_rotvec(bone_name, frame)
        return spRotation.from_rotvec(rotvec)   # type: ignore

    def get_bone_matrix_basis(self, bone_name, frame=0) -> np.ndarray:
        """pose2rest: relative to the bone space at rest pose."""
        if bone_name == self.PELVIS:
            transl = self.get_transl(frame)
        else:
            transl = np.zeros(3)
        rot = self.get_bone_rotation(bone_name, frame)
        matrix_basis = rot.as_matrix()
        matrix_basis = np.pad(matrix_basis, (0, 1))
        matrix_basis[:3, 3] = transl
        matrix_basis[3, 3] = 1
        return matrix_basis

    @classmethod
    def get_parent_bone_name(cls, bone_name) -> Optional[str]:
        idx = cls.NAME_TO_SMPL_IDX[bone_name]
        parent_idx = cls.PARENTS[idx]
        if parent_idx == -1:
            return None
        else:
            return cls.SMPL_IDX_TO_NAME[parent_idx]


def calculate_bone_matrix(
    matrix_record: Dict[str, Matrix],
    matrix_basis_record: Dict[str, Matrix],
    bone_info: BoneInfo,
) -> Matrix:
    """Calculate the pose_bone's matrix (bone2arm)"""
    bone_name = bone_info.name
    if bone_name in matrix_record:
        return matrix_record[bone_name]

    local = bone_info.matrix_local
    if bone_info.is_pose_bone:
        basis = matrix_basis_record[bone_name]
    else:
        basis = bone_info.matrix_basis
    parent = bone_info.parent
    if parent is None:
        mat = local @ basis
    else:
        parent_local = parent.matrix_local
        mat = (
            calculate_bone_matrix(matrix_record, matrix_basis_record, parent)
            @ parent_local.inverted() @ local @ basis
        )

    return mat


def retarget_one_frame(
    src_smpl_data: np.ndarray,
    tgt_name2bone: Dict[str, str],
    src_skeleton_json: Path = SRC_SKELETON_JSON,
    tgt_skeleton_json: Path = TGT_SKELETON_JSON,
) -> Dict[str, Tuple[float, float, float]]:
    """Do skeleton retargeting

    Args:
        src_smpl_data (np.ndarray): source motion data in SMPL/SMPLX.
        tgt_name2bone (Dict[str, str]): mapping unified names to the target skeleton's actual bone names.
        src_name2bone (Optional[Dict[str, str]], optional): mapping unified names to the source
            (SMPL) skeleton's actual names. If it is None, the returning src_motion_data_renamed will be empty.
            Defaults to None.
        src_skeleton_json (Path, optional): json of source skeleton's info. Defaults to SRC_SKELETON_JSON.
        tgt_skeleton_json (Path, optional): json of target skeleton's info. Defaults to TGT_SKELETON_JSON.
        insert_src_rest_pose_at_0 (bool, optional): whether to insert a rest pose to src_smpl_data
            at the 0-th frame. Defaults to True.

    Returns:
        Dict[str, List[List[float]]]: motion_data
        # ! the target motion data is in UE space
    """
    src_motion = SMPLMotion(src_smpl_data, insert_rest_pose=False)

    src_skeleton = SkeletonInfo.from_json(src_skeleton_json)
    tgt_skeleton = SkeletonInfo.from_json(tgt_skeleton_json)
    src2tgt_hip_align_scaling = tuple([tgt_skeleton.pelvis_height / src_skeleton.pelvis_height] * 3)
    src_matrix_world = src_skeleton.armature_matrix
    tgt_matrix_world = tgt_skeleton.armature_matrix

    motion_data = {}
    for frame in range(1):
        matrix_src_record: Dict[str, Matrix] = {}
        matrix_basis_src_record: Dict[str, Matrix] = {}
        matrix_tgt_record: Dict[str, Matrix] = {}
        matrix_basis_tgt_record: Dict[str, Matrix] = {}

        for name in src_skeleton.bone_names:
            src_bone = src_skeleton.bones[name]
            tgt_bone = tgt_skeleton.bones[name]
            is_pelvis = (name == src_motion.PELVIS)
            # * pose2rest
            mat_basis_src = Matrix(src_motion.get_bone_matrix_basis(name, frame))
            matrix_basis_src_record[name] = mat_basis_src
            # * bone2armature with pose
            mat_src = calculate_bone_matrix(matrix_src_record, matrix_basis_src_record, src_bone)
            # * restBone2posedParent
            if tgt_bone.parent is None:
                tgt_local_mat = tgt_bone.matrix_local
            else:
                # bone2armature of the parent
                tgt_parent_matrix = calculate_bone_matrix(
                    matrix_tgt_record, matrix_basis_tgt_record,
                    tgt_bone.parent
                )
                tgt_local_mat = (
                    tgt_parent_matrix
                    @ tgt_bone.parent.matrix_local.inverted()
                    @ tgt_bone.matrix_local
                )
            # * bone2armature at the 0-th frame
            mat_0_src = src_bone.matrix
            mat_0_tgt = tgt_bone.matrix

            # [*] Retargeting
            mat_tgt = calculate_one_bone_retargeting(
                is_pelvis=is_pelvis,
                tgt_matrix_world=tgt_matrix_world,
                src_matrix_world=src_matrix_world,
                src2tgt_hip_align_scaling=src2tgt_hip_align_scaling,
                mat_0_src=mat_0_src,
                mat_0_tgt=mat_0_tgt,
                mat_src=mat_src,
            )
            mat_basis_tgt = tgt_local_mat.inverted() @ mat_tgt
            matrix_basis_tgt_record[name] = mat_basis_tgt

            # [*] Record the result
            T, Q, _ = mat_basis_tgt.decompose()
            euler = _bl_quat_to_ue(Q)
            # motion_data.setdefault(name, []).append(euler)
            motion_data[name] = euler
            if is_pelvis:
                transl = _bl_vector_to_ue(T)
                # blender to ue
                motion_data["transl"] = tuple(transl)

    motion_data_renamed = {}
    for name, data in motion_data.items():
        if name != "transl":
            bone_name = tgt_name2bone[name]
            motion_data_renamed[bone_name] = data
    motion_data_renamed["transl"] = motion_data["transl"]

    return motion_data_renamed


def _bl_quat_to_ue(quat: Quaternion) -> Tuple[float, float, float]:
    """From blender to ue.
    Only suitable for matrix in world space or armature space.
    Very strange but UE4's euler angles in around x/y axis follow the right-handed convention,
        and the z axis angles follow the left-handed convention.
    So the we convert the coordinate system as vectors do, then negate z-axis euler.
    """
    rot = spRotation.from_quat([quat.x, quat.y, quat.z, quat.w])    # type: ignore
    euler = rot.as_euler('xyz', degrees=True)
    # bl to ue rotation conversion: z -> -z
    euler = (euler[0], -euler[1], -euler[2])
    return euler


def _bl_vector_to_ue(vec: Vector) -> Tuple[float, float, float]:
    """From blender to ue. (y -> -y)"""
    return (vec[0], -vec[1], vec[2])


def main(
    src_smpl_path: Path,
    tgt_name2bone: Dict[str, str] = XIAOTAO_NAME_TO_BONE,
    # src_name2bone: Dict[str, str] = SMPL_NAME_TO_BONE,
    src_skeleton_json: Path = SRC_SKELETON_JSON,
    tgt_skeleton_json: Path = TGT_SKELETON_JSON,
    # insert_src_rest_pose_at_0: bool = False,
):
    src_smpl_data = np.load(src_smpl_path, allow_pickle=True)
    src_smpl_data = src_smpl_data['smpl'].item()

    import time
    ctime = time.time()
    motion_data = retarget_one_frame(
        src_smpl_data,
        tgt_name2bone=tgt_name2bone,
        src_skeleton_json=src_skeleton_json,
        tgt_skeleton_json=tgt_skeleton_json,
    )
    cost = ctime - time.time()
    print("cost time:", cost)

    return {"XiaoTao": motion_data}


if __name__ == '__main__':
    motion_data = main(
        src_smpl_path=Path("./inference_result.npz"),
    )
    print(motion_data)
