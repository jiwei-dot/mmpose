# 简介
基于单目视频的动作捕捉，其输入为单目视频，输出为BVH文件。本项目将其拆分为5个步骤，视频中的每帧图片首先通过**Step1**获取全身二维关键点坐标，之后通过**Step2**将二维关键点提升到三维空间，再然后经过**Step3**将三维关键点坐标通过优化算法转化为BVH文件格式，最后的**Step4**和**Step5**是脚部优化部分，**Step4**根据人体下半身二维关键点坐标判断脚部触地状态，**Step5**根据该状态将触地脚部贴近到地面。

## Step1: 从视频生成二维关键点

```shell
python custom_demo/video_to_kpts2d.py \
    --video-path ${VIDEO_PATH} \
    --out-root ${OUT_ROOT} \
    [--person-det-config ${PERSON_DET_CONFIG}] \
    [--person-det-checkpoint ${PERSON_DET_CHECKPOINT}] \
    [--person-det-cat-id ${PERSON_DET_CAT_ID}] \
    [--hand-det-config ${HAND_DET_CONFIG}] \
    [--hand-det-checkpoint ${HAND_DET_CHECKPOINT}] \
    [--hand-nms-thr ${HAND_NMS_THR}] \
    [--bbox-thr ${BBOX_THR}] \
    [--wholebody-kps2d-config ${WHOLEBODY_KPS2D_CONFIG}] \
    [--wholebody-kps2d-checkpoint ${WHOLEBODY_KPS2D_CONFIG}] \
    [--hand-kps2d-config ${HAND_KPS2D_CONFIG}] \
    [--hand-kps2d-checkpoint ${HAND_KPS2D_CHECKPOINT}] \
    [--kpt-thr ${KPT_THR}] \
    [--device ${GPU_ID or CPU}] \
    [--use-oks-tracking] \
    [--tracking-thr ${TRACKING_THR}]
```
参数解释：
- video-path: 需要进行动捕的视频路径，**必须指定**
- out-root: 保存所生成二维关键点文件的路径，**必须指定**
- person-det-config: 人体检测网络配置文件
- person-det-checkpoint: 人体检测网络权重文件
- person-det-cat-id:人体检测网络目标类别
- hand-det-config: 手部检测网络配置文件
- hand-det-checkpoint: 手部检测网络权重文件
- hand-nms-thr: 手部检测网络NMS阈值
- bbox-thr: 人体检测网络和手部检测网络的预测框置信度阈值
- wholebody-kps2d-config: 全身二维关键点检测网络配置文件
- wholebody-kps2d-checkpoint: 全身二维关键点检测网络权重文件
- hand-kps2d-config: 手部二维关键点检测网络配置文件
- hand-kps2d-checkpoint: 手部二维关键点检测网络权重文件
- kpt-thr: 关键点置信度阈值，用在将独立检测出的手分配给人
- device: 运行设备
- use-oks-tracking: 是否开启oks-tracking目标追踪
- tracking-thr: 目标追踪阈值


示例：
```shell
python custom_demo/video_to_kpts2d.py
    --video-path workspace/videos/cxk.mp4 \
    --out-root workspace
```
在程序运行结束后，workspace目录下会多出一个名为``video_cxk_kpts2d.pkl``文件


## Step2: 将二维关键点提升到三维空间

<figure>
<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMWNiMDIyNjZiMzIxOGU5YWMxZGU5NzViMTU5MjA2NDA5ZWQ5YTFlMiZlcD12MV9pbnRlcm5hbF9naWZzX2dpZklkJmN0PWc/uHc5J5HGIGshC8Wql9/giphy.gif" width=1300/>
<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNTZjN2U3MmQ5ZDYxNGE2ZDI4MmFkZDc3YWE3ZmM0MzE5ZjY1MTVhZCZlcD12MV9pbnRlcm5hbF9naWZzX2dpZklkJmN0PWc/exfsgj90D6lyPf3tPO/giphy.gif" width=1300/>
</figure>



```shell
python custom_demo/kpts2d_to_kpts3d.py \
    --pkl-path ${PKL_PATH} \
    --video-path ${VIDEO_PATH} \
    --out-root ${OUT_ROOT} \
    [--device ${GPU_ID or CPU}] \
    [--body3d-lifter-config ${BODY3D_LIFTER_CONFIG}] \
    [--body3d-lifter-checkpoint ${BODY3D_LIFTER_CHECKPOINT}] \
    [--lefthand3d-lifter-config ${LEFTHAND3D_LIFTER_CONFIG}] \
    [--lefthand3d-lifter-checkpoint ${LEFTHAND3D_LIFTER_CHECKPOINT}] \
    [--righthand3d-lifter-config ${RIGHTHAND3D_LIFTER_CONFIG}]\
    [--righthand3d-lifter-checkpoint ${RIGHTHAND3D_LIFTER_CHECKPOINT}] \
    [--leftfoot3d-lifter-config ${LEFTFOOT3D_LIFTER_CONFIG}] \
    [--leftfoot3d-lifter-checkpoint ${LEFTFOOT3D_LIFTER_CHECKPOINT}] \
    [--rightfoot3d-lifter-config ${RIGHTFOOT3D_LIFTER_CONFIG}] \
    [--rightfoot3d-lifter-checkpoint ${RIGHTFOOT3D_LIFTER_CHECKPOINT}] \
    [--smooth] \
    [--smooth-filter-cfg ${SMOOTH_FILTER_CFG}] \
    [--norm-pose-2d] \
    [--complex-process-hand] \
    [--hand-area-thr ${HAND_AREA_THR}]
```

### 参数解释：

- pkl-path: Step1所生成的二维关键点文件, **必须指定**
- video-path: 需要进行动捕的视频路径，**必须指定**
- out-root: 保存所生成三维关键点文件的路径，**必须指定**
- device: 运行设备
- body3d-lifter-config: 躯干部位提升网络配置文件
- body3d-lifter-checkpoint: 躯干部位提升网络权重文件
- lefthand3d-lifter-config: 左手部位提升网络配置文件
- lefthand3d-lifter-checkpoint: 左手部位提升网络权重文件
- righthand3d-lifter-config: 右手部位提升网络配置文件
- righthand3d-lifter-checkpoint: 右手部位提升网络权重文件
- leftfoot3d-lifter-config: 左脚部位提升网络配置文件
- leftfoot3d-lifter-checkpoint: 左脚部位提升网络权重文件
- rightfoot3d-lifter-config: 右脚部位提升网络配置文件
- rightfoot3d-lifter-checkpoint: 右脚部位提升网络权重文件
- smooth: 是否对二维姿态进行平滑，**强烈建议开启**
- smooth-filter-cfg: 姿态平滑配置文件
- norm-pose-2d: 对输入的二维姿态进行标准化处理，**强烈建议开启**
- complex-process-hand: 是否对手部进行复杂的后处理操作
- hand-area-thr: 对手部进行复杂后处理时需要指定的阈值，该值为手部与人体面积之比，当大于该阈值时认为网络预测不准确


### 示例：
```shell
python custom_demo/kpts2d_to_kpts3d.py \
    --pkl-path workspace/video_cxk_kpts2d.pkl \
    --video-path workspace/videos/cxk.mp4 \
    --out-root workspace \
    --smooth \
    --norm-pose-2d \
    --complex-process-hand
```
在程序运行结束后，workspace目录下会多出一个名为``video_cxk_kpts3d.pkl``文件和一个名为``vis_cxk.mp4``视频文件


## Step3: 将三维关键点转化为BVH文件(该方法非常慢, 10~15s/frame)
<figure>
<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNjM3N2YxNjFiY2ZlZmMwNDAxOTdhMzU2OWRkYzFmOTdmZTZmNjZhNSZlcD12MV9pbnRlcm5hbF9naWZzX2dpZklkJmN0PWc/JKWjQxvhbvMn7Rx5U6/giphy.gif" width=250/>
<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMzY4MTE3MGI0NWQzNjlkMDNkNmNjNTQxY2ZhNGMyNDU0NjQ1MmYxNCZlcD12MV9pbnRlcm5hbF9naWZzX2dpZklkJmN0PWc/HqXYCYIgOZEx0hIbdn/giphy.gif" width=250/>
</figure>

```shell
python custom_demo/kpts3d_to_bvh.py \
    --pkl-path ${PKL_PATH} \
    --out-root ${OUT_ROOT} \
    --template-bvh ${TEMPLATE_BVH} \
    --track-id ${TRACK_ID} \
    --device ${GPU_ID or CPU}
```

### 参数解释：

- pkl-path: Step2所生成的三维关键点文件, **必须指定**
- out-root: 保存所生成BVH文件的路径，**必须指定**
- template-bvh: T-pose文件路径
- track-id: 一个视频里面可能有多个人，具体针对哪一个人生成BVH文件，默认为0
-  device: 运行设备

### 示例：
```shell
python custom_demo/kpts3d_to_bvh.py \
    --pkl-path workspace/video_cxk_kpts3d.pkl \
    --out-root workspace
```
在程序运行结束后，workspace目录下会多出一个名为``cxk.bvh``文件

## Step4: 脚部触地判断(用于脚部IK)

<figure>
<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZGE3NjJjNDQxNjgwYzMwMTY2YjU1ZjRiYjBmODJmNDJjOGE2NWNiYiZlcD12MV9pbnRlcm5hbF9naWZzX2dpZklkJmN0PWc/Tf9fknbjnver9VUHiu/giphy.gif" width=250/>
<img src="https://media.giphy.com/media/d6hgSkSLhSIBlIIqpC/giphy.gif" width=250/>
</figure>


```shell
python custom_demo/check_foot_contact.py \
    --pkl-path ${PKL_PATH} \
    --out-root ${OUT_ROOT} \
    [--checkpoint ${CHECKPOINT}] \
    [--device ${GPU_ID or CPU}] \
    [--show-result] \
    [--video-path ${VIDEO_PATH}]
```

### 参数解释：
- pkl-path: Step1所生成的二维关键点文件, **必须指定**
- out-root: 保存所生成脚触地文件的路径，**必须指定**
- checkpoint: 脚部触地网络权重文件
- device: 运行设备
- show-result: 是否可视化触地结果
- video-path: 当需要可视化触地结果时需要指定视频路径


### 示例：
```shell
python custom_demo/check_foot_contact.py \
    --pkl-path workspace/video_cxk_kpts2d.pkl \
    --out-root workspace \
    --show-result \
    --video-path workspace/videos/cxk.mp4
```
在程序运行结束后，workspace目录下会多出一个名为``footcontact_cxk.pkl``文件和一个名为``footcontact_cxk.mp4``视频文件

## Step5: 脚部IK(这一部分代码存在bug, 使用two-bone-ik算法时目标末端点总是不可到达)
```shell
python custom_demo/foot_ik.py \
    --bvh-file ${BVH_FILE} \
    --out-root ${OUT_ROOT} \
    --footcontact-pkl-file ${FOOTCONTACT_PKL_FILE} \
    [--track-id ${TRACK_ID}]
```

### 参数解释：
- bvh-file: Step3所生成的BVH文件, **必须指定**
- out-root: 保存所生成新的BVH文件的路径，**必须指定**
- footcontact-pkl-file: Step4所生成的触地状态文件, **必须指定**
- track-id: 一个视频里面可能有多个人，具体针对哪一个人进行脚部IK，默认为0

### 示例：
```shell
python custom_demo/foot_ik.py \
    --pkl-path workspace/cxk.bvh \
    --out-root workspace \
    --footcontact-pkl-file workspace/footcontact_cxk.pkl
```
在程序运行结束后，workspace目录下会多出一个名为``new_cxk.bvh``文件



