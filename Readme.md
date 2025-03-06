## Overview / 项目概述

This project is a Python script that analyzes Muscle-up movements in a video using computer vision techniques. It detects key points of the human body, calculates physical metrics such as displacement, time, power, horsepower, and maximum velocity during the ascending phase of a Muscle-up, and overlays the results on the video along with a "Beast Probability" score.

本项目是一个 Python 脚本，利用计算机视觉技术分析视频中的 Muscle-up（双力臂,缩写mu）、pull-up（引体向上）等动作（同样适用于其他大多数动作，注意修改过滤无效动作等细节的逻辑，后面会更新），后面以muscle-up为例。它检测人体关键点，计算 Muscle-up 上升阶段的位移、时间、功率、马力和最大速度等物理指标，并在视频上叠加结果以及“Beast 概率”评分。

---

## Features / 功能特性

- **Pose Detection**: Uses MediaPipe Pose to detect 33 key points of the human body in real-time.  
  **姿态检测**: 使用 MediaPipe Pose 实时检测人体的 33 个关键点。
- **Physical Metrics Calculation**: Computes displacement, time, power, horsepower, and max velocity based on shoulder movement.  
  **物理指标计算**: 根据肩膀移动计算位移、时间、功率、马力和最大速度。
- **Visualization**: Overlays key points, bounding box, Beast Probability, and analysis results on the video.  
  **可视化**: 在视频上叠加关键点、目标检测框、Beast 概率和分析结果。
- **Output**: Saves the processed video with annotations and returns a list of Muscle-up data.  
  **输出**: 保存带注释的处理后视频，并返回 Muscle-up 数据列表。

---

## Dependencies / 依赖项

To run this project, you need to install the following Python libraries:  
要运行此项目，需要安装以下 Python 库：

- `opencv-python` (cv2): For video processing and visualization.  
  用于视频处理和可视化。
- `mediapipe`: For pose detection and key point tracking.  
  用于姿态检测和关键点跟踪。
- `numpy`: For numerical computations.  
  用于数值计算。

Install them using pip:  
使用 pip 安装：
```bash
pip install opencv-python mediapipe numpy
```

---

## Usage / 使用方法

### Code Example / 代码示例
```python
from muscle_up_analysis import process_muscle_up_video

# Define input parameters / 定义输入参数
video_path = 'path/to/your/video.mp4'           # Input video path / 输入视频路径
height_cm = 180                                 # Your height in cm / 你的身高（厘米）
weight_kg = 70                                  # Your weight in kg / 你的体重（千克）
output_video_path = 'path/to/output_video.mp4'  # Output video path / 输出视频路径

# Run the analysis / 运行分析
results = process_muscle_up_video(video_path, height_cm, weight_kg, start_time=1.0, output_video_path=output_video_path)

# Print results / 打印结果
for i, data in enumerate(results):
    print(f"Muscle-up {i+1}:")
    print(f"  Displacement: {data['displacement']:.2f} m")
    print(f"  Time: {data['time']:.2f} s")
    print(f"  Power: {data['power']:.2f} W")
    print(f"  Horsepower: {data['horsepower']:.2f} hp")
    print(f"  Max Velocity: {data['max_velocity']:.2f} m/s")
```

### Parameters / 参数说明
- `video_path` (str): Path to the input video file.  
  输入视频文件的路径。
- `height_cm` (float): Your height in centimeters (used for scaling).  
  你的身高（厘米，用于缩放）。
- `weight_kg` (float): Your weight in kilograms (used for power calculation).  
  你的体重（千克，用于功率计算）。
- `start_time` (float, optional): Time in seconds to start analysis (default: 1.0).  
  开始分析的时间（秒，默认值为 1.0）。
- `output_video_path` (str, optional): Path to save the output video (default: 'output_video.mp4').  
  保存输出视频的路径（默认值为 'output_video.mp4'）。

---

## Output / 输出内容

- **Processed Video**: A video file with annotated key points, bounding box, Beast Probability, and Muscle-up metrics.  
  **处理后的视频**: 包含关键点、目标检测框、Beast 概率和 Muscle-up 指标的视频文件。
- **Data List**: A list of dictionaries containing metrics for each detected Muscle-up:  
  **数据列表**: 包含每个检测到的 Muscle-up 的指标的字典列表：
  - `displacement` (m): Vertical displacement during ascent.  
    上升阶段的垂直位移（米）。
  - `time` (s): Duration of the ascent.  
    上升阶段的持续时间（秒）。
  - `power` (W): Average power output.  
    平均功率输出（瓦）。
  - `horsepower` (hp): Power in horsepower.  
    马力（匹）。
  - `max_velocity` (m/s): Maximum velocity during ascent.  
    上升阶段的最大速度（米/秒）。
    
[example:muscle-up](https://www.bilibili.com/video/BV1xk9RY3E9k/?vd_source=bc499aa91cecc9b938f44372fe471cce#reply256332236225)
[example:pull-up](https://www.bilibili.com/video/BV1jHRKYsEtb/?spm_id_from=333.1387.homepage.video_card.click&vd_source=bc499aa91cecc9b938f44372fe471cce)
---

## Notes / 注意事项

- Ensure the video clearly shows the full body performing Muscle-ups for accurate detection.  
  确保视频清晰显示执行 Muscle-up 的全身，以获得准确的检测结果。
- The script assumes the person is facing the camera with minimal occlusion.  
  脚本假设人物面向摄像头，且遮挡较少。
- Adjust `start_time` if the Muscle-up action begins later in the video.  
  如果 Muscle-up 动作在视频中较晚开始，请调整 `start_time`。
