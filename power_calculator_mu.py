import cv2
import mediapipe as mp
import numpy as np
import random

# 初始化 MediaPipe Pose 模型和绘图工具
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def process_muscle_up_video(video_path, height_cm, weight_kg, start_time=1.0, output_video_path='output_video.mp4'):
    """
    处理 muscle-up 视频，检测关键点，计算上升阶段的位移、时间、功率、马力和最大速度，
    并在画面上显示结果、目标检测框和 Beast 概率。
    
    参数:
        video_path (str): 输入视频文件路径
        height_cm (float): 身高，单位：厘米
        weight_kg (float): 体重，单位：千克
        start_time (float): 开始分析的时间（秒），默认为 4.0 秒
        output_video_path (str): 输出视频文件路径，默认为 'output_video.mp4'
    
    返回:
        list: 包含每个 muscle-up 的位移 (m)、时间 (s)、功率 (W)、马力 (hp) 和最大速度 (m/s) 的字典列表
    """
    # 打开视频
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError("无法打开视频文件，请检查路径是否正确")

    # 获取视频参数
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = int(start_time * fps)

    # 初始化 Pose 模型
    pose = mp_pose.Pose(
        static_image_mode=False, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )

    # 设置输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # 初始化变量
    pixel_height = None
    scale_factor = None
    prev_shoulder_y = None
    prev_time = None
    muscle_up_data = []
    ascend_start_frame = None
    ascend_start_y = None
    max_height_y = None
    velocities = []
    ascend_time = 0  # 累计有效上升时间
    beast_probability = random.uniform(50, 90)  # 初始化 Beast 概率

    frame_num = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        current_time = frame_num / fps

        # 转换为 RGB 格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # 绘制关键点和连接线（红色）
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            # 获取关键点
            landmarks = results.pose_landmarks.landmark
            shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            current_shoulder_y = shoulder.y * frame_height

            # 标定：使用身高计算像素到实际距离的比例
            if pixel_height is None:
                head = landmarks[mp_pose.PoseLandmark.NOSE]
                left_foot = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
                right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
                foot_y = max(left_foot.y, right_foot.y) * frame_height
                pixel_height = abs(head.y * frame_height - foot_y)
                scale_factor = (height_cm / 100) / pixel_height 

            # 绘制目标检测框（蓝色）
            x_min = min([lm.x * frame_width for lm in landmarks]) - 20
            x_max = max([lm.x * frame_width for lm in landmarks]) + 20
            y_min = min([lm.y * frame_height for lm in landmarks]) - 20
            y_max = max([lm.y * frame_height for lm in landmarks]) + 20
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 255), 2)  # 蓝色框

            # 计算 Beast 概率
            if frame_num < start_frame:
                # 在 muscle-up 开始前随机波动
                beast_probability += random.uniform(-5, 5)
                beast_probability = max(50, min(90, beast_probability))
            elif prev_shoulder_y is not None:
                # 在 muscle-up 过程中根据速度和功率计算
                delta_y = current_shoulder_y - prev_shoulder_y
                delta_time = current_time - prev_time
                velocity = (delta_y * scale_factor) / delta_time if delta_time > 0 else 0
                if ascend_start_frame is not None:
                    current_power = weight_kg * 9.8 * (ascend_start_y - current_shoulder_y) * scale_factor / ascend_time if ascend_time > 0 else 0
                    speed_factor = min(abs(velocity) * 10, 30)  # 速度贡献（最大 30%）
                    power_factor = min(current_power / 1000 * 20, 50)  # 功率贡献（最大 50%）
                    beast_probability = min(50 + speed_factor + power_factor, 100)
                else:
                    beast_probability = 50  # 未开始动作时保持基础值

            # 显示 Beast 概率（全程显示）
            beast_text = f"Beast Probability: {beast_probability:.1f}%"
            cv2.putText(frame, beast_text, (80, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

            # 从第 4 秒开始分析 muscle-up
            if frame_num >= start_frame and prev_shoulder_y is not None:
                delta_y = current_shoulder_y - prev_shoulder_y
                delta_time = current_time - prev_time
                velocity = (delta_y * scale_factor) / delta_time if delta_time > 0 else 0  # 速度 (m/s)

                # 开始上升：当 delta_y < -5 且速度 > 0.5 m/s
                if delta_y < -5 and abs(velocity) > 0.5 and ascend_start_frame is None:
                    ascend_start_frame = frame_num
                    ascend_start_y = current_shoulder_y
                    max_height_y = current_shoulder_y
                    velocities = [velocity]
                    ascend_time = 0  # 重置上升时间

                # 上升过程中
                elif ascend_start_frame is not None:
                    velocities.append(velocity)
                    if current_shoulder_y < max_height_y:
                        max_height_y = current_shoulder_y

                    # 仅当速度 > 0.1 m/s 时累计时间，过滤停顿
                    if abs(velocity) > 0.1:
                        ascend_time += delta_time

                    # 上升结束：当 delta_y > 5
                    if delta_y > 5:
                        pixel_displacement = ascend_start_y - max_height_y
                        real_displacement = pixel_displacement * scale_factor
                        max_velocity = max([abs(v) for v in velocities])

                        # 过滤无效动作：位移 >= 1.0 米，时间 0.5-2.0 秒
                        if real_displacement >= 1.0 and 0.5 <= ascend_time <= 2.0:
                            work = weight_kg * 9.8 * real_displacement
                            power = work / ascend_time
                            horsepower = power / 735
                            muscle_up_data.append({
                                'displacement': real_displacement,
                                'time': ascend_time,
                                'power': power,
                                'horsepower': horsepower,
                                'max_velocity': max_velocity
                            })

                            # 在画面上显示结果（红色、更大字体、更大行间距）
                            text_lines = [
                                f"Muscle Up {len(muscle_up_data)}:",
                                f"Displacement: {real_displacement:.2f} m",
                                f"Time: {ascend_time:.2f} s",
                                f"Power: {power:.2f} W ({horsepower:.2f} hp)",
                                f"Max Velocity: {max_velocity:.2f} m/s"
                            ]
                            y_pos = frame_height // 2 - 100  # 屏幕中间偏上
                            for line in text_lines:
                                cv2.putText(
                                    frame, 
                                    line, 
                                    (50, y_pos),  # 靠左 (x=50)
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    2.0,  # 字体大小 2.0
                                    (0, 0, 255),  # 红色
                                    4  # 线条粗细 4
                                )
                                y_pos += 70  # 增大行间距到 70

                            # 写入并暂停画面 2 秒
                            for _ in range(int(fps * 2)):
                                out.write(frame)

                        # 重置上升状态
                        ascend_start_frame = None
                        ascend_start_y = None
                        max_height_y = None
                        velocities = []
                        ascend_time = 0

            prev_shoulder_y = current_shoulder_y
            prev_time = current_time

        # 写入帧到输出视频
        out.write(frame)
        frame_num += 1

    # 释放资源
    video.release()
    out.release()
    pose.close()
    cv2.destroyAllWindows()

    return muscle_up_data

# 示例用法
if __name__ == "__main__":
    video_path = 'D:/vscode_files/funny_project/6_vertical_muscle_up.mp4'
    height_cm = 180  # 替换为你的身高（厘米）
    weight_kg = 70   # 替换为你的体重（千克）
    output_video_path = 'D:/vscode_files/funny_project/6_vertical_muscleups_analysis.mp4'

    try:
        muscle_up_results = process_muscle_up_video(video_path, height_cm, weight_kg, start_time=1.0, output_video_path=output_video_path)
        if not muscle_up_results:
            print("未检测到有效的 muscle-up 动作")
        for i, data in enumerate(muscle_up_results):
            print(f"Muscle-up {i+1}:")
            print(f"  位移: {data['displacement']:.2f} 米")
            print(f"  时间: {data['time']:.2f} 秒")
            print(f"  功率: {data['power']:.2f} 瓦")
            print(f"  马力: {data['horsepower']:.2f} 马力")
            print(f"  最大速度: {data['max_velocity']:.2f} 米/秒")
    except Exception as e:
        print(f"发生错误: {e}")


# # # ###字幕中文版本
# import cv2
# import mediapipe as mp
# import numpy as np
# import random
# from PIL import Image, ImageDraw, ImageFont

# # 初始化 MediaPipe Pose 模型和绘图工具
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils

# # 定义中文文本绘制函数（修正颜色通道）
# def draw_chinese_text(image, text, position, font_size, color_bgr):
#     # 将 OpenCV 图像转换为 PIL 图像
#     image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     draw = ImageDraw.Draw(image_pil)
    
#     # 加载支持中文的字体（Windows 示例，可根据系统调整路径）
#     try:
#         font = ImageFont.truetype("simhei.ttf", font_size)  # SimHei 是 Windows 常用中文字体
#     except:
#         font = ImageFont.load_default()
#         print("警告: 未找到 simhei.ttf，使用默认字体，中文可能仍显示乱码")

#     # 将 BGR 颜色转换为 RGB
#     color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])  # BGR -> RGB

#     # 绘制中文文本
#     draw.text(position, text, font=font, fill=color_rgb)
    
#     # 转换回 OpenCV 格式
#     image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
#     return image_cv

# def process_muscle_up_video(video_path, height_cm, weight_kg, start_time=4.0, output_video_path='output_video.mp4'):
#     """
#     处理 muscle-up 视频，检测关键点，计算上升阶段的位移、时间、功率、马力和最大速度，
#     并在画面上显示中文结果、目标检测框和野兽概率。
    
#     参数:
#         video_path (str): 输入视频文件路径
#         height_cm (float): 身高，单位：厘米
#         weight_kg (float): 体重，单位：千克
#         start_time (float): 开始分析的时间（秒），默认为 4.0 秒
#         output_video_path (str): 输出视频文件路径，默认为 'output_video.mp4'
    
#     返回:
#         list: 包含每个 muscle-up 的位移 (m)、时间 (s)、功率 (W)、马力 (hp) 和最大速度 (m/s) 的字典列表
#     """
#     # 打开视频
#     video = cv2.VideoCapture(video_path)
#     if not video.isOpened():
#         raise ValueError("无法打开视频文件，请检查路径是否正确")

#     # 获取视频参数
#     fps = video.get(cv2.CAP_PROP_FPS)
#     frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     start_frame = int(start_time * fps)

#     # 初始化 Pose 模型
#     pose = mp_pose.Pose(
#         static_image_mode=False, 
#         min_detection_confidence=0.5, 
#         min_tracking_confidence=0.5
#     )

#     # 设置输出视频
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

#     # 初始化变量
#     pixel_height = None
#     scale_factor = None
#     prev_shoulder_y = None
#     prev_time = None
#     muscle_up_data = []
#     ascend_start_frame = None
#     ascend_start_y = None
#     max_height_y = None
#     velocities = []
#     ascend_time = 0  # 累计有效上升时间
#     beast_probability = random.uniform(50, 90)  # 初始化野兽概率

#     frame_num = 0
#     while video.isOpened():
#         ret, frame = video.read()
#         if not ret:
#             break

#         current_time = frame_num / fps

#         # 转换为 RGB 格式
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(rgb_frame)

#         if results.pose_landmarks:
#             # 绘制关键点和连接线（红色）
#             mp_drawing.draw_landmarks(
#                 frame,
#                 results.pose_landmarks,
#                 mp_pose.POSE_CONNECTIONS,
#                 landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
#                 connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
#             )

#             # 获取关键点
#             landmarks = results.pose_landmarks.landmark
#             shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
#             current_shoulder_y = shoulder.y * frame_height

#             # 标定：使用身高计算像素到实际距离的比例
#             if pixel_height is None:
#                 head = landmarks[mp_pose.PoseLandmark.NOSE]
#                 left_foot = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
#                 right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
#                 foot_y = max(left_foot.y, right_foot.y) * frame_height
#                 pixel_height = abs(head.y * frame_height - foot_y)
#                 scale_factor = (height_cm / 100) / pixel_height

#             # 绘制目标检测框（蓝色）
#             x_min = min([lm.x * frame_width for lm in landmarks]) - 20
#             x_max = max([lm.x * frame_width for lm in landmarks]) + 20
#             y_min = min([lm.y * frame_height for lm in landmarks]) - 20
#             y_max = max([lm.y * frame_height for lm in landmarks]) + 20
#             cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)  # 蓝色框

#             # 计算野兽概率
#             if frame_num < start_frame:
#                 # 在 muscle-up 开始前随机波动
#                 beast_probability += random.uniform(-5, 5)
#                 beast_probability = max(50, min(90, beast_probability))
#             elif prev_shoulder_y is not None:
#                 # 在 muscle-up 过程中根据速度和功率计算
#                 delta_y = current_shoulder_y - prev_shoulder_y
#                 delta_time = current_time - prev_time
#                 velocity = (delta_y * scale_factor) / delta_time if delta_time > 0 else 0
#                 if ascend_start_frame is not None:
#                     current_power = weight_kg * 9.8 * (ascend_start_y - current_shoulder_y) * scale_factor / ascend_time if ascend_time > 0 else 0
#                     speed_factor = min(abs(velocity) * 10, 30)  # 速度贡献（最大 30%）
#                     power_factor = min(current_power / 1000 * 20, 50)  # 功率贡献（最大 50%）
#                     beast_probability = min(50 + speed_factor + power_factor, 100)
#                 else:
#                     beast_probability = 50  # 未开始动作时保持基础值

#             # 显示野兽概率（全程显示，中文）
#             beast_text = f"野兽概率: {beast_probability:.1f}%"
#             frame = draw_chinese_text(frame, beast_text, (50, 100), 40, (0, 255, 255))  # 黄色 (BGR)

#             # 从第 4 秒开始分析 muscle-up
#             if frame_num >= start_frame and prev_shoulder_y is not None:
#                 delta_y = current_shoulder_y - prev_shoulder_y
#                 delta_time = current_time - prev_time
#                 velocity = (delta_y * scale_factor) / delta_time if delta_time > 0 else 0  # 速度 (m/s)

#                 # 开始上升：当 delta_y < -5 且速度 > 0.5 m/s
#                 if delta_y < -5 and abs(velocity) > 0.5 and ascend_start_frame is None:
#                     ascend_start_frame = frame_num
#                     ascend_start_y = current_shoulder_y
#                     max_height_y = current_shoulder_y
#                     velocities = [velocity]
#                     ascend_time = 0  # 重置上升时间

#                 # 上升过程中
#                 elif ascend_start_frame is not None:
#                     velocities.append(velocity)
#                     if current_shoulder_y < max_height_y:
#                         max_height_y = current_shoulder_y

#                     # 仅当速度 > 0.1 m/s 时累计时间，过滤停顿
#                     if abs(velocity) > 0.1:
#                         ascend_time += delta_time

#                     # 上升结束：当 delta_y > 5
#                     if delta_y > 5:
#                         pixel_displacement = ascend_start_y - max_height_y
#                         real_displacement = pixel_displacement * scale_factor
#                         max_velocity = max([abs(v) for v in velocities])

#                         # 过滤无效动作：位移 >= 1.0 米，时间 0.5-2.0 秒
#                         if real_displacement >= 1.0 and 0.5 <= ascend_time <= 2.0:
#                             work = weight_kg * 9.8 * real_displacement
#                             power = work / ascend_time
#                             horsepower = power / 735
#                             muscle_up_data.append({
#                                 'displacement': real_displacement,
#                                 'time': ascend_time,
#                                 'power': power,
#                                 'horsepower': horsepower,
#                                 'max_velocity': max_velocity
#                             })

#                             # 在画面上显示结果（中文、红色、更大字体、更大行间距）
#                             text_lines = [
#                                 f"第 {len(muscle_up_data)} 次双力臂：",
#                                 f"位移: {real_displacement:.2f} 米",
#                                 f"时间: {ascend_time:.2f} 秒",
#                                 f"功率: {power:.2f} 瓦 ({horsepower:.2f} 马力)",
#                                 f"最大速度: {max_velocity:.2f} 米/秒"
#                             ]
#                             y_pos = frame_height // 2 - 100  # 屏幕中间偏上
#                             for line in text_lines:
#                                 frame = draw_chinese_text(frame, line, (50, y_pos), 50, (0, 0, 255))  # 红色 (BGR)
#                                 y_pos += 70  # 增大行间距到 70

#                             # 写入并暂停画面 2 秒
#                             for _ in range(int(fps * 2)):
#                                 out.write(frame)

#                         # 重置上升状态
#                         ascend_start_frame = None
#                         ascend_start_y = None
#                         max_height_y = None
#                         velocities = []
#                         ascend_time = 0

#             prev_shoulder_y = current_shoulder_y
#             prev_time = current_time

#         # 写入帧到输出视频
#         out.write(frame)
#         frame_num += 1

#     # 释放资源
#     video.release()
#     out.release()
#     pose.close()
#     cv2.destroyAllWindows()

#     return muscle_up_data

# # 示例用法
# if __name__ == "__main__":
#     video_path = 'D:/vscode_files/funny_project/7_explosive_muscleups.mp4'
#     height_cm = 180  # 替换为你的身高（厘米）
#     weight_kg = 70   # 替换为你的体重（千克）
#     output_video_path = 'D:/vscode_files/funny_project/7_explosive_muscleups_analysis_zh.mp4'

#     try:
#         muscle_up_results = process_muscle_up_video(video_path, height_cm, weight_kg, start_time=4.0, output_video_path=output_video_path)
#         if not muscle_up_results:
#             print("未检测到有效的 muscle-up 动作")
#         for i, data in enumerate(muscle_up_results):
#             print(f"第 {i+1} 次双力臂：")
#             print(f"  位移: {data['displacement']:.2f} 米")
#             print(f"  时间: {data['time']:.2f} 秒")
#             print(f"  功率: {data['power']:.2f} 瓦")
#             print(f"  马力: {data['horsepower']:.2f} 马力")
#             print(f"  最大速度: {data['max_velocity']:.2f} 米/秒")
#     except Exception as e:
#         print(f"发生错误: {e}")
