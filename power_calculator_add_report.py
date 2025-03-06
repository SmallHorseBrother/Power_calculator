import cv2
import mediapipe as mp
import numpy as np
import random
import matplotlib.pyplot as plt
from jinja2 import Template
import os

# 初始化 MediaPipe Pose 模型和绘图工具
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def process_muscle_up_video(video_path, height_cm, weight_kg, start_time=1.0, output_video_path='output_video.mp4', report_path='report.html'):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError("无法打开视频文件，请检查路径是否正确")

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = int(start_time * fps)

    pose = mp_pose.Pose(
        static_image_mode=False, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    pixel_height = None
    scale_factor = None
    prev_shoulder_y = None
    prev_time = None
    muscle_up_data = []
    ascend_start_frame = None
    ascend_start_y = None
    max_height_y = None
    velocities = []
    ascend_time = 0
    beast_probability = random.uniform(50, 90)
    text_to_display = None
    display_frames_left = 0

    frame_num = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        current_time = frame_num / fps
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            landmarks = results.pose_landmarks.landmark
            shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            current_shoulder_y = shoulder.y * frame_height

            if pixel_height is None:
                head = landmarks[mp_pose.PoseLandmark.NOSE]
                left_foot = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
                right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
                foot_y = max(left_foot.y, right_foot.y) * frame_height
                pixel_height = abs(head.y * frame_height - foot_y)
                scale_factor = (height_cm / 100) / pixel_height * 1.05

            x_min = min([lm.x * frame_width for lm in landmarks]) - 20
            x_max = max([lm.x * frame_width for lm in landmarks]) + 20
            y_min = min([lm.y * frame_height for lm in landmarks]) - 20
            y_max = max([lm.y * frame_height for lm in landmarks]) + 20
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 255), 2)

            if frame_num < start_frame:
                beast_probability += random.uniform(-5, 5)
                beast_probability = max(50, min(90, beast_probability))
            elif prev_shoulder_y is not None:
                delta_y = current_shoulder_y - prev_shoulder_y
                delta_time = current_time - prev_time
                velocity = (delta_y * scale_factor) / delta_time if delta_time > 0 else 0
                if ascend_start_frame is not None:
                    current_power = weight_kg * 9.8 * (ascend_start_y - current_shoulder_y) * scale_factor / ascend_time if ascend_time > 0 else 0
                    speed_factor = min(abs(velocity) * 10, 30)
                    power_factor = min(current_power / 1000 * 20, 50)
                    beast_probability = min(50 + speed_factor + power_factor, 100)
                else:
                    beast_probability = 50

            beast_text = f"Beast Probability: {beast_probability:.1f}%"
            cv2.putText(frame, beast_text, (80, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

            if frame_num >= start_frame and prev_shoulder_y is not None:
                delta_y = current_shoulder_y - prev_shoulder_y
                delta_time = current_time - prev_time
                velocity = (delta_y * scale_factor) / delta_time if delta_time > 0 else 0

                if delta_y < -5 and abs(velocity) > 0.2 and ascend_start_frame is None:
                    ascend_start_frame = frame_num
                    ascend_start_y = current_shoulder_y
                    max_height_y = current_shoulder_y
                    velocities = [velocity]
                    ascend_time = 0

                elif ascend_start_frame is not None:
                    velocities.append(velocity)
                    if current_shoulder_y < max_height_y:
                        max_height_y = current_shoulder_y

                    if abs(velocity) > 0.1:
                        ascend_time += delta_time

                    if delta_y > 5:
                        pixel_displacement = ascend_start_y - max_height_y
                        real_displacement = pixel_displacement * scale_factor
                        max_velocity = max([abs(v) for v in velocities])

                        if real_displacement >= 0.1 and 0.2 <= ascend_time <= 5:
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

                            text_lines = [
                                f"Pull-up {len(muscle_up_data)}:",
                                f"Displacement: {real_displacement:.2f} m",
                                f"Time: {ascend_time:.2f} s",
                                f"Power: {power:.2f} W ({horsepower:.2f} hp)",
                                f"Max Velocity: {max_velocity:.2f} m/s"
                            ]
                            display_frames_left = int(fps * 1)
                            text_to_display = text_lines

                        ascend_start_frame = None
                        ascend_start_y = None
                        max_height_y = None
                        velocities = []
                        ascend_time = 0

            prev_shoulder_y = current_shoulder_y
            prev_time = current_time

        if display_frames_left > 0 and text_to_display:
            y_pos = frame_height // 2 - 100
            for line in text_to_display:
                cv2.putText(
                    frame, 
                    line, 
                    (50, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    2.0,
                    (255, 255, 0),
                    4
                )
                y_pos += 70
            display_frames_left -= 1
        else:
            text_to_display = None

        out.write(frame)
        frame_num += 1

    video.release()
    out.release()
    pose.close()
    cv2.destroyAllWindows()

    generate_html_report(muscle_up_data, report_path, height_cm, weight_kg)
    return muscle_up_data

def generate_html_report(data, report_path, height_cm, weight_kg):
    if not data:
        summary = {
            'total_pull_ups': 0,
            'avg_displacement': 0,
            'avg_time': 0,
            'avg_power': 0,
            'avg_horsepower': 0,
            'avg_max_velocity': 0
        }
        chart_paths = {}
    else:
        summary = {
            'total_pull_ups': len(data),
            'avg_displacement': np.mean([d['displacement'] for d in data]),
            'avg_time': np.mean([d['time'] for d in data]),
            'avg_power': np.mean([d['power'] for d in data]),
            'avg_horsepower': np.mean([d['horsepower'] for d in data]),
            'avg_max_velocity': np.mean([d['max_velocity'] for d in data])
        }

        pull_up_numbers = range(1, len(data) + 1)
        
        plt.figure(figsize=(8, 4))
        plt.plot(pull_up_numbers, [d['displacement'] for d in data], marker='o')
        plt.xlabel('Pull-up Number')
        plt.ylabel('Displacement (m)')
        plt.title('Displacement per Pull-up')
        plt.grid()
        displacement_chart = 'displacement_chart.png'
        plt.savefig(displacement_chart)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(pull_up_numbers, [d['time'] for d in data], marker='o', color='green')
        plt.xlabel('Pull-up Number')
        plt.ylabel('Time (s)')
        plt.title('Time per Pull-up')
        plt.grid()
        time_chart = 'time_chart.png'
        plt.savefig(time_chart)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(pull_up_numbers, [d['power'] for d in data], marker='o', color='red')
        plt.xlabel('Pull-up Number')
        plt.ylabel('Power (W)')
        plt.title('Power per Pull-up')
        plt.grid()
        power_chart = 'power_chart.png'
        plt.savefig(power_chart)
        plt.close()

        chart_paths = {
            'displacement': displacement_chart,
            'time': time_chart,
            'power': power_chart
        }

    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pull-up Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 80%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
            th { background-color: #f2f2f2; }
            img { max-width: 100%; height: auto; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>Pull-up Analysis Report</h1>
        <h2>Summary / 总结</h2>
        <p>Height / 身高: {{ height_cm }} cm</p>
        <p>Weight / 体重: {{ weight_kg }} kg</p>
        <p>Total Pull-ups / 总引体向上次数: {{ summary.total_pull_ups }}</p>
        <p>Average Displacement / 平均位移: {{ "%.2f"|format(summary.avg_displacement) }} m</p>
        <p>Average Time / 平均时间: {{ "%.2f"|format(summary.avg_time) }} s</p>
        <p>Average Power / 平均功率: {{ "%.2f"|format(summary.avg_power) }} W</p>
        <p>Average Horsepower / 平均马力: {{ "%.2f"|format(summary.avg_horsepower) }} hp</p>
        <p>Average Max Velocity / 平均最大速度: {{ "%.2f"|format(summary.avg_max_velocity) }} m/s</p>

        <h2>Detailed Data / 详细数据</h2>
        <table>
            <tr>
                <th>Pull-up #</th>
                <th>Displacement (m)</th>
                <th>Time (s)</th>
                <th>Power (W)</th>
                <th>Horsepower (hp)</th>
                <th>Max Velocity (m/s)</th>
            </tr>
            {% for item in data %}
            <tr>
                <td>{{ loop.index }}</td>
                <td>{{ "%.2f"|format(item.displacement) }}</td>
                <td>{{ "%.2f"|format(item.time) }}</td>
                <td>{{ "%.2f"|format(item.power) }}</td>
                <td>{{ "%.2f"|format(item.horsepower) }}</td>
                <td>{{ "%.2f"|format(item.max_velocity) }}</td>
            </tr>
            {% endfor %}
        </table>

        {% if chart_paths %}
        <h2>Visualization / 数据可视化</h2>
        <h3>Displacement / 位移</h3>
        <img src="{{ chart_paths.displacement }}" alt="Displacement Chart">
        <h3>Time / 时间</h3>
        <img src="{{ chart_paths.time }}" alt="Time Chart">
        <h3>Power / 功率</h3>
        <img src="{{ chart_paths.power }}" alt="Power Chart">
        {% endif %}
    </body>
    </html>
    """

    template = Template(html_template)
    html_content = template.render(
        data=data,
        summary=summary,
        height_cm=height_cm,
        weight_kg=weight_kg,
        chart_paths=chart_paths if data else {}
    )

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == "__main__":
    video_path = 'D:/vscode_files/funny_project/28_gold_pullups.mp4'
    height_cm = 180
    weight_kg = 70
    output_video_path = 'D:/vscode_files/funny_project/28_gold_pullups_analysis.mp4'
    report_path = 'D:/vscode_files/funny_project/pull_up_report.html'

    try:
        muscle_up_results = process_muscle_up_video(video_path, height_cm, weight_kg, start_time=0, output_video_path=output_video_path, report_path=report_path)
        if not muscle_up_results:
            print("未检测到有效的引体向上动作")
        for i, data in enumerate(muscle_up_results):
            print(f"Pull-up {i+1}:")
            print(f"  位移: {data['displacement']:.2f} 米")
            print(f"  时间: {data['time']:.2f} 秒")
            print(f"  功率: {data['power']:.2f} 瓦")
            print(f"  马力: {data['horsepower']:.2f} 马力")
            print(f"  最大速度: {data['max_velocity']:.2f} 米/秒")
    except Exception as e:
        print(f"发生错误: {e}")

