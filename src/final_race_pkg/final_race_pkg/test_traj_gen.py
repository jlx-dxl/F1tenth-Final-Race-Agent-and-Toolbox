#!/usr/bin/env python3

import sys
import numpy as np
import json
import os
import csv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
                             QLineEdit, QHBoxLayout, QGridLayout, QLabel, QSlider, 
                             QScrollArea)
from PyQt5.QtCore import Qt, QSize
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.interpolate import splprep, splev
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.curves = [
            {"points": [], "spline_order": 3, "smoothing_factor": 0, "waypoints_file": "waypoints1.json", "curve_file": "curve1.csv"},
            {"points": [], "spline_order": 3, "smoothing_factor": 0, "waypoints_file": "waypoints2.json", "curve_file": "curve2.csv"}
        ]
        self.initUI()

    def initUI(self):
        self.setWindowTitle("3D Curve Projection Plotter")
        self.setGeometry(100, 100, 1400, 800)

        # Main layout and central widget
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        central_widget.setLayout(main_layout)

        # Split left area into two parts for two curves
        curve1_area = QGridLayout()
        curve2_area = QGridLayout()
        left_layout.addLayout(curve1_area)
        left_layout.addLayout(curve2_area)

        # Create input areas for both curves
        for i, area in enumerate([curve1_area, curve2_area]):
            self.create_curve_input_area(area, i)

        # Plot area on the right
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)  # 2D projection
        main_layout.addLayout(left_layout)
        main_layout.addWidget(self.canvas)

    def create_curve_input_area(self, layout, curve_index):
        # Spline order and smoothing factor sliders
        controls_layout = QHBoxLayout()
        order_slider, order_label = self.create_slider(f"Spline Order", self.curves[curve_index]["spline_order"], lambda s, l: self.update_order(s, l, curve_index), 1, 5, 1)
        smoothing_slider, smoothing_label = self.create_slider(f"Smoothing Factor", self.curves[curve_index]["smoothing_factor"], lambda s, l: self.update_smoothing(s, l, curve_index), 0, 6, 1)
        controls_layout.addWidget(order_label)
        controls_layout.addWidget(order_slider)
        controls_layout.addWidget(smoothing_label)
        controls_layout.addWidget(smoothing_slider)
        layout.addLayout(controls_layout, 0, 0, 1, 1)
        
        # 将滑块和标签的引用存储到curves字典中
        self.curves[curve_index]["spline_order_slider"] = order_slider
        self.curves[curve_index]["spline_order_label"] = order_label
        self.curves[curve_index]["smoothing_slider"] = smoothing_slider
        self.curves[curve_index]["smoothing_label"] = smoothing_label

        # Scrollable area for points
        scroll_widget = QWidget()
        points_layout = QVBoxLayout(scroll_widget)
        self.curves[curve_index]["layout"] = points_layout

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area, 1, 0, 1, 1)

        # Buttons layout
        buttons_layout = QHBoxLayout()
        layout.addLayout(buttons_layout, 2, 0, 1, 1)

        # Add point button
        add_point_button = QPushButton(f'+ Add Point', self)
        add_point_button.clicked.connect(lambda: self.add_point(curve_index))
        buttons_layout.addWidget(add_point_button)

        # Delete point button
        delete_point_button = QPushButton(f'- Delete Point', self)
        delete_point_button.clicked.connect(lambda: self.delete_point(curve_index))
        buttons_layout.addWidget(delete_point_button)

        # Plot button
        plot_button = QPushButton('Plot All Curves', self)
        plot_button.clicked.connect(self.plot_curve)
        buttons_layout.addWidget(plot_button)

        # Save button
        save_button = QPushButton(f'Save Curve {curve_index + 1}', self)
        save_button.clicked.connect(lambda: self.save_data(curve_index))
        buttons_layout.addWidget(save_button)
        
        # Load initial data if exists
        self.load_waypoints(curve_index)
        

    def create_slider(self, label_text, initial_value, callback, min_value, max_value, tick_interval):
        label = QLabel(f"{label_text}: {initial_value}")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_value)
        slider.setMaximum(max_value)
        slider.setValue(initial_value)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(tick_interval)
        slider.valueChanged.connect(lambda: callback(slider, label))
        slider.valueChanged.connect(self.plot_curve)  # Trigger replot when slider values change
        return slider, label

    def update_order(self, slider, label, curve_index):
        self.curves[curve_index]["spline_order"] = slider.value()
        label.setText(f"Spline Order: {self.curves[curve_index]['spline_order']}")
        self.plot_curve()

    def update_smoothing(self, slider, label, curve_index):
        self.curves[curve_index]["smoothing_factor"] = slider.value()
        label.setText(f"Smoothing Factor: {self.curves[curve_index]['smoothing_factor']}")
        self.plot_curve()

    def add_point(self, curve_index, x=None, y=None, z=None, w=1.0):
        points_layout = self.curves[curve_index]["layout"]
        point_layout = QHBoxLayout()

        # Label for point index
        index_label = QLabel(f"{len(self.curves[curve_index]['points']) + 1}")
        point_layout.addWidget(index_label)

        # Inputs for x, y, and z
        x_input = QLineEdit(x)
        y_input = QLineEdit(y)
        z_input = QLineEdit(z)

        # Convert None or non-string values to empty strings
        x = "" if x is None or not isinstance(x, str) else x
        y = "" if y is None or not isinstance(y, str) else y
        z = "" if z is None or not isinstance(z, str) else z

        # Add buttons for adjusting coordinate values
        def adjust_value(line_edit, delta):
            try:
                current_value = float(line_edit.text())
                new_value = current_value + delta
                line_edit.setText(f"{new_value:.1f}")
                self.plot_curve()  # Replot after value adjustment
            except ValueError:
                line_edit.setText(f"{delta:.1f}")
                self.plot_curve()  # Replot after value adjustment

        up_x_button = QPushButton('▲')
        down_x_button = QPushButton('▼')
        up_y_button = QPushButton('▲')
        down_y_button = QPushButton('▼')
        up_z_button = QPushButton('▲')
        down_z_button = QPushButton('▼')

        up_x_button.setFixedSize(QSize(20, 20))
        down_x_button.setFixedSize(QSize(20, 20))
        up_y_button.setFixedSize(QSize(20, 20))
        down_y_button.setFixedSize(QSize(20, 20))
        up_z_button.setFixedSize(QSize(20, 20))
        down_z_button.setFixedSize(QSize(20, 20))

        # Set up connections for buttons
        up_x_button.clicked.connect(lambda: adjust_value(x_input, 0.1))
        down_x_button.clicked.connect(lambda: adjust_value(x_input, -0.1))
        up_y_button.clicked.connect(lambda: adjust_value(y_input, 0.1))
        down_y_button.clicked.connect(lambda: adjust_value(y_input, -0.1))
        up_z_button.clicked.connect(lambda: adjust_value(z_input, 0.1))
        down_z_button.clicked.connect(lambda: adjust_value(z_input, -0.1))

        # Slider for adjusting weight 'w'
        w_slider = QSlider(Qt.Horizontal)
        w_slider.setMinimum(1)  # Minimum weight to avoid zero
        w_slider.setMaximum(10)  # Maximum corresponds to weight 1.0
        w_slider.setValue(int(w * 10))  # Convert weight to slider position
        w_slider.setTickPosition(QSlider.TicksBelow)
        w_slider.setTickInterval(1)

        w_label = QLabel(f"{w:.1f}")
        w_slider.valueChanged.connect(lambda: w_label.setText(f"{w_slider.value() / 10.0:.1f}"))
        w_slider.valueChanged.connect(self.plot_curve)  # Trigger replot when weight slider changes

        # Add widgets to layout
        point_layout.addWidget(x_input)
        point_layout.addWidget(up_x_button)
        point_layout.addWidget(down_x_button)
        point_layout.addWidget(y_input)
        point_layout.addWidget(up_y_button)
        point_layout.addWidget(down_y_button)
        point_layout.addWidget(z_input)
        point_layout.addWidget(up_z_button)
        point_layout.addWidget(down_z_button)
        point_layout.addWidget(w_slider)
        point_layout.addWidget(w_label)
        points_layout.addLayout(point_layout)

        # Append the inputs and slider as a tuple to the points list
        self.curves[curve_index]['points'].append((x_input, y_input, z_input, w_slider))

    def delete_point(self, curve_index):
        points = self.curves[curve_index]['points']
        if points:
            to_delete = points.pop()
            for widget in to_delete:
                widget.setParent(None)
                widget.deleteLater()
            layout = self.curves[curve_index]["layout"]
            layout_item = layout.takeAt(layout.count() - 1)
            if layout_item:
                layout_item.deleteLater()
            self.plot_curve()  # Replot after deleting a point

    def load_waypoints(self, curve_index):
        waypoints_file = self.curves[curve_index]['waypoints_file']
        if os.path.exists(waypoints_file):
            with open(waypoints_file, 'r') as f:
                data = json.load(f)

                # 获取样条阶数和平滑因子
                spline_order = data.get('spline_order', 3)
                smoothing_factor = data.get('smoothing_factor', 0)

                # 更新滑块值并手动更新标签
                self.curves[curve_index]["spline_order_slider"].setValue(spline_order)
                self.curves[curve_index]["smoothing_slider"].setValue(smoothing_factor)
                self.curves[curve_index]["spline_order_label"].setText(f"Spline Order: {spline_order}")
                self.curves[curve_index]["smoothing_label"].setText(f"Smoothing Factor: {smoothing_factor}")

                # 强制重绘滑块和标签
                self.curves[curve_index]["spline_order_slider"].repaint()
                self.curves[curve_index]["smoothing_slider"].repaint()
                self.curves[curve_index]["spline_order_label"].update()
                self.curves[curve_index]["smoothing_label"].update()
            
                # 处理点数据
                points_data = data.get('points', [])
                for point in points_data:
                    w = float(point['w']) if 'w' in point else 1.0
                    self.add_point(curve_index, str(point['x']), str(point['y']), str(point['z']), w)

    def save_data(self, curve_index):
        points = self.curves[curve_index]['points']
        waypoints_file = self.curves[curve_index]['waypoints_file']
        curve_file = self.curves[curve_index]['curve_file']
        point_data = []
        json_data = {
            'spline_order': self.curves[curve_index]['spline_order'],
            'smoothing_factor': self.curves[curve_index]['smoothing_factor'],
            'points': []
        }
        # Prepare for CSV
        with open(curve_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['X', 'Y', 'Z', 'Weight', 'Yaw'])

            for x_input, y_input, z_input, w_slider in points:
                try:
                    x = float(x_input.text())
                    y = float(y_input.text())
                    z = float(z_input.text())
                    w = w_slider.value() / 10.0
                    point_data.append((x, y, z, w))
                    json_data['points'].append({'x': x, 'y': y, 'z': z, 'w': w})
                except ValueError:
                    continue

            if len(point_data) <= self.curves[curve_index]['spline_order']:
                print(f"Curve {curve_index + 1}: Please input at least {self.curves[curve_index]['spline_order'] + 1} valid points.")
            else:
                # Compute spline and 200 interpolated points
                points_np = np.array(point_data)
                tck, u = splprep(points_np[:, :3].T, s=self.curves[curve_index]['smoothing_factor'], k=self.curves[curve_index]['spline_order'], w=points_np[:, 3])
                new_points = splev(np.linspace(0, 1, 200), tck)

                # Calculate yaw angles
                x_vals = new_points[0]
                y_vals = new_points[1]
                yaw_angles = np.arctan2(np.diff(y_vals, prepend=y_vals[0]), np.diff(x_vals, prepend=x_vals[0]))

                for i in range(200):
                    x, y, z, w = new_points[0][i], new_points[1][i], new_points[2][i], points_np[i % len(points), 3]
                    yaw = yaw_angles[i]
                    writer.writerow([x, y, z, w, yaw])

        with open(waypoints_file, 'w') as f:
            json.dump(json_data, f, indent=4)
        print(f"Data for curve {curve_index + 1} saved to '{waypoints_file}' and '{curve_file}'.")

    def plot_curve(self):
        if not hasattr(self, 'ax'):  # Check if 'ax' is already initialized
            return  # Optionally, raise an exception or create a logger warning

        self.ax.clear()
        if hasattr(self, 'colorbars'):
            for cbar in self.colorbars:
                cbar.remove()
        self.colorbars = []

        color_maps = ['cool', 'autumn']  # Updated color maps for higher contrast
        scatters = []  # To keep scatter objects for colorbars
        markers = ['^', '*']  # Triangle and star markers
        colors = ['blue', 'red']  # Colors for the different curves' waypoints

        for curve_index in range(2):
            points = self.curves[curve_index]['points']
            if not points:
                continue
            point_data = []
            weights = []
            for i, (x_input, y_input, z_input, w_slider) in enumerate(points):
                try:
                    x = float(x_input.text())
                    y = float(y_input.text())
                    z = float(z_input.text())
                    w = w_slider.value() / 10.0
                    point_data.append((x, y, z))
                    weights.append(w)

                    # Mark each waypoint with its index number
                    self.ax.scatter(x, y, color=colors[curve_index], marker=markers[curve_index], s=100)  # s is the size of the marker
                    self.ax.text(x, y, f'{i+1}', color=colors[curve_index], fontsize=12, ha='right')
                except ValueError:
                    continue

            if len(point_data) <= self.curves[curve_index]['spline_order']:
                print(f"Curve {curve_index + 1}: Please input at least", self.curves[curve_index]['spline_order'] + 1, "valid points.")
                continue

            points = np.array(point_data)
            tck, u = splprep(points.T, s=self.curves[curve_index]['smoothing_factor'], k=self.curves[curve_index]['spline_order'], per=True, w=weights)
            new_points = splev(np.linspace(0, 1, 200), tck)
            
            scatter = self.ax.scatter(new_points[0], new_points[1], c=new_points[2], cmap=color_maps[curve_index], label=f'Curve {curve_index + 1}')
            scatters.append(scatter)

        # Handling colorbars
        for scatter, cmap in zip(scatters, color_maps):
            colorbar = self.figure.colorbar(scatter, ax=self.ax, orientation='vertical', pad=0.1, fraction=0.02)
            colorbar.set_label(f'Color scale for {cmap}')
            self.colorbars.append(colorbar)  # Keep track of colorbars
            
        # Define polygons' vertices
        vertices1 = [(10.4, -1.5), (10.2, 4.1), (2.7, 4.0), (2.7, 10.1), (-4.1, 9.9), (-4.1, -1.5)]
        vertices2 = [(5.4, 1.4), (-0.8, 1.3), (-1.0, 6.6), (-1.8, 6.5), (-1.6, 0.6), (5.3, 0.7)]

        # Create Polygon objects
        polygon1 = Polygon(vertices1, closed=True, edgecolor='black', fill=None, linewidth=3.0)
        polygon2 = Polygon(vertices2, closed=True, edgecolor='black', fill=True, facecolor='black')
        
        # Add polygons to the plot
        self.ax.add_patch(polygon1)
        self.ax.add_patch(polygon2)

        self.ax.legend()
        self.ax.set_xlabel('X coordinate')
        self.ax.set_ylabel('Y coordinate')
        self.ax.figure.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = Window()
    ex.show()
    sys.exit(app.exec_())
