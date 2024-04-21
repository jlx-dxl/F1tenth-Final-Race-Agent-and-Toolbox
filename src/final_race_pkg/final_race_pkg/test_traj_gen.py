#!/usr/bin/env python3

import sys
import numpy as np
import json
import os
import csv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
                             QLineEdit, QHBoxLayout, QGridLayout, QLabel, QSlider, QScrollArea)
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.interpolate import splprep, splev
from matplotlib.colors import Normalize

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
        order_slider, order_label = self.create_slider(f"Curve {curve_index + 1} Spline Order", self.curves[curve_index]["spline_order"], lambda s, l: self.update_order(s, l, curve_index), 1, 5, 1)
        smoothing_slider, smoothing_label = self.create_slider(f"Curve {curve_index + 1} Smoothing Factor", self.curves[curve_index]["smoothing_factor"], lambda s, l: self.update_smoothing(s, l, curve_index), 0, 6, 1)
        controls_layout.addWidget(order_label)
        controls_layout.addWidget(order_slider)
        controls_layout.addWidget(smoothing_label)
        controls_layout.addWidget(smoothing_slider)
        layout.addLayout(controls_layout, 0, 0, 1, 1)

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
        add_point_button = QPushButton(f'+ Add Point to Curve {curve_index + 1}', self)
        add_point_button.clicked.connect(lambda: self.add_point(curve_index))
        buttons_layout.addWidget(add_point_button)

        # Delete point button
        delete_point_button = QPushButton(f'- Delete Point from Curve {curve_index + 1}', self)
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
        return slider, label

    def update_order(self, slider, label, curve_index):
        self.curves[curve_index]["spline_order"] = slider.value()
        label.setText(f"Spline Order: {self.curves[curve_index]['spline_order']}")

    def update_smoothing(self, slider, label, curve_index):
        self.curves[curve_index]["smoothing_factor"] = slider.value()
        label.setText(f"Smoothing Factor: {self.curves[curve_index]['smoothing_factor']}")

    def add_point(self, curve_index, x=None, y=None, z=None, w=1.0):
        points_layout = self.curves[curve_index]["layout"]
        point_layout = QHBoxLayout()

        # Label for point index
        index_label = QLabel(f"{len(self.curves[curve_index]['points']) + 1}")
        point_layout.addWidget(index_label)

        # Convert None or non-string values to empty strings
        x = "" if x is None or not isinstance(x, str) else x
        y = "" if y is None or not isinstance(y, str) else y
        z = "" if z is None or not isinstance(z, str) else z

        x_input = QLineEdit(x)
        y_input = QLineEdit(y)
        z_input = QLineEdit(z)
        w_slider = QSlider(Qt.Horizontal)
        w_slider.setMinimum(1)  # Set minimum to avoid zero
        w_slider.setMaximum(10)  # Adjust for 0.1 increments from 0.1 to 1.0
        w_slider.setValue(int(float(w) * 10))  # Scale w to the slider's range
        w_slider.setTickPosition(QSlider.TicksBelow)
        w_slider.setTickInterval(1)

        # Display the current value of the slider for w
        w_label = QLabel(f"{float(w):.1f}")
        w_slider.valueChanged.connect(lambda: w_label.setText(f"{w_slider.value() / 10.0:.1f}"))

        point_layout.addWidget(x_input)
        point_layout.addWidget(y_input)
        point_layout.addWidget(z_input)
        point_layout.addWidget(w_slider)
        point_layout.addWidget(w_label)
        points_layout.addLayout(point_layout)
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

    def load_waypoints(self, curve_index):
        waypoints_file = self.curves[curve_index]['waypoints_file']
        if os.path.exists(waypoints_file):
            with open(waypoints_file, 'r') as f:
                data = json.load(f)
                self.curves[curve_index]['spline_order'] = data.get('spline_order', 3)
                self.curves[curve_index]['smoothing_factor'] = data.get('smoothing_factor', 0)

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
        with open(curve_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['X', 'Y', 'Z', 'Weight'])

            for x_input, y_input, z_input, w_slider in points:
                try:
                    x = float(x_input.text())
                    y = float(y_input.text())
                    z = float(z_input.text())
                    w = w_slider.value() / 10.0
                    point_data.append((x, y, z))
                    json_data['points'].append({'x': x, 'y': y, 'z': z, 'w': w})
                    writer.writerow([x, y, z, w])
                except ValueError:
                    continue

        with open(waypoints_file, 'w') as f:
            json.dump(json_data, f, indent=4)
        print(f"Data for curve {curve_index + 1} saved to '{waypoints_file}' and '{curve_file}'.")

    def plot_curve(self):
        self.ax.clear()
        colors = ['viridis', 'plasma']
        for curve_index in range(2):
            points = self.curves[curve_index]['points']
            if not points:
                continue
            point_data = []
            weights = []
            for x_input, y_input, z_input, w_slider in points:
                try:
                    x = float(x_input.text())
                    y = float(y_input.text())
                    z = float(z_input.text())
                    w = w_slider.value() / 10.0
                    point_data.append((x, y, z))
                    weights.append(w)
                except ValueError:
                    continue

            if len(point_data) <= self.curves[curve_index]['spline_order']:
                print(f"Curve {curve_index + 1}: Please input at least", self.curves[curve_index]['spline_order'] + 1, "valid points.")
                continue

            points = np.array(point_data)
            tck, u = splprep(points.T, s=self.curves[curve_index]['smoothing_factor'], k=self.curves[curve_index]['spline_order'], per=True, w=weights)
            new_points = splev(np.linspace(0, 1, 100), tck)
            self.ax.scatter(new_points[0], new_points[1], c=new_points[2], cmap=colors[curve_index], label=f'Curve {curve_index + 1}')

        self.ax.legend()
        self.ax.set_xlabel('X coordinate')
        self.ax.set_ylabel('Y coordinate')
        self.ax.figure.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = Window()
    ex.show()
    sys.exit(app.exec_())
