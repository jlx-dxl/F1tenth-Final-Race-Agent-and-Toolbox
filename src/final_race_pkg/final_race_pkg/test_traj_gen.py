#!/usr/bin/env python3

import sys
import numpy as np
import json
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
                             QLineEdit, QHBoxLayout, QGridLayout, QLabel, QSlider, QScrollArea)
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.interpolate import splprep, splev
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.points = []
        self.waypoints_file = 'waypoints.json'
        self.spline_order = 3  # Default spline order
        self.smoothing_factor = 0  # Default smoothing factor
        self.initUI()

    def initUI(self):
        self.setWindowTitle("3D Curve Projection Plotter")
        self.setGeometry(100, 100, 1400, 600)  # Adjusted size to accommodate color bar

        # Main layout and central widget
        main_layout = QGridLayout()
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        central_widget.setLayout(main_layout)

        # Spline order and smoothing factor sliders
        controls_layout = QHBoxLayout()
        self.order_slider, self.order_label = self.create_slider("Spline Order", self.spline_order, self.update_order, 1, 5, 1)
        self.smoothing_slider, self.smoothing_label = self.create_slider("Smoothing Factor", self.smoothing_factor, self.update_smoothing, 0, 6, 1)
        controls_layout.addWidget(self.order_label)
        controls_layout.addWidget(self.order_slider)
        controls_layout.addWidget(self.smoothing_label)
        controls_layout.addWidget(self.smoothing_slider)
        main_layout.addLayout(controls_layout, 0, 0, 1, 1)

        # Scrollable area for points
        scroll_widget = QWidget()
        self.points_layout = QVBoxLayout(scroll_widget)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area, 1, 0, 1, 1)

        # Buttons layout
        buttons_layout = QHBoxLayout()
        main_layout.addLayout(buttons_layout, 2, 0, 1, 1)

        # Add point button
        add_point_button = QPushButton('+ Add Point', self)
        add_point_button.clicked.connect(self.add_point)
        buttons_layout.addWidget(add_point_button)

        # Delete point button
        delete_point_button = QPushButton('- Delete Point', self)
        delete_point_button.clicked.connect(self.delete_point)
        buttons_layout.addWidget(delete_point_button)

        # Plot button
        plot_button = QPushButton('Plot', self)
        plot_button.clicked.connect(self.plot_curve)
        buttons_layout.addWidget(plot_button)

        # Save button
        save_button = QPushButton('Save', self)
        save_button.clicked.connect(self.save_data)
        buttons_layout.addWidget(save_button)

        # Figure and canvas
        self.figure = Figure(figsize=(10, 5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)  # 2D projection
        main_layout.addWidget(self.canvas, 0, 1, 3, 2)  # Adjust grid position

        self.load_waypoints()

    def create_slider(self, label_text, initial_value, callback, min_value, max_value, tick_interval):
        label = QLabel(f"{label_text}: {initial_value}")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_value)
        slider.setMaximum(max_value)
        slider.setValue(initial_value)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(tick_interval)
        slider.valueChanged.connect(lambda: callback(slider, label))
        slider.valueChanged.connect(self.plot_curve)  # Add this line to replot when slider values change
        return slider, label

    def update_order(self, slider, label):
        self.spline_order = slider.value()
        label.setText(f"Spline Order: {self.spline_order}")

    def update_smoothing(self, slider, label):
        self.smoothing_factor = slider.value()
        label.setText(f"Smoothing Factor: {self.smoothing_factor}")

    def add_point(self, x=None, y=None, z=None, w=1.0):
        point_layout = QHBoxLayout()

        # Label for point index
        index_label = QLabel(f"{len(self.points) + 1}")
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
        w_slider.valueChanged.connect(self.plot_curve)  # Automatically replot when the weight slider changes

        point_layout.addWidget(x_input)
        point_layout.addWidget(y_input)
        point_layout.addWidget(z_input)
        point_layout.addWidget(w_slider)
        point_layout.addWidget(w_label)
        self.points_layout.addLayout(point_layout)
        self.points.append((x_input, y_input, z_input, w_slider))

    def delete_point(self):
        if self.points:
            to_delete = self.points.pop()
            for widget in to_delete:
                widget.setParent(None)
                widget.deleteLater()
            layout = self.points_layout.takeAt(self.points_layout.count() - 1)
            if layout:
                layout.deleteLater()

    def load_waypoints(self):
        if os.path.exists(self.waypoints_file):
            with open(self.waypoints_file, 'r') as f:
                data = json.load(f)
                self.spline_order = data.get('spline_order', 3)
                self.smoothing_factor = data.get('smoothing_factor', 0)
                self.order_slider.setValue(self.spline_order)
                self.smoothing_slider.setValue(self.smoothing_factor)

                points_data = data.get('points', [])
                for point in points_data:
                    w = float(point['w']) if 'w' in point else 1.0
                    self.add_point(str(point['x']), str(point['y']), str(point['z']), w)

                # Ensure that there are enough points after loading
                if len(self.points) < self.spline_order + 1:
                    additional_points_needed = self.spline_order + 1 - len(self.points)
                    for _ in range(additional_points_needed):
                        self.add_point()

    def plot_curve(self):
        point_data = []
        weights = []
        for i, (x_input, y_input, z_input, w_slider) in enumerate(self.points):
            try:
                x = float(x_input.text())
                y = float(y_input.text())
                z = float(z_input.text())
                w = w_slider.value() / 10.0
                point_data.append((x, y, z))
                weights.append(w)
            except ValueError:
                continue

        if len(point_data) <= self.spline_order:
            self.ax.clear()
            if hasattr(self, 'cbar'):
                self.cbar.remove()
            self.ax.figure.canvas.draw()
            print("Please input at least", self.spline_order + 1, "valid points.")
            return

        points = np.array(point_data)
        tck, u = splprep(points.T, s=self.smoothing_factor, k=self.spline_order, per=True, w=weights)
        new_points = splev(np.linspace(0, 1, 100), tck)
        self.ax.clear()

        # Remove existing colorbar if exists
        if hasattr(self, 'cbar'):
            self.cbar.remove()

        scatter = self.ax.scatter(new_points[0], new_points[1], c=new_points[2], cmap='viridis', label='Projected Curve')
        self.ax.set_xlabel('X coordinate')
        self.ax.set_ylabel('Y coordinate')

        # Adding a color bar to indicate z-values
        self.cbar = self.figure.colorbar(scatter, ax=self.ax, orientation='vertical')
        self.cbar.set_label('Z coordinate')

        # Add waypoint indices to the plot
        for i, (x, y, z) in enumerate(point_data):
            self.ax.text(x, y, f'{i+1}', color="red", fontsize=12, ha='right', va='top')
            self.ax.plot(x, y, marker='*', color='blue', markersize=10)  # Use star markers for waypoints
            
        self.ax.figure.canvas.draw()

    def save_data(self):
        point_data = []
        json_data = {
            'spline_order': self.spline_order,
            'smoothing_factor': self.smoothing_factor,
            'points': []
        }
        for x_input, y_input, z_input, w_slider in self.points:
            try:
                x = float(x_input.text())
                y = float(y_input.text())
                z = float(z_input.text())
                w = w_slider.value() / 10.0
                point_data.append((x, y, z))
                json_data['points'].append({'x': x, 'y': y, 'z': z, 'w': w})
            except ValueError:
                continue

        if len(point_data) <= self.spline_order:
            print("Please input at least", self.spline_order + 1, "valid points.")
            return

        with open(self.waypoints_file, 'w') as f:
            json.dump(json_data, f, indent=4)
        print(f"Data saved to '{self.waypoints_file}'.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = Window()
    ex.show()
    sys.exit(app.exec_())
