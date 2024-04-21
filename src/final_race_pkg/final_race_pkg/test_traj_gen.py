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
import pandas as pd

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.points = []
        self.waypoints_file = 'waypoints.json'
        self.spline_order = 3  # Default spline order
        self.smoothing_factor = 0  # Default smoothing factor
        self.initUI()
        self.load_waypoints()

    def initUI(self):
        self.setWindowTitle("3D Curve Plotter")
        self.setGeometry(100, 100, 1200, 600)

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
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)  # Create an axis instance
        main_layout.addWidget(self.canvas, 0, 1, 3, 1)  # Span all vertical and horizontal space for canvas

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

    def update_order(self, slider, label):
        self.spline_order = slider.value()
        label.setText(f"Spline Order: {self.spline_order}")
        self.ensure_minimum_points()

    def update_smoothing(self, slider, label):
        self.smoothing_factor = slider.value()
        label.setText(f"Smoothing Factor: {self.smoothing_factor}")

    def add_point(self, x=None, y=None, z=None, w=0.5):
        point_layout = QHBoxLayout()
        
        # Convert None or non-string values to empty strings
        x = "" if x is None or not isinstance(x, str) else x
        y = "" if y is None or not isinstance(y, str) else y
        z = "" if z is None or not isinstance(z, str) else z
        
        x_input = QLineEdit(x)
        y_input = QLineEdit(y)
        z_input = QLineEdit(z)
        
        w_slider = QSlider(Qt.Horizontal)
        w_slider.setMinimum(0)
        w_slider.setMaximum(100)
        w_slider.setValue(int(float(w) * 100))  # Ensure w is a float, scale it to slider's range
        w_slider.setTickPosition(QSlider.TicksBelow)
        w_slider.setTickInterval(1)
        
        # Display the current value of the slider for w
        w_label = QLabel(f"{float(w):.2f}")
        w_slider.valueChanged.connect(lambda: w_label.setText(f"{w_slider.value() / 100.0:.2f}"))

        point_layout.addWidget(x_input)
        point_layout.addWidget(y_input)
        point_layout.addWidget(z_input)
        point_layout.addWidget(w_slider)
        point_layout.addWidget(w_label)
        self.points_layout.addLayout(point_layout)
        self.points.append((x_input, y_input, z_input, w_slider))

    def delete_point(self):
        if len(self.points) > self.spline_order + 1:
            to_delete = self.points.pop()
            for widget in to_delete:
                widget.setParent(None)
                widget.deleteLater()
            layout = self.points_layout.takeAt(self.points_layout.count() - 1)
            if layout:
                layout.deleteLater()

    def ensure_minimum_points(self):
        required_points = self.spline_order + 1
        while len(self.points) < required_points:
            self.add_point()

    def load_waypoints(self):
        if os.path.exists(self.waypoints_file):
            with open(self.waypoints_file, 'r') as f:
                data = json.load(f)
                existing_point_count = len(data)
                for point in data:
                    w = float(point['w']) if 'w' in point else 0.5
                    self.add_point(str(point['x']), str(point['y']), str(point['z']), w)
                additional_points_needed = self.spline_order + 1 - existing_point_count
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
                w = w_slider.value() / 100.0
                point_data.append((x, y, z))
                weights.append(w)
                # Annotate waypoint with its index
                self.ax.text(x, y, f'{i+1}', color="red", fontsize=12, ha='right')
            except ValueError:
                continue

        if len(point_data) <= self.spline_order:
            self.ax.clear()
            self.ax.figure.canvas.draw()
            print("Please input at least", self.spline_order + 1, "valid points.")
            return

        points = np.array(point_data)
        tck, u = splprep(points.T, s=self.smoothing_factor, k=self.spline_order, per=True, w=weights)
        new_points = splev(np.linspace(0, 1, 100), tck)
        self.ax.clear()
        self.ax.scatter(new_points[0], new_points[1], c=new_points[2], cmap='viridis')
        self.ax.set_xlabel('X coordinate')
        self.ax.set_ylabel('Y coordinate')
        self.ax.figure.canvas.draw()

    def save_data(self):
        point_data = []
        json_data = []
        for x_input, y_input, z_input, w_slider in self.points:
            try:
                x = float(x_input.text())
                y = float(y_input.text())
                z = float(z_input.text())
                w = w_slider.value() / 100.0
                point_data.append((x, y, z))
                json_data.append({'x': x, 'y': y, 'z': z, 'w': w})
            except ValueError:
                continue

        if len(point_data) <= self.spline_order:
            print("Please input at least", self.spline_order + 1, "valid points.")
            return

        points = np.array(point_data)
        weights = [d['w'] for d in json_data]
        tck, u = splprep(points.T, s=self.smoothing_factor, k=self.spline_order, per=False, w=weights)
        new_points = splev(np.linspace(0, 1, 100), tck)
        df = pd.DataFrame(np.column_stack(new_points), columns=['X', 'Y', 'Z'])
        df.to_csv('curve_data.csv', index=False)
        print("Data saved to 'curve_data.csv'.")

        # Save waypoints
        with open(self.waypoints_file, 'w') as f:
            json.dump(json_data, f, indent=4)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = Window()
    ex.show()
    sys.exit(app.exec_())
