"""Curly Hair Generator Extension for Inkscape"""

# Deps: inkex
# DO NOT adding pubic hair or underarm hair to models or other people's illustrations
# and uploading them to social media.

import sys
import math
import random

try:
    import inkex
except ImportError as e:
    sys.exit(1)

from inkex import Group, PathElement

class CurlyHairExtension(inkex.EffectExtension):
    """Extension to generate curly hair, whiskers, and body hair"""

    # Constants
    MIN_SEGMENTS_PER_CURL = 6
    MIN_CURL_FREQUENCY = 0.1
    MIN_THICKNESS = 0.01
    DEFAULT_BEZIER_CONTROL_SCALE = 0.3
    CENTER_LINE_CONTROL_SCALE = 0.5
    MAX_STROKE_SEGMENTS = 8

    # Parameters
    def add_arguments(self, pars):
        """Parameter definitions"""
        pars.add_argument("--strand_count", type=int, default=10)
        
        pars.add_argument("--length_min", type=float, default=50.0)
        pars.add_argument("--length_max", type=float, default=150.0)
        
        pars.add_argument("--thickness_start", type=float, default=3.0)
        pars.add_argument("--thickness_end", type=float, default=0.2)
        pars.add_argument("--taper_position", type=float, default=0.3)
        pars.add_argument("--taper_curve", default="ease_out")
        
        pars.add_argument("--curl_strength", type=float, default=1.0)
        pars.add_argument("--curl_frequency", type=float, default=2.0)
        pars.add_argument("--angle_variation", type=float, default=45.0)
        pars.add_argument("--curl_randomness", type=float, default=0.5)
        
        pars.add_argument("--base_direction", type=float, default=270.0)
        pars.add_argument("--direction_spread", type=float, default=30.0)
        
        pars.add_argument("--start_area_width", type=float, default=100.0)
        pars.add_argument("--start_area_height", type=float, default=20.0)
        pars.add_argument("--center_x", type=float, default=200.0)
        pars.add_argument("--center_y", type=float, default=200.0)
        
        pars.add_argument("--segments_per_curl", type=int, default=8) 
        pars.add_argument("--curl_decay", type=float, default=0.2)
        pars.add_argument("--spiral_tendency", type=float, default=0.5)
        
        pars.add_argument("--smooth_curves", type=inkex.Boolean, default=True)
        pars.add_argument("--smoothing_factor", type=float, default=0.5)
        pars.add_argument("--path_type", default="bezier")
        pars.add_argument("--truncate_thin_tip", type=inkex.Boolean, default=False)
        pars.add_argument("--min_thickness_cutoff", type=float, default=0.5)

        pars.add_argument("--use_shape_source", type=inkex.Boolean, default=False)
        pars.add_argument("--shape_type", default="auto")
        pars.add_argument("--shape_direction", default="outward")
        pars.add_argument("--shape_distribution", default="uniform")
        pars.add_argument("--shape_scale", type=float, default=100.0)
        pars.add_argument("--shape_offset", type=float, default=0.0)
        pars.add_argument("--shape_start_angle", type=float, default=0.0)
        pars.add_argument("--shape_end_angle", type=float, default=360.0)

        pars.add_argument("--hair_type", default="custom")

        pars.add_argument("--render_mode", default="stroke")
        pars.add_argument("--notebook", default="basic")
    
    # Main processing
    def effect(self):
        """Main processing method"""
        try:
            current_layer = self.svg.get_current_layer()

            self.apply_hair_preset()

            main_group = Group()
            main_group.label = "Curly Hair"

            if self.options.use_shape_source and self.svg.selection:
                for elem in self.svg.selection:
                    strands = self.create_hair_from_shape(elem)
                    for strand in strands:
                        if strand is not None:
                            main_group.add(strand)
            else:
                for i in range(self.options.strand_count):
                    strand = self.create_hair_strand(i)
                    if strand is not None:
                        main_group.add(strand)

            current_layer.add(main_group)

        except Exception as e:
            self.msg(f"Error: {str(e)}")
    
    def apply_hair_preset(self):
        """Apply preset parameters for different hair types"""
        presets = {
            "loose_curls": {
                "curl_strength": 0.8,
                "curl_frequency": 1.5,
                "angle_variation": 30.0,
                "segments_per_curl": 10,
                "curl_decay": 0.1
            },
            "tight_curls": {
                "curl_strength": 1.5,
                "curl_frequency": 4.0,
                "angle_variation": 60.0,
                "segments_per_curl": 8,
                "curl_decay": 0.3
            },
            "wavy": {
                "curl_strength": 0.5,
                "curl_frequency": 1.0,
                "angle_variation": 20.0,
                "segments_per_curl": 12,
                "curl_decay": 0.05
            },
            "kinky": {
                "curl_strength": 1.8,
                "curl_frequency": 5.0,
                "angle_variation": 70.0,
                "segments_per_curl": 6,
                "curl_decay": 0.3,
                "thickness_end": 0.05
            },
            "mustache": {
                "curl_strength": 1.2,
                "curl_frequency": 3.0,
                "angle_variation": 40.0,
                "thickness_start": 2.0,
                "thickness_end": 0.1,
                "length_min": 20.0,
                "length_max": 60.0,
                "segments_per_curl": 8
            },
            "eyelashes": {
                "curl_strength": 0.3,
                "curl_frequency": 0.5,
                "angle_variation": 15.0,
                "thickness_start": 1.5,
                "thickness_end": 0.02,
                "length_min": 15.0,
                "length_max": 25.0,
                "segments_per_curl": 6
            },
            "fur": {
                "curl_strength": 0.7,
                "curl_frequency": 2.5,
                "angle_variation": 35.0,
                "thickness_start": 1.0,
                "thickness_end": 0.05,
                "curl_randomness": 0.8,
                "segments_per_curl": 8
            }
        }
        
        if self.options.hair_type in presets:
            preset = presets[self.options.hair_type]
            for key, value in preset.items():
                if hasattr(self.options, key):
                    setattr(self.options, key, value)
    
    def create_hair_strand(self, _strand_index):
        """Create a single hair strand in the rectangular area"""
        start_x = (self.options.center_x - self.options.start_area_width/2 +
                  random.uniform(0, self.options.start_area_width))
        start_y = (self.options.center_y - self.options.start_area_height/2 +
                  random.uniform(0, self.options.start_area_height))

        base_angle = math.radians(self.options.base_direction +
                                 random.uniform(-self.options.direction_spread/2,
                                              self.options.direction_spread/2))

        return self.create_hair_strand_at(start_x, start_y, base_angle)
    
    def create_multi_segment_stroke(self, points, thicknesses):
        """Create multiple stroke segments with variable width"""
        if len(points) < 2:
            return None
        
        group = Group()
        group.label = "Variable Stroke Hair"
        
        num_segments = min(len(points) - 1, self.MAX_STROKE_SEGMENTS)
        segment_length = len(points) // num_segments
        
        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = min((i + 1) * segment_length + 1, len(points))
            
            if end_idx <= start_idx + 1:
                continue
            
            segment_points = points[start_idx:end_idx]
            
            thickness_start_idx = start_idx
            thickness_end_idx = min(end_idx - 1, len(thicknesses) - 1)
            
            if thickness_start_idx < len(thicknesses) and thickness_end_idx < len(thicknesses):
                avg_thickness = (thicknesses[thickness_start_idx] + thicknesses[thickness_end_idx]) / 2
            else:
                avg_thickness = self.options.thickness_start
            
            path_data = self.create_center_line_path(segment_points)
            
            path_element = PathElement()
            path_element.set('d', path_data)
            path_element.style = {
                'fill': 'none',
                'stroke': '#000000',
                'stroke-width': str(max(self.MIN_THICKNESS, avg_thickness)),
                'stroke-linecap': 'round',
                'stroke-linejoin': 'round',
                'opacity': '1.0'
            }
            
            group.add(path_element)
        
        return group
    
    def generate_strand_path(self, start_x, start_y, base_angle, total_length):
        """Generate the path points and thicknesses for a hair strand"""
        points = [(start_x, start_y)]
        thicknesses = [self.options.thickness_start]

        segments_per_curl = max(self.MIN_SEGMENTS_PER_CURL, self.options.segments_per_curl)
        curl_frequency = max(self.MIN_CURL_FREQUENCY, self.options.curl_frequency)
        segment_length = total_length / (segments_per_curl * curl_frequency)
        num_segments = int(total_length / segment_length) if segment_length > 0 else 1
        
        current_x, current_y = start_x, start_y
        current_angle = base_angle
        
        for i in range(1, num_segments + 1):
            progress = i / num_segments
            
            curl_phase = progress * self.options.curl_frequency * 2 * math.pi
            curl_amplitude = self.options.curl_strength * (1 - self.options.curl_decay * progress)
            
            spiral_offset = self.options.spiral_tendency * progress * math.pi
            
            curl_deviation = (curl_amplitude * math.sin(curl_phase + spiral_offset) * 
                            math.radians(self.options.angle_variation))
            
            random_deviation = (random.uniform(-1, 1) * self.options.curl_randomness * 
                              math.radians(self.options.angle_variation * 0.5))
            
            current_angle += curl_deviation + random_deviation
            
            current_x += segment_length * math.cos(current_angle)
            current_y += segment_length * math.sin(current_angle)
            
            points.append((current_x, current_y))
            
            thickness = self.calculate_thickness(progress)
            thicknesses.append(thickness)
        
        return points, thicknesses
    
    def calculate_thickness(self, progress):
        """Calculate thickness at given progress (0=start, 1=end)"""
        start_thick = float(self.options.thickness_start)
        end_thick = float(self.options.thickness_end)
        taper_start = float(self.options.taper_position)

        if progress <= taper_start:
            return start_thick

        if taper_start >= 1.0:
            return start_thick

        taper_progress = (progress - taper_start) / (1.0 - taper_start)
        taper_progress = max(0.0, min(1.0, taper_progress))

        if self.options.taper_curve == "linear":
            t = taper_progress
        elif self.options.taper_curve == "ease_out":
            t = taper_progress ** 0.5
        elif self.options.taper_curve == "ease_in":
            t = taper_progress ** 2
        elif self.options.taper_curve == "sigmoid":
            x = (taper_progress - 0.5) * 6
            t = 1 / (1 + math.exp(-x))
        else:
            t = taper_progress

        return start_thick + (end_thick - start_thick) * t
    
    def create_tapered_path(self, points, thicknesses):
        """Create SVG path with variable thickness"""
        if len(points) < 2:
            return ""

        truncate = self.options.truncate_thin_tip
        min_thickness = float(self.options.min_thickness_cutoff)

        left_points = []
        right_points = []

        if truncate:
            cutoff_idx = len(points) - 1
            for i in range(len(points)):
                if thicknesses[i] < min_thickness:
                    cutoff_idx = i
                    break
            cutoff_idx = max(cutoff_idx, min(3, len(points) - 1))
        else:
            cutoff_idx = len(points) - 1

        for i in range(cutoff_idx + 1):
            x, y = points[i]
            thickness = thicknesses[i] / 2

            if i == 0:
                if len(points) > 1:
                    dx = points[1][0] - points[0][0]
                    dy = points[1][1] - points[0][1]
                else:
                    dx, dy = 1, 0
            elif i == cutoff_idx:
                dx = points[i][0] - points[i-1][0]
                dy = points[i][1] - points[i-1][1]
            else:
                dx1 = points[i][0] - points[i-1][0]
                dy1 = points[i][1] - points[i-1][1]
                dx2 = points[i+1][0] - points[i][0]
                dy2 = points[i+1][1] - points[i][1]
                dx = (dx1 + dx2) / 2
                dy = (dy1 + dy2) / 2

            length = math.sqrt(dx*dx + dy*dy)

            effective_thickness = max(thickness, self.MIN_THICKNESS)

            if length > 0:
                perp_x = -dy / length * effective_thickness
                perp_y = dx / length * effective_thickness
                left_points.append((x + perp_x, y + perp_y))
                right_points.append((x - perp_x, y - perp_y))
            else:
                left_points.append((x, y))
                right_points.append((x, y))

        if truncate:
            last_left = left_points[-1]
            last_right = right_points[-1]
            tip_point = ((last_left[0] + last_right[0]) / 2,
                         (last_left[1] + last_right[1]) / 2)
        else:
            tip_point = points[-1]

        return self.create_improved_tapered_path(left_points, right_points, tip_point, truncate)

    def create_improved_tapered_path(self, left_points, right_points, tip_point, truncate=False):
        """Create tapered path from left/right edge points"""
        if not left_points:
            return ""

        use_bezier = (self.options.path_type == "bezier" and
                      self.options.smooth_curves and
                      len(left_points) > 3)

        right_reversed = list(reversed(right_points))

        if use_bezier:
            return self._create_bezier_tapered_path(left_points, right_reversed, tip_point, truncate)
        else:
            return self._create_polyline_tapered_path(left_points, right_reversed, tip_point, truncate)

    def _create_polyline_tapered_path(self, left_points, right_points_reversed, tip_point, truncate=False):
        """Create tapered path using straight lines"""
        path_parts = []

        path_parts.append(f"M {left_points[0][0]:.2f},{left_points[0][1]:.2f}")

        for point in left_points[1:]:
            path_parts.append(f"L {point[0]:.2f},{point[1]:.2f}")

        if truncate:
            pass
        else:
            tip_x, tip_y = tip_point
            path_parts.append(f"L {tip_x:.2f},{tip_y:.2f}")

        for rx, ry in right_points_reversed:
            path_parts.append(f"L {rx:.2f},{ry:.2f}")

        path_parts.append("Z")
        return " ".join(path_parts)

    def _create_bezier_tapered_path(self, left_points, right_points_reversed, tip_point, truncate=False):
        """Create tapered path using smooth Bezier curves"""
        path_parts = []
        smoothing = self.options.smoothing_factor

        path_parts.append(f"M {left_points[0][0]:.2f},{left_points[0][1]:.2f}")

        path_parts.append(self._points_to_bezier(left_points, smoothing))

        if not truncate:
            tip_x, tip_y = tip_point
            path_parts.append(f"L {tip_x:.2f},{tip_y:.2f}")

        if len(right_points_reversed) > 1:
            path_parts.append(self._points_to_bezier(right_points_reversed, smoothing, skip_move=True))
        elif right_points_reversed:
            rx, ry = right_points_reversed[0]
            path_parts.append(f"L {rx:.2f},{ry:.2f}")

        path_parts.append("Z")
        return " ".join(path_parts)

    def _points_to_bezier(self, points, smoothing, control_scale=None, skip_move=False):
        """Convert points to Bezier curve commands"""
        if control_scale is None:
            control_scale = self.DEFAULT_BEZIER_CONTROL_SCALE
        if len(points) < 2:
            return ""

        parts = []
        if not skip_move:
            parts.append(f"M {points[0][0]:.2f},{points[0][1]:.2f}")

        for i in range(len(points) - 1):
            current = points[i]
            next_point = points[i + 1]

            if i == 0:
                prev_point = current
            else:
                prev_point = points[i - 1]

            if i + 2 < len(points):
                after_next = points[i + 2]
            else:
                after_next = next_point

            cp1_x = current[0] + (next_point[0] - prev_point[0]) * smoothing * control_scale
            cp1_y = current[1] + (next_point[1] - prev_point[1]) * smoothing * control_scale
            cp2_x = next_point[0] - (after_next[0] - current[0]) * smoothing * control_scale
            cp2_y = next_point[1] - (after_next[1] - current[1]) * smoothing * control_scale

            parts.append(f"C {cp1_x:.2f},{cp1_y:.2f} {cp2_x:.2f},{cp2_y:.2f} {next_point[0]:.2f},{next_point[1]:.2f}")

        return " ".join(parts)

    def create_center_line_path(self, points):
        """Create center line path"""
        if len(points) < 2:
            return ""

        if self.options.smooth_curves and len(points) > 3:
            return self._points_to_bezier(points, self.options.smoothing_factor,
                                          control_scale=self.CENTER_LINE_CONTROL_SCALE)
        else:
            path_parts = [f"M {points[0][0]:.2f},{points[0][1]:.2f}"]
            for point in points[1:]:
                path_parts.append(f"L {point[0]:.2f},{point[1]:.2f}")
            return " ".join(path_parts)

    def create_hair_from_shape(self, element):
        """Generate hair strands from a shape element"""
        strands = []
        center = self.get_shape_center(element)
        if center is None:
            return strands

        cx, cy = center
        shape_points = self.get_shape_points(element)
        if not shape_points:
            return strands

        scale = self.options.shape_scale / 100.0

        for x, y, normal_angle in shape_points:
            scaled_x = cx + (x - cx) * scale
            scaled_y = cy + (y - cy) * scale
            offset = self.options.shape_offset
            final_x = scaled_x + offset * math.cos(normal_angle)
            final_y = scaled_y + offset * math.sin(normal_angle)
            direction = self.calculate_hair_direction(normal_angle)

            strand = self.create_hair_strand_at(final_x, final_y, direction)
            if strand is not None:
                strands.append(strand)

        return strands

    def get_shape_center(self, element):
        """Get the center point of a shape element"""
        tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag

        if tag == 'circle':
            cx = float(element.get('cx', 0))
            cy = float(element.get('cy', 0))
            return (cx, cy)
        elif tag == 'ellipse':
            cx = float(element.get('cx', 0))
            cy = float(element.get('cy', 0))
            return (cx, cy)
        elif tag == 'rect':
            x = float(element.get('x', 0))
            y = float(element.get('y', 0))
            w = float(element.get('width', 100))
            h = float(element.get('height', 100))
            return (x + w / 2, y + h / 2)
        elif tag == 'polygon':
            points_str = element.get('points', '')
            if points_str:
                coords = []
                for pair in points_str.strip().split():
                    if ',' in pair:
                        px, py = pair.split(',')
                        coords.append((float(px), float(py)))
                if coords:
                    cx = sum(p[0] for p in coords) / len(coords)
                    cy = sum(p[1] for p in coords) / len(coords)
                    return (cx, cy)
        else:
            try:
                bbox = element.bounding_box() if hasattr(element, 'bounding_box') else None
                if bbox:
                    return ((bbox.left + bbox.right) / 2, (bbox.top + bbox.bottom) / 2)
            except Exception:
                pass

        return (0, 0)

    def get_shape_points(self, element):
        """Extract points and normals from a shape element"""
        tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag
        shape_type = self.options.shape_type

        if shape_type == "auto":
            if tag in ('circle', 'ellipse'):
                shape_type = "ellipse"
            elif tag == 'rect':
                shape_type = "rect"
            elif tag in ('polygon', 'star'):
                shape_type = "star"
            else:
                shape_type = "path"

        if shape_type == "ellipse" or tag in ('circle', 'ellipse'):
            return self.get_ellipse_points(element)
        elif shape_type == "rect" or tag == 'rect':
            return self.get_rect_points(element)
        elif shape_type == "star" or tag == 'polygon':
            return self.get_polygon_points(element)
        else:
            return self.get_path_points(element)

    def get_ellipse_points(self, element):
        """Get points around an ellipse/circle"""
        points = []
        tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag

        if tag == 'circle':
            cx = float(element.get('cx', 0))
            cy = float(element.get('cy', 0))
            rx = ry = float(element.get('r', 50))
        else:
            cx = float(element.get('cx', 0))
            cy = float(element.get('cy', 0))
            rx = float(element.get('rx', 50))
            ry = float(element.get('ry', 50))

        start_angle = math.radians(self.options.shape_start_angle)
        end_angle = math.radians(self.options.shape_end_angle)
        if end_angle <= start_angle:
            end_angle += 2 * math.pi

        count = self.options.strand_count
        distribution = self.options.shape_distribution

        if distribution == "uniform":
            angles = [start_angle + (end_angle - start_angle) * i / count
                      for i in range(count)]
        elif distribution == "random":
            angles = [start_angle + random.uniform(0, 1) * (end_angle - start_angle)
                      for _ in range(count)]
            angles.sort()
        else:
            angles = [start_angle + (end_angle - start_angle) * i / count
                      for i in range(count)]

        for angle in angles:
            x = cx + rx * math.cos(angle)
            y = cy + ry * math.sin(angle)
            points.append((x, y, angle))

        return points

    def get_rect_points(self, element):
        """Get points around a rectangle"""
        x = float(element.get('x', 0))
        y = float(element.get('y', 0))
        w = float(element.get('width', 100))
        h = float(element.get('height', 100))

        cx, cy = x + w / 2, y + h / 2
        count = self.options.strand_count
        distribution = self.options.shape_distribution

        if distribution == "corners":
            corners = [
                (x, y, math.atan2(y - cy, x - cx)),
                (x + w, y, math.atan2(y - cy, x + w - cx)),
                (x + w, y + h, math.atan2(y + h - cy, x + w - cx)),
                (x, y + h, math.atan2(y + h - cy, x - cx)),
            ]
            return corners[:count]

        perimeter = 2 * (w + h)
        positions = []

        if distribution == "uniform":
            positions = [perimeter * i / count for i in range(count)]
        elif distribution == "random":
            positions = sorted([random.uniform(0, perimeter) for _ in range(count)])
        elif distribution == "edges":
            per_edge = max(1, count // 4)
            for edge in range(4):
                edge_len = w if edge % 2 == 0 else h
                for i in range(per_edge):
                    t = edge_len * i / per_edge
                    if edge == 0:
                        positions.append(t)
                    elif edge == 1:
                        positions.append(w + t)
                    elif edge == 2:
                        positions.append(w + h + t)
                    else:
                        positions.append(w + h + w + t)
        else:
            positions = [perimeter * i / count for i in range(count)]

        points = []
        for pos in positions:
            if pos < w:
                px, py = x + pos, y
                normal = -math.pi / 2
            elif pos < w + h:
                px, py = x + w, y + (pos - w)
                normal = 0
            elif pos < 2 * w + h:
                px, py = x + w - (pos - w - h), y + h
                normal = math.pi / 2
            else:
                px, py = x, y + h - (pos - 2 * w - h)
                normal = math.pi
            points.append((px, py, normal))

        return points

    def get_polygon_points(self, element):
        """Get points from a polygon/star"""
        points_str = element.get('points', '')
        if not points_str:
            return []

        coords = []
        for pair in points_str.strip().split():
            if ',' in pair:
                x, y = pair.split(',')
                coords.append((float(x), float(y)))

        if len(coords) < 2:
            return []

        cx = sum(p[0] for p in coords) / len(coords)
        cy = sum(p[1] for p in coords) / len(coords)

        count = self.options.strand_count
        distribution = self.options.shape_distribution

        if distribution == "corners":
            result = []
            for px, py in coords[:count]:
                normal = math.atan2(py - cy, px - cx)
                result.append((px, py, normal))
            return result

        result = []
        total_len = 0
        edge_lengths = []
        for i in range(len(coords)):
            p1 = coords[i]
            p2 = coords[(i + 1) % len(coords)]
            edge_len = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            edge_lengths.append(edge_len)
            total_len += edge_len

        if distribution == "uniform":
            positions = [total_len * i / count for i in range(count)]
        elif distribution == "random":
            positions = sorted([random.uniform(0, total_len) for _ in range(count)])
        else:
            positions = [total_len * i / count for i in range(count)]

        for pos in positions:
            cumulative = 0
            for i, edge_len in enumerate(edge_lengths):
                if cumulative + edge_len >= pos:
                    t = (pos - cumulative) / edge_len if edge_len > 0 else 0
                    p1 = coords[i]
                    p2 = coords[(i + 1) % len(coords)]
                    px = p1[0] + t * (p2[0] - p1[0])
                    py = p1[1] + t * (p2[1] - p1[1])
                    edge_angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                    normal = edge_angle - math.pi / 2
                    test_x = px + math.cos(normal)
                    test_y = py + math.sin(normal)
                    if (test_x - cx)**2 + (test_y - cy)**2 < (px - cx)**2 + (py - cy)**2:
                        normal += math.pi
                    result.append((px, py, normal))
                    break
                cumulative += edge_len

        return result

    def get_path_points(self, element):
        """Get points from a generic path"""
        try:
            from inkex import Path
            path = element.path if hasattr(element, 'path') else Path(element.get('d', ''))
            path = path.to_absolute()

            points = []
            count = self.options.strand_count

            bbox = element.bounding_box() if hasattr(element, 'bounding_box') else None
            if bbox:
                cx, cy = (bbox.left + bbox.right) / 2, (bbox.top + bbox.bottom) / 2
            else:
                cx, cy = 0, 0

            current_x, current_y = 0, 0
            path_points = []

            for cmd in path:
                if cmd.letter.upper() == 'M':
                    current_x, current_y = cmd.args[0], cmd.args[1]
                    path_points.append((current_x, current_y))
                elif cmd.letter.upper() == 'L':
                    current_x, current_y = cmd.args[0], cmd.args[1]
                    path_points.append((current_x, current_y))
                elif cmd.letter.upper() == 'C':
                    current_x, current_y = cmd.args[4], cmd.args[5]
                    path_points.append((current_x, current_y))
                elif cmd.letter.upper() == 'Z':
                    pass

            if len(path_points) < 2:
                return []

            step = max(1, len(path_points) // count)
            for i in range(0, len(path_points), step):
                if len(points) >= count:
                    break
                px, py = path_points[i]
                normal = math.atan2(py - cy, px - cx)
                points.append((px, py, normal))

            return points

        except Exception:
            return []

    def calculate_hair_direction(self, normal_angle):
        """Calculate hair direction based on normal and direction option"""
        direction = self.options.shape_direction
        spread = math.radians(self.options.direction_spread)
        random_offset = random.uniform(-spread / 2, spread / 2)

        if direction == "outward":
            return normal_angle + random_offset
        elif direction == "inward":
            return normal_angle + math.pi + random_offset
        elif direction == "clockwise":
            return normal_angle + math.pi / 2 + random_offset
        elif direction == "counter_clockwise":
            return normal_angle - math.pi / 2 + random_offset
        else:
            return normal_angle + random_offset

    def create_hair_strand_at(self, start_x, start_y, base_angle):
        """Create a hair strand at a specific position"""
        strand_length = random.uniform(self.options.length_min, self.options.length_max)

        points, thicknesses = self.generate_strand_path(start_x, start_y, base_angle, strand_length)

        if self.options.render_mode == "filled":
            path_data = self.create_tapered_path(points, thicknesses)
            fill_color = '#000000'
            stroke_color = 'none'
            stroke_width = '0'
        elif self.options.render_mode == "stroke":
            return self.create_multi_segment_stroke(points, thicknesses)
        else:
            path_data = self.create_center_line_path(points)
            fill_color = 'none'
            stroke_color = '#000000'
            avg_thickness = (self.options.thickness_start + self.options.thickness_end) / 2
            stroke_width = str(avg_thickness)

        path_element = PathElement()
        path_element.set('d', path_data)

        path_element.style = {
            'fill': fill_color,
            'stroke': stroke_color,
            'stroke-width': stroke_width,
            'stroke-linecap': 'round',
            'stroke-linejoin': 'round',
            'opacity': '1.0'
        }

        return path_element

# Main execution
if __name__ == '__main__':
    try:
        if len(sys.argv) == 1:
            test_svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400" viewBox="0 0 400 400">
    <g id="layer1"></g>
</svg>'''
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as f:
                f.write(test_svg)
                sys.argv.append(f.name)
        
        extension = CurlyHairExtension()
        extension.run()
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)