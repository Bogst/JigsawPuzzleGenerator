import itertools
import json
import math
from enum import Enum, auto
from string import ascii_lowercase

from PIL import Image, ImageOps, ImageDraw, ImageFont

import random

from exceptions import BackgroundImageNotBigEnough


class BorderOrientation(Enum):
    horizontal = auto()
    vertical = auto()


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def make_bezier(xys):
    # xys should be a sequence of 2-tuples (Bezier control points)
    n = len(xys)
    combinations = pascal_row(n-1)
    def bezier(ts):
        # This uses the generalized formula for bezier curves
        # http://en.wikipedia.org/wiki/B%C3%A9zier_curve#Generalization
        result = []
        for t in ts:
            tpowers = (t**i for i in range(n))
            upowers = reversed([(1-t)**i for i in range(n)])
            coefs = [c*a*b for c, a, b in zip(combinations, tpowers, upowers)]
            result.append(
                tuple(sum([coef*p for coef, p in zip(coefs, ps)]) for ps in zip(*xys)))
        return result
    return bezier


def pascal_row(n, memo={}):
    # This returns the nth row of Pascal's Triangle
    if n in memo:
        return memo[n]
    result = [1]
    x, numerator = 1, n
    for denominator in range(1, n//2+1):
        x *= numerator
        x /= denominator
        result.append(x)
        numerator -= 1
    if n&1 == 0:
        # n is even
        result.extend(reversed(result[:-1]))
    else:
        result.extend(reversed(result))
    memo[n] = result
    return result


def get_control_points(cell_width, cell_height, cell_col, cell_row, start_point, end_point=None, orientation: BorderOrientation = BorderOrientation.horizontal):
    control_points = [start_point]

    edge_deviation_border = 1/3  # must always be sub-unitary (the edge can't go closer than 1 / x of the cell edge )
    peg_deviation_border = 1/3  # must always be sub-unitary (the peg can't go closer than 1 / x of the edge)
    bulge_width_range = [1/12, 1/3]  # must always be sub-unitary (the width of the bulge from the center to one edge, (radius not diameter))
    bulge_height_range = [1/6, 1/3]  # must always be sub-unitary (the height of the peg)
    edge_dampening = 1 / 4  # must always be sub-unitary (dampen the effect of the bulge)

    if orientation == BorderOrientation.horizontal:

        curr_cell_x = cell_width * cell_col
        curr_cell_y = cell_height * cell_row + (
                    cell_height / 2)  # offset grid to have a full piece at the top (the grid is computed to have an edge as the center of the cell)

        if end_point is None:
            endpoint_lower_limit = cell_height * edge_deviation_border
            endpoint_upper_limit = cell_height - endpoint_lower_limit
            end_point = (
                curr_cell_x + cell_width,
                curr_cell_y + random.uniform(endpoint_lower_limit, endpoint_upper_limit)
            )

        peg_min_x = cell_width * peg_deviation_border
        peg_max_x = cell_width - peg_min_x

        control_points.append((start_point[0] + (cell_width * edge_dampening), start_point[1]))  # diminuate the effect on the beggining of the curve

        peg_x = curr_cell_x + random.uniform(peg_min_x, peg_max_x)
        peg_y = (start_point[1] + end_point[1]) / 2

        bulge_width_left = random.uniform(bulge_width_range[0], bulge_width_range[1])
        bulge_width_left *= cell_width
        bulge_width_right = random.uniform(bulge_width_range[0], bulge_width_range[1])
        bulge_width_right *= cell_width

        peg_height = random.uniform(bulge_height_range[0], bulge_height_range[1])
        peg_height *= cell_height
        if random.choice([True, False]):  # peg (True) or hole from the point of view of the bottom piece
            peg_apex = peg_y + peg_height
        else:
            peg_apex = peg_y - peg_height

        for _ in range(random.randint(2, 5)):  # curve sharpness / intensity
            control_points.append((peg_x, peg_y))

        for _ in range(random.randint(2, 5)):
            control_points.append((peg_x - bulge_width_left, peg_apex))

        for _ in range(2, 5):
            control_points.append((peg_x, peg_apex))

        for _ in range(random.randint(2, 5)):
            control_points.append((peg_x + bulge_width_right, peg_apex))

        for _ in range(random.randint(2, 5)):
            control_points.append((peg_x, peg_y))

        control_points.append((end_point[0] - (cell_width * edge_dampening), end_point[1]))  # make the end of the border more flat

        control_points.append(end_point)

    else:
        curr_cell_x = cell_width * cell_col + (cell_width / 2) # offset grid to have a full piece at the top (the grid is computed to have an edge as the center of the cell)
        curr_cell_y = cell_height * cell_row

        if end_point is None:
            endpoint_lower_limit = cell_width * edge_deviation_border
            endpoint_upper_limit = cell_width - endpoint_lower_limit
            end_point = (
                curr_cell_x + random.uniform(endpoint_lower_limit, endpoint_upper_limit),
                curr_cell_y + cell_height
            )

        peg_min_y = (cell_height * peg_deviation_border)
        peg_max_y = cell_height - peg_min_y

        control_points.append((start_point[0], start_point[1] + (cell_height * edge_dampening)))  # diminuate the curving at the edges

        peg_y = curr_cell_y + random.uniform(peg_min_y, peg_max_y)
        peg_x = (start_point[0] + end_point[0]) / 2

        bulge_width_up = random.uniform(bulge_width_range[0], bulge_width_range[1])
        bulge_width_up *= cell_height
        bulge_width_down = random.uniform(bulge_width_range[0], bulge_width_range[1])
        bulge_width_down *= cell_height

        peg_height = random.uniform(bulge_height_range[0], bulge_height_range[1])
        peg_height *= cell_width
        if random.choice([True, False]):
            peg_apex = peg_x + peg_height
        else:
            peg_apex = peg_x - peg_height

        for _ in range(random.randint(2, 5)):
            control_points.append(((start_point[0] + end_point[0])/2, peg_y))

        for _ in range(random.randint(2, 5)):
            control_points. append((peg_apex, peg_y - bulge_width_up))

        for _ in range(random.randint(2, 5)):
            control_points.append((peg_apex, peg_y))

        for _ in range(random.randint(2, 5)):
            control_points.append((peg_apex, peg_y + bulge_width_down))

        for _ in range(random.randint(2, 5)):
            control_points.append((peg_x, peg_y))

        control_points.append((end_point[0], end_point[1] - (cell_height * edge_dampening)))  # dimminuate the curving at the edges

        control_points.append(end_point)

    return control_points


def get_lines_from_control_points(control_points, resolution=25):
    ts = [t / resolution for t in range(resolution + 1)]
    border = make_bezier(control_points)
    points = border(ts)

    return points


def chk_too_close(line1, line2, delta, draw=None):
    for point1_idx in range(1, len(line1[1:-1])-1):
        for point2_idx in range(0, len(line2[0:])-1):
            if draw is not None and intersect(line1[point1_idx], line1[point1_idx + 1], line2[point2_idx], line2[point2_idx + 1]):
                average = ((line1[point1_idx][0] + line2[point2_idx][0] + line1[point1_idx + 1][0] + line2[point2_idx + 1][0]) / 4,
                           (line1[point1_idx][1] + line2[point2_idx][1] + line1[point1_idx + 1][1] + line2[point2_idx + 1][1]) / 4)
                draw.ellipse(((average[0] - 25, average[1] - 25), (average[0]+25, average[1]+25)), outline=1)
                return True

    for point1 in line1[2:-2]:
        if draw is not None:
            draw.ellipse(((point1[0]-2, point1[1]-2), (point1[0]+2, point1[1]+2)), outline=1)
        for point2 in line2[:]:
            if draw is not None:
                draw.ellipse(((point2[0] - 2, point2[1] - 2), (point2[0] + 2, point2[1] + 2)), outline=1)
            dist = math.sqrt(math.pow((point2[0] - point1[0]), 2) + math.pow((point2[1] - point1[1]), 2))
            if dist < delta:
                if draw is not None:
                    average = ((point1[0]+point2[0]) / 2, (point1[1]+point2[1]) / 2)
                    draw.ellipse(((average[0] - delta/2, average[1] - delta/2), (average[0]+delta/2, average[1] + delta/2)), outline=1)
                return True
    return False


def get_background_image(image_path: str, puzzle_size: [], background_to_puzzle_ratio=2):
    background_image = Image.open(image_path, mode='r')

    background_image_required_space = (int(puzzle_size[0] * background_to_puzzle_ratio),
                                       int(puzzle_size[1] * background_to_puzzle_ratio))

    if background_image.size[0] < background_image_required_space[0] or \
            background_image.size[1] < background_image_required_space[1]:

        raise BackgroundImageNotBigEnough(f"{background_image.size} < {background_image_required_space}")

    crop_x_offset = random.randint(0, background_image.size[0] - background_image_required_space[0])
    crop_y_offset = random.randint(0, background_image.size[1] - background_image_required_space[1])

    background_image = background_image.crop((crop_x_offset,
                                              crop_y_offset,
                                              crop_x_offset + background_image_required_space[0],
                                              crop_y_offset + background_image_required_space[1]))

    if random.choice([True, False]):
        background_image = ImageOps.flip(background_image)

    if random.choice([True, False]):
        background_image = ImageOps.mirror(background_image)

    return background_image


def get_horizontal_edge_line_points(puzzle_configs):
    horizontal_edge_line_points = []
    for i in range(puzzle_configs.NUM_ROWS - 1):
        horizontal_edge_line_points.append([])
        end_point = (0, puzzle_configs.ROW_HEIGHT * (i+1))
        for j in range(puzzle_configs.NUM_COLS):
            control_points = get_control_points(puzzle_configs.COL_WIDTH, puzzle_configs.ROW_HEIGHT, j, i, end_point)
            horizontal_edge_line_points[i].append(get_lines_from_control_points(control_points))
            end_point = control_points[-1]

    return horizontal_edge_line_points


def get_vertical_edge_line_points(horizontal_edge_line_points: [], puzzle_configs):
    vertical_edge_line_points = []
    for i in range(puzzle_configs.NUM_COLS - 1):
        vertical_edge_line_points.append([])
        end_point = (puzzle_configs.COL_WIDTH * (i + 1), 0)
        for j in range(puzzle_configs.NUM_ROWS):
            does_intersect = True
            while does_intersect:
                does_intersect = False
                control_points = get_control_points(puzzle_configs.COL_WIDTH, puzzle_configs.ROW_HEIGHT, i, j, end_point,
                                                    orientation=BorderOrientation.vertical)
                if len(vertical_edge_line_points[i]) <= j:
                    vertical_edge_line_points[i].append(get_lines_from_control_points(control_points))
                else:
                    vertical_edge_line_points[i][j] = get_lines_from_control_points(control_points)
                    if puzzle_configs.LOG_TO_SCREEN:
                        print(f"\rCorrected col {i}, row {j}")
                delta = min(puzzle_configs.COL_WIDTH, puzzle_configs.ROW_HEIGHT) / 10
                if j > 0:
                    does_intersect = does_intersect or chk_too_close(vertical_edge_line_points[i][j],
                                                                     horizontal_edge_line_points[j-1][i], delta)
                    does_intersect = does_intersect or chk_too_close(vertical_edge_line_points[i][j],
                                                                     horizontal_edge_line_points[j-1][i+1], delta)
                if j < puzzle_configs.NUM_ROWS - 1:
                    does_intersect = does_intersect or chk_too_close(vertical_edge_line_points[i][j],
                                                                     horizontal_edge_line_points[j][i], delta)
                    does_intersect = does_intersect or chk_too_close(vertical_edge_line_points[i][j],
                                                                     horizontal_edge_line_points[j][i + 1], delta)

            end_point = control_points[-1]

    return vertical_edge_line_points


def draw_puzzle_from_points(vertical_edge_line_points, horizontal_edge_line_points, puzzle_configs):
    puzzle_outline = Image.new('1', (puzzle_configs.IMG_WIDTH, puzzle_configs.IMG_HEIGHT), 0)
    draw = ImageDraw.Draw(puzzle_outline)

    for i in range(puzzle_configs.NUM_ROWS):
        for j in range(puzzle_configs.NUM_COLS):
            if j < puzzle_configs.NUM_COLS - 1:
                for k in range(len(vertical_edge_line_points[j][i]) - 1):
                    draw.line([vertical_edge_line_points[j][i][k], vertical_edge_line_points[j][i][k + 1]], fill=1, width=1)
            if i < puzzle_configs.NUM_ROWS - 1:
                for k in range(len(horizontal_edge_line_points[i][j]) - 1):
                    draw.line([horizontal_edge_line_points[i][j][k], horizontal_edge_line_points[i][j][k + 1]], fill=1, width=1)

    draw.line(((0, 0), (0, puzzle_configs.IMG_HEIGHT-1)), fill=1)
    draw.line(((0, 0), (puzzle_configs.IMG_WIDTH-1, 0)), fill=1)
    draw.line(((0, puzzle_configs.IMG_HEIGHT-1), (puzzle_configs.IMG_WIDTH-1, puzzle_configs.IMG_HEIGHT-1)), fill=1)
    draw.line(((puzzle_configs.IMG_WIDTH-1, 0), (puzzle_configs.IMG_WIDTH-1, puzzle_configs.IMG_HEIGHT-1)), fill=1)

    return puzzle_outline


def transform(x, y, matrix):
    (a, b, c, d, e, f) = matrix
    return a * x + b * y + c, d * x + e * y + f


def save_solution(solution, path):
    file_formatted_solution = {"vertical_matches": [], "horizontal_matches": []}
    for row in solution["vertical_matches"]:
        for col in row:
            file_formatted_solution['vertical_matches'].append((col[0], col[1]))
    for row in solution["horizontal_matches"]:
        for col in row:
            file_formatted_solution['horizontal_matches'].append((col[0], col[1]))

    solution_json = json.dumps(file_formatted_solution)
    with open(path, "w") as solution_file:
        solution_file.write(solution_json)


def get_rotation_matrix(size, rotation_angle):
    w, h = size
    rotn_center = (w / 2.0, h / 2.0)
    post_trans = (0, 0)

    matrix = [
        round(math.cos(math.radians(rotation_angle)), 15),
        round(math.sin(math.radians(rotation_angle)), 15),
        0.0,
        round(-math.sin(math.radians(rotation_angle)), 15),
        round(math.cos(math.radians(rotation_angle)), 15),
        0.0,
    ]

    matrix[2], matrix[5] = transform(
        -rotn_center[0] - post_trans[0], -rotn_center[1] - post_trans[1], matrix
    )
    matrix[2] += rotn_center[0]
    matrix[5] += rotn_center[1]
    return matrix


def get_crop_box_for_piece(row_nr, col_nr, puzzle_configs):
    crop_box = [0, 0, puzzle_configs.IMG_WIDTH, puzzle_configs.IMG_HEIGHT]
    if row_nr == 0:
        crop_box[1] = 0
        crop_box[3] = puzzle_configs.ROW_HEIGHT * 1.5
    elif row_nr == puzzle_configs.NUM_ROWS - 1:
        crop_box[1] = puzzle_configs.IMG_HEIGHT - (puzzle_configs.ROW_HEIGHT * 1.5)
        crop_box[3] = puzzle_configs.IMG_HEIGHT
    else:
        crop_box[1] = (row_nr - 0.5) * puzzle_configs.ROW_HEIGHT
        crop_box[3] = (row_nr + 1.5) * puzzle_configs.ROW_HEIGHT

    if col_nr == 0:
        crop_box[0] = 0
        crop_box[2] = puzzle_configs.COL_WIDTH * 1.5
    elif col_nr == puzzle_configs.NUM_COLS - 1:
        crop_box[0] = puzzle_configs.IMG_WIDTH - (puzzle_configs.COL_WIDTH * 1.5)
        crop_box[2] = puzzle_configs.IMG_WIDTH
    else:
        crop_box[0] = (col_nr - 0.5) * puzzle_configs.COL_WIDTH
        crop_box[2] = (col_nr + 1.5) * puzzle_configs.COL_WIDTH
    return crop_box


def iter_all_strings():
    for size in itertools.count(1):
        for s in itertools.product(ascii_lowercase, repeat=size):
            yield "".join(s)


def draw_solution_on_image(background_image_with_shapes, solution, puzzle_configs):
    d = ImageDraw.Draw(background_image_with_shapes)
    r = 20
    fnt = ImageFont.truetype("arial.ttf", 40)
    connection_generator = iter_all_strings()
    for row_nr, row in enumerate(solution['vertical_matches']):
        for col_nr, col in enumerate(row):
            connect = next(connection_generator)
            for notch in col:
                x, y = notch
                d.ellipse(((x - 2, y - 2), (x + r, y + r / 2)), fill="#FFFFFF", outline="#000000")
                d.text((x, y), text=f"{connect}", font=fnt, fill="#FFFFFF", stroke_width=2, stroke_fill="#000000")

    for row_nr, row in enumerate(solution['horizontal_matches']):
        for col_nr, col in enumerate(row):
            connect = next(connection_generator)
            for notch in col:
                x, y = notch
                d.rectangle(((x - 2, y - 2), (x + r, y + r / 2)), fill="#FFFFFF", outline="#000000")
                d.text((x, y), text=f"{connect}", font=fnt, fill="#FFFFFF", stroke_width=2, stroke_fill="#000000")

